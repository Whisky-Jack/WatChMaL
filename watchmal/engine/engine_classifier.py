# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# hydra imports
from hydra.utils import instantiate

# generic imports
from math import floor, ceil
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys
from sys import stdout

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData

#extraneous testing imports

class ClassifierEngine:
    def __init__(self, model, rank, gpu, data_loaders, dump_path):
        # create the directory for saving the log and dump files
        self.dirpath = dump_path

        self.rank = rank

        self.model = model

        self.device = torch.device(gpu)

        # Setup the parameters to save given the model type
        if isinstance(self.model, DDP):
            self.is_distributed = True
            self.model_accs = self.model.module
            self.ngpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.model_accs = self.model
        
        self.data_loaders = data_loaders

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # define the placeholder attributes
        self.data      = None
        self.labels    = None
        self.energies  = None
        self.eventids  = None
        self.rootfiles = None
        self.angles    = None
        self.event_ids = None
        
        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")
    
    def configure_optimizers(self, optimizer_config):
        """
        Inspired by pytorch lightning approach
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_data_loaders(self, data_config, loaders_config):
        print("Configuring dataloaders")
        """
        for name, loader_config in loaders_config.items():
            print(name)
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config)
        """

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data.

        Parameters:
            train = whether to compute gradients for backpropagation
            self should have attributes model, criterion, softmax, data, label
        
        Returns : a dict of loss, predicted labels, softmax, accuracy, and raw model outputs
        """

        with torch.set_grad_enabled(train):
            # move the data and the labels to the GPU (if using CPU this has no effect)
            self.data = self.data.to(self.device)
            self.labels = self.labels.to(self.device)

            model_out = self.model(self.data)

            self.loss = self.criterion(model_out, self.labels)
            
            softmax          = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out,dim=-1)
            accuracy         = (predicted_labels == self.labels).sum().item() / float(predicted_labels.nelement())
            

        # TODO: fixed calls to cpu() and detach() in loss
        return {'loss'             : self.loss.cpu().detach().item(),
                'predicted_labels' : predicted_labels.detach().cpu().numpy(),
                'softmax'          : softmax.detach().cpu().numpy(),
                'accuracy'         : accuracy,
                'raw_pred_labels'  : model_out}
    
    def backward(self):
        self.optimizer.zero_grad()  # reset accumulated gradient
        # TODO: added contiguous
        #self.loss.contiguous()
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params
    
    # ========================================================================

    def train(self, train_config):
        """
        Train the model on the training set.
        
        Parameters : None
        
        Outputs : 
        TODO: fix training outputs
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """

        # initialize training params
        epochs          = train_config.epochs
        report_interval = train_config.report_interval
        val_interval    = train_config.val_interval
        num_val_batches = train_config.num_val_batches
        checkpointing   = train_config.checkpointing

        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        epoch = 0.
        self.iteration = 0

        # keep track of the validation accuracy
        best_val_acc = 0.0
        best_val_loss = 1.0e6

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # global training loop for multiple epochs
        while (floor(epoch) < epochs):
            if self.rank == 0:
                print('Epoch',floor(epoch), 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            
            times = []

            start_time = time()

            train_loader = self.data_loaders["train"]

            # TODO: kind of ugly control flow for distributed behaviour
            # local training loop for batches in a single epoch
            for i, train_data in enumerate(self.data_loaders["train"]):
                
                # run validation on given intervals
                # TODO: verify that validation should only run on rank 0
                if self.iteration % val_interval == 0:
                    # set model to eval mode
                    self.model.eval()

                    val_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": 0., "accuracy": 0., "saved_best": 0}

                    # TODO: restore validation functionality
                    for val_batch in range(num_val_batches):
                        try:
                            val_data = next(val_iter)
                        except StopIteration:
                            # TODO: may need to call set_epoch on val_loader here
                            val_iter = iter(self.data_loaders["validation"])
                        
                        # extract the event data from the input data tuple
                        self.data      = val_data['data'].float()
                        self.labels    = val_data['labels'].long()
                        self.energies  = val_data['energies'].float()
                        self.angles    = val_data['angles'].float()
                        self.event_ids = val_data['event_ids'].float()

                        val_res = self.forward(False)
                        
                        val_metrics["loss"] += val_res["loss"]
                        val_metrics["accuracy"] += val_res["accuracy"]
                    
                    
                    # return model to training mode
                    self.model.train()

                    # record the validation stats to the csv
                    val_metrics["loss"] /= num_val_batches
                    val_metrics["accuracy"] /= num_val_batches

                    if self.is_distributed:
                        local_val_metrics = torch.tensor([val_metrics["loss"], val_metrics["accuracy"]]).to(self.device)
                        global_val_metrics = [torch.zeros_like(local_val_metrics).to(self.device) for i in range(self.ngpus)]
                        torch.distributed.all_gather(global_val_metrics, local_val_metrics)

                    # TODO: rework local_rank
                    if self.rank == 0:
                        # Save if this is the best model so far
                        combined_val_metrics = np.array(torch.stack(global_val_metrics).cpu())

                        global_val_loss = np.mean(combined_val_metrics[:, 0])
                        global_val_accuracy = np.mean(combined_val_metrics[:, 1])

                        val_metrics["loss"] = global_val_loss
                        val_metrics["accuracy"] = global_val_accuracy

                        if val_metrics["loss"] < best_val_loss:
                            print('best validation loss so far!: {}'.format(best_val_loss))
                            self.save_state(best=True)
                            val_metrics["saved_best"] = 1

                            best_val_loss = val_metrics["loss"]

                        # Save the latest model
                        if checkpointing:
                            self.save_state(best=False)
                                        
                        self.val_log.record(val_metrics)
                        self.val_log.write()
                        #TODO: Removed flush
                        #self.val_log.flush()
                
                # Train on batch
                self.data      = train_data['data'].float()
                self.labels    = train_data['labels'].long()
                self.energies  = train_data['energies'].float()
                self.angles    = train_data['angles'].float()
                self.event_ids = train_data['event_ids'].float()

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                epoch          += 1./len(self.data_loaders["train"])
                self.iteration += 1

                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": res["loss"], "accuracy": res["accuracy"]}
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                #TODO: Removed flush
                #self.train_log.flush()
                
                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f" %
                          (self.iteration, epoch, res["loss"], res["accuracy"]))
                
                if epoch >= epochs:
                    break
        
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()
    
    def get_synchronized_metrics(self, metric_dict):
        global_metric_dict = {}
        for tensor_name, tensor in zip(metric_dict.keys(), metric_dict.values()):
            print(tensor_name)
            global_tensor = [torch.zeros_like(tensor).to(self.device) for i in range(self.ngpus)]
            torch.distributed.all_gather(global_tensor, tensor)
            global_metric_dict[tensor_name] = torch.cat(global_tensor)
        
        return global_metric_dict
    
    def evaluate(self, test_config):
        """
        Evaluate the performance of the trained model on the validation set.
        
        Parameters : None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
        """
        test_dict = {"test_0":torch.ones(1).to(self.device), "test_1":torch.zeros(1).to(self.device)}
        if self.rank == 1:
            test_dict["test_0"] *= 2
            test_dict["test_1"] += self.rank
        
        global_dict = self.get_synchronized_metrics(test_dict)

        if self.rank == 0:
            print(global_dict["test_0"])
            print(global_dict["test_1"])
        """
        print("evaluating in directory: ", self.dirpath)
        
        # Variables to output at the end
        eval_loss = 0.0
        eval_acc = 0.0
        eval_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, indices, labels, predictions, softmaxes= [],[],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            for it, eval_data in enumerate(self.data_loaders["test"]):

                self.data = eval_data['data'].float()
                self.labels = eval_data['labels'].long()

                # Run the forward procedure and output the result
                result = self.forward(False)

                eval_loss += result['loss']
                eval_acc += result['accuracy']
                
                # Copy the tensors back to the CPU
                self.labels = self.labels.to("cpu")
                eval_indices = eval_data['indices'].long().to("cpu")
                
                # Add the local result to the final result
                indices.extend(eval_indices)
                labels.extend(self.labels)
                predictions.extend(result['predicted_labels'])
                softmaxes.extend(result["softmax"])

                print("eval_iteration : " + str(it) + " eval_loss : " + str(result["loss"]) + " eval_accuracy : " + str(result["accuracy"]))

                eval_iterations += 1
                # TODO: remove when debugging finished
                break
        
        # convert arrays to torch tensors
        print("loss : " + str(eval_loss/eval_iterations) + " accuracy : " + str(eval_acc/eval_iterations))

        local_eval_metrics_dict = {"iterations": torch.tensor([eval_iterations]).to(self.device),
                                "loss": torch.tensor([eval_loss]).to(self.device),
                                "eval_acc": torch.tensor([eval_acc]).to(self.device)}
        
        local_eval_results_dict = {"indices":torch.tensor(np.array(indices)).to(self.device),
                                "labels":torch.tensor(np.array(labels)).to(self.device),
                                "predictions":torch.tensor(np.array(predictions)).to(self.device),
                                "softmaxes":torch.tensor(np.array(softmaxes)).to(self.device)}

        if self.is_distributed:

            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                print(global_eval_metrics_dict)
        """
            if self.rank == 0:
                val_metrics = np.array(torch.stack(all_val_metrics).cpu())
                
                indices     = np.array(torch.cat(all_indices).cpu())
                labels      = np.array(torch.cat(all_labels).cpu())
                predictions = np.array(torch.cat(all_predictions).cpu())
                softmaxes   = np.array(torch.cat(all_softmaxes).cpu())
        else:
            val_metrics = np.array(torch.stack([local_val_metrics]).cpu())
            
            indices     = np.array(local_indices.cpu())
            labels      = np.array(local_labels.cpu())
            predictions = np.array(local_predictions.cpu())
            softmaxes   = np.array(local_softmaxes.cpu())

        if self.rank == 0:
            print("Sorting Outputs...")
            print("Indices shape: ", indices.shape)
            sorted_indices = np.argsort(indices)

            np.save(self.dirpath + "indices.npy", sorted_indices)

            print("Saving Data...")
            np.save(self.dirpath + "labels.npy", labels[sorted_indices])
            np.save(self.dirpath + "predictions.npy", predictions[sorted_indices])
            np.save(self.dirpath + "softmax.npy", softmaxes[sorted_indices])

            ######################################
            print(indices[sorted_indices][0:50])
            print(predictions[sorted_indices][0:50])
            ######################################

            val_loss = np.sum(val_metrics[:, 0])
            val_acc = np.sum(val_metrics[:, 1])
            val_iterations = np.sum(val_metrics[:, 2])

            print("\nAvg eval loss : " + str(val_loss/val_iterations),
                "\nAvg eval acc : " + str(val_acc/val_iterations))
        """
        
        
    # ========================================================================

    def restore_state(self, weight_file):
        """
        Restore model using weights stored from a previous run.
        
        Parameters : weight_file
        
        Outputs : 
            
        Returns : None
        """
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)

            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            
            # load network weights
            self.model_accs.load_state_dict(checkpoint['state_dict'])
            
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # load iteration count
            self.iteration = checkpoint['global_step']
        
        print('Restoration complete.')
    
    def save_state(self,best=False):
        """
        Save model weights to a file.
        
        Parameters : best
        
        Outputs : 
            
        Returns : filename
        """
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     ("BEST" if best else ""),
                                     ".pth")
        
        # Save model state dict in appropriate from depending on number of gpus
        model_dict = self.model_accs.state_dict()
        
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': model_dict
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename
