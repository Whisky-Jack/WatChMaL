# hydra imports
import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader

# generic imports
import os

logger = logging.getLogger('train')

@hydra.main(config_path='config/', config_name='resnet_train')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    #TODO: is this needed
    #ngpus_per_node = torch.cuda.device_count()

    # TODO: reset this when dataloading debugged
    ngpus = len(config.gpu_list)
    
    # TODO: initialize process group env variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # TODO: reset >= when dataloading debugged
    if ngpus >= 1:
        print("Using multiprocessing")
        print("Requesting GPUs. GPU list : " + str(config.gpu_list))
        devids = ["cuda:{0}".format(x) for x in config.gpu_list]
        print("Using DistributedDataParallel on these devices: {}".format(devids))
        mp.spawn(main_worker_function, nprocs=ngpus, args=(ngpus, config))
    else:
        print("Only one gpu found")
        gpu = config.gpu_list[0]
        main_worker_function(gpu, ngpus, config)

def main_worker_function(gpu, ngpus_per_node, config):
    print("Running main worker on device: {}".format(gpu))
    # TODO: how should this interact with self.device
    torch.cuda.set_device(gpu)

    world_size = ngpus_per_node
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=gpu,
    )

    # Instantiate model and engine
    model = instantiate(config.model).to(gpu)

    # configure the device to be used for model training and inference
    if ngpus_per_node >= 1:
        print("Using DistributedDataParallel model")
        # if more than one gpu given, then we must be using multiprocessing
        # TODO: remove find_unused_parameters=True
        model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

    # Configure data loaders
    data_loaders = {}
    for task, task_config in config.tasks.items():
        if 'data_loaders' in task_config:
            for name, loader_config in task_config.data_loaders.items():
                data_loaders[name] = get_data_loader(**config.data, **loader_config, gpu=gpu, ngpus=ngpus_per_node)

    """
    train_loader = data_loaders["train"]
    print(type(train_loader))
    iter(train_loader)

    input("Press Enter to continue...")
    """

    # Instantiate the engine
    engine = instantiate(config.engine, model=model, gpu=gpu, data_loaders=data_loaders)
    
    # Configure optimizers
    # TODO: optimizers should be refactored into a dict probably
    for task, task_config in config.tasks.items():
        if 'optimizer' in task_config:
            # TODO: reconsider optimizer instantiation
            engine.configure_optimizers(task_config.optimizer)

    # Reload previous state
    if 'load_state' in config:
        engine.reload(config.load_model)

    # Perform tasks
    for task, task_config in config.tasks.items():
        getattr(engine, task)(task_config)
        # TODO: remove
        break

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()