import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import instantiate
import numpy as np
from watchmal.dataset.samplers import DistributedSamplerWrapper


from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

def get_data_loader(dataset, batch_size, sampler, num_workers, is_distributed, seed, split_path=None, split_key=None, transforms=None):
    
    split_indices = np.load(split_path, allow_pickle=True)[split_key]
    
    sampler = SubsetRandomSampler(split_indices)
    dataset = instantiate(dataset, transforms=transforms, is_distributed=is_distributed)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)

    """

def get_data_loader(dataset, batch_size, sampler, num_workers, is_distributed, seed, split_path=None, split_key=None, transforms=None):
    # TODO: reset this entire section
    
    dataset = instantiate(dataset, transforms=transforms, is_distributed=is_distributed)
    
    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = instantiate(sampler)
    
    
    # TODO: uncomment
    
    if is_distributed:
        ngpus = torch.distributed.get_world_size()
        # TODO: return batch_sizing
        #batch_size = int(batch_size/ngpus)
        
        # TODO: revert back after testing
        sampler = DistributedSampler(dataset)
        #sampler = DistributedSamplerWrapper(sampler=sampler, dataset=dataset, seed=seed)
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    """