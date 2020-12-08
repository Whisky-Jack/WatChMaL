# PyTorch imports
from torch.utils.data import Dataset
import h5py
import numpy as np
from abc import ABC, abstractmethod
import copy
import time

import torch.multiprocessing as mp

# TODO: remove flag when done debugging
solution = 1

# solution 0 -- single gpu default mode
# solution 1 -- load everything in initialization
# solution 2 -- memmap everything

class H5Dataset(Dataset, ABC):

    def __init__(self, h5_path, is_distributed, transforms=None):
        # TODO: remove
        print("initializing dataset")
        print(mp.get_context())
        self.h5_path = h5_path
        time0 = time.time()
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_length = h5_file["labels"].shape[0]

            hdf5_hit_pmt    = h5_file["hit_pmt"]
            hdf5_hit_time   = h5_file["hit_time"]
            hdf5_hit_charge = h5_file["hit_charge"]

            # initialize memmap param dict
            self.pmt_dict    = {'shape':hdf5_hit_pmt.shape,    'offset':hdf5_hit_pmt.id.get_offset(),   'dtype':hdf5_hit_pmt.dtype}
            self.time_dict   = {'shape':hdf5_hit_time.shape,   'offset':hdf5_hit_time.id.get_offset(),  'dtype':hdf5_hit_time.dtype}
            self.charge_dict = {'shape':hdf5_hit_charge.shape, 'offset':hdf5_hit_charge.id.get_offset(),'dtype':hdf5_hit_charge.dtype}
            
            # Load the contents which could fit easily into memory
            if solution == 0:
                self.labels           = np.array(h5_file["labels"])
                self.energies         = np.array(h5_file["energies"])
                self.positions        = np.array(h5_file["positions"])
                self.angles           = np.array(h5_file["angles"])
                self.root_files       = np.array(h5_file["root_files"])
                self.event_ids        = np.array(h5_file["event_ids"])
                self.event_hits_index = np.append(h5_file["event_hits_index"], self.pmt_dict['shape'][0]).astype(np.int64)

            if solution == 2:
                hdf5_hit_labels           = h5_file["labels"]
                hdf5_hit_energies         = h5_file["energies"]
                hdf5_hit_positions        = h5_file["positions"]
                hdf5_hit_angles           = h5_file["angles"]
                hdf5_hit_root_files       = h5_file["root_files"]
                hdf5_hit_event_ids        = h5_file["event_ids"]
                hdf5_hit_event_hits_index = h5_file["event_hits_index"]

                self.label_dict      = {'shape':hdf5_hit_labels.shape,       'offset':hdf5_hit_labels.id.get_offset(),       'dtype':hdf5_hit_labels.dtype}
                self.energies_dict   = {'shape':hdf5_hit_energies.shape,     'offset':hdf5_hit_energies.id.get_offset(),     'dtype':hdf5_hit_energies.dtype}
                self.positions_dict  = {'shape':hdf5_hit_positions.shape,    'offset':hdf5_hit_positions.id.get_offset(),    'dtype':hdf5_hit_positions.dtype}
                self.angles_dict     = {'shape':hdf5_hit_angles.shape,       'offset':hdf5_hit_angles.id.get_offset(),       'dtype':hdf5_hit_angles.dtype}
                self.root_files_dict = {'shape':hdf5_hit_root_files.shape,       'offset':hdf5_hit_root_files.id.get_offset(),       'dtype':hdf5_hit_root_files.dtype}
                self.event_ids_dict  = {'shape':hdf5_hit_event_ids.shape,    'offset':hdf5_hit_event_ids.id.get_offset(),    'dtype':hdf5_hit_event_ids.dtype}

                self.event_hits_index_dict = {'shape':hdf5_hit_event_hits_index.shape, 'offset':hdf5_hit_event_hits_index.id.get_offset(), 'dtype':hdf5_hit_event_hits_index.dtype}
                
            # TODO: uncomment
        
        # TODO: uncomment hanging fix
        #if not is_distributed:
        #    self.initialize()
        time1 = time.time()
        print(time1 - time0)

    def initialize(self):
        """
        memmaps must be instantiated this way for multiprocessing (memmaps can't be pickled)
        """
        # Create a memory map for event_data - loads event data into memory only on __getitem__()
        self.hit_pmt = np.memmap(self.h5_path, mode="r",
                                shape=self.pmt_dict['shape'],
                                offset=self.pmt_dict['offset'],
                                dtype=self.pmt_dict['dtype'])
        
        self.time = np.memmap(self.h5_path, mode="r",
                                shape=self.time_dict['shape'],
                                offset=self.time_dict['offset'],
                                dtype=self.time_dict['dtype'])

        self.charge = np.memmap(self.h5_path, mode="r",
                                shape=self.charge_dict['shape'],
                                offset=self.charge_dict['offset'],
                                dtype=self.charge_dict['dtype'])
        
        # Solution 1
        if solution == 1:
            with h5py.File(self.h5_path, 'r') as h5_file:
                self.labels           = np.array(h5_file["labels"])
                self.energies         = np.array(h5_file["energies"])
                self.positions        = np.array(h5_file["positions"])
                self.angles           = np.array(h5_file["angles"])
                self.root_files       = np.array(h5_file["root_files"])
                self.event_ids        = np.array(h5_file["event_ids"])
                self.event_hits_index = np.append(h5_file["event_hits_index"], self.pmt_dict['shape'][0]).astype(np.int64)
        
        # Solution 2
        if solution == 2:
            self.labels = np.memmap(self.h5_path, mode="r",
                                    shape=self.label_dict['shape'],
                                    offset=self.label_dict['offset'],
                                    dtype=self.label_dict['dtype'])

            self.energies = np.memmap(self.h5_path, mode="r",
                                    shape=self.energies_dict['shape'],
                                    offset=self.energies_dict['offset'],
                                    dtype=self.energies_dict['dtype'])

            self.positions = np.memmap(self.h5_path, mode="r",
                                    shape=self.positions_dict['shape'],
                                    offset=self.positions_dict['offset'],
                                    dtype=self.positions_dict['dtype'])

            self.angles = np.memmap(self.h5_path, mode="r",
                                    shape=self.angles_dict['shape'],
                                    offset=self.angles_dict['offset'],
                                    dtype=self.angles_dict['dtype'])
            
            self.event_ids = np.memmap(self.h5_path, mode="r",
                                    shape=self.event_ids_dict['shape'],
                                    offset=self.event_ids_dict['offset'],
                                    dtype=self.event_ids_dict['dtype'])

            self.root_files = np.memmap(self.h5_path, mode="r",
                                    shape=self.root_files_dict['shape'],
                                    offset=self.root_files_dict['offset'],
                                    dtype=self.root_files_dict['dtype'])
            
            self.event_hits_index = np.memmap(self.h5_path, mode="r",
                                    shape=self.event_hits_index_dict['shape'],
                                    offset=self.event_hits_index_dict['offset'],
                                    dtype=self.event_hits_index_dict['dtype'])

        # Create attribute so that method won't be invoked again
        self.initialized = True

    @abstractmethod
    def get_data(self, hit_pmts, hit_charges, hit_times):
        pass

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        if not hasattr(self, 'initialized'):
            self.initialize()
        
        start = self.event_hits_index[item]
        stop = self.event_hits_index[item + 1]

        hit_pmts = self.hit_pmt[start:stop].astype(np.int16)
        hit_charges = self.charge[start:stop]
        hit_times = self.time[start:stop]

        data = self.get_data(hit_pmts, hit_charges, hit_times)

        if solution == 2:
            #print("item: ", item)
            data_dict = {
                "data": data,
                "labels": self.labels[item],
                "energies": copy.deepcopy(self.energies[item]),
                "angles": copy.deepcopy(self.angles[item]),
                "positions": copy.deepcopy(self.positions[item]),
                #"root_files": copy.deepcopy(self.root_files[item]),
                "event_ids": copy.deepcopy(self.event_ids[item]),
                "indices": item
            }
        else:
            data_dict = {
            "data": data,
            "labels": self.labels[item],
            "energies": self.energies[item],
            "angles": self.angles[item],
            "positions": self.positions[item],
            #"root_files": self.root_files[item],
            "event_ids": self.event_ids[item],
            "indices": item
        }

        return data_dict
