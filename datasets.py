import numpy as np
from torch.utils.data import Dataset
import pickle

class TrainDataset(Dataset):
    def __init__(self, pickle_file):
        super(TrainDataset, self).__init__()
        inff = open(pickle_file, "rb")
        self.file = pickle.load(inff)
        inff.close()
        self.len = len(self.file)

    def __getitem__(self, idx): return (np.expand_dims(self.file[idx][0], 0), self.file[idx][1])

    def __len__(self): return self.len

class EvalDataset(Dataset):
    def __init__(self, pickle_file):
        inff = open(pickle_file, "rb")
        self.file = pickle.load(inff)
        inff.close()
        self.len = len(self.file)

    def __getitem__(self, idx): return (np.expand_dims(self.file[idx][0][:, :], 0), self.file[idx][1])

    def __len__(self): return self.len