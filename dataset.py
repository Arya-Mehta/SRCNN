
import torch
from torch.utils.data import Dataset
import h5py
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, train_file, transform=None):
        super(TrainDataset, self).__init__()
        self.train_file = train_file
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        with h5py.File(self.train_file, 'r') as f:
            return len(f['lr'][()])
    
    def __getitem__(self, idx):
        with h5py.File(self.train_file, 'r') as f:
            low_res = f['lr'][idx]
            high_res = f['hr'][idx]

        low_res = self.transform(low_res)
        high_res =self.transform(high_res)
        
        return low_res, high_res

class EvalDataset(Dataset):
    def __init__(self, eval_file, transform=None):
        super(EvalDataset, self).__init__()
        self.eval_file = eval_file
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        with h5py.File(self.eval_file, 'r') as f:
            return len(f['lr'][()])
    
    def __getitem__(self, idx):
        with h5py.File(self.eval_file, 'r') as f:
            low_res = f['lr'][idx]
            high_res = f['hr'][idx]

        low_res = self.transform(low_res)
        high_res =self.transform(high_res)
        
        return low_res, high_res


