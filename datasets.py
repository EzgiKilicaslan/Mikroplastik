import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision 
import numpy as np

num_workers_loader = 16

def get_classes(dataset):
    classes = [x[1] for x in np.array(dataset.dataset.imgs)[dataset.indices].tolist()]
    class_1 = classes.count('1')
    class_0 = classes.count('0')
    weights = [1./class_0, 1./class_1]
    samples_weight = [weights[int(t)] for t in classes]
    return samples_weight

class MicroplasticDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, folder: str = "micro_plastic_merged/"):
        super().__init__()
        self.batch_size = batch_size
        self.folder = folder

    def setup(self, stage=None):
        full_dataset = torchvision.datasets.ImageFolder(root = self.folder, transform = torchvision.transforms.Compose([torchvision.transforms.Resize((2048, 2048)), torchvision.transforms.ToTensor()]))
# target_transform=torchvision.transforms.Compose([lambda x:torch.LongTensor([x]), lambda x:F.one_hot(x,9)[0].type(torch.FloatTensor)])
        self.microplastic_train, self.microplastic_val, self.microplastic_test = random_split(full_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    # 36 none
    # 650
        
    def train_dataloader(self):
        return DataLoader(self.microplastic_train, batch_size=self.batch_size, num_workers=num_workers_loader, sampler=torch.utils.data.WeightedRandomSampler(get_classes(self.microplastic_train), len(self.microplastic_train)))

    def test_dataloader(self):
        return DataLoader(self.microplastic_test, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.microplastic_val, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False)