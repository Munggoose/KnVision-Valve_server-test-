# import torchvision.datasets as datasets
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class psc_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        
    
    
    else:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': True}
        shuffle = {'train': True, 'test': False}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(), ])

        dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
        
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader