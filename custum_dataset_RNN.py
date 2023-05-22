import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from IPython import embed


class Datadata(torch.utils.data.Dataset):
    def __init__(self):
        cir_list = []
        cycle = 5
        ten = 50
       
        angle = np.linspace(0, (cycle * ten -1)/(cycle * ten)*cycle*2*np.pi,cycle * ten)
        #angl = np.linspace(0, 149,150)

        #num = np.linspace(-50, 99, 150)
        cos_x = 2 * np.cos(angle) + 2.
        sin_y = 2 * np.sin(angle) + 2.

        points = np.array([cos_x, sin_y]).T
        xx = points
        yy = np.concatenate([points[1:], points[:1]])

        points = np.array([xx,yy])

        points = torch.tensor(points, dtype=torch.float32)
        
        cir_list.append(points)
        cir_list = torch.stack(cir_list)
        self.data =  cir_list
    
    
        

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        train = self.data[idx]
       
        
        return train
    
