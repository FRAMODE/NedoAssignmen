import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from IPython import embed


class Datadata(torch.utils.data.Dataset):
    def __init__(self):
        cir_list = []
        cycle = 1
        ten = 50
       
        angle = np.linspace(0, (cycle * ten -1)/(cycle * ten)*cycle*2*np.pi,cycle * ten)
        #angl = np.linspace(0, 149,150)

        #num = np.linspace(-50, 99, 150)
        x = np.sin(angle + 1/2 * np.pi)
        y = np.sin(2 * angle) 

        points = np.array([x, y]).T

        #points = np.array([x[:30], y[:30]]).T #test

        #xx = points[:29]
        #yy = points[1:]

        xx = points
        yy = np.concatenate([points[1:], points[:1]])
        
    

        points = np.array([xx,yy])

        points = torch.tensor(points, dtype=torch.float32)
        
        cir_list.append(points)
        cir_list = torch.stack(cir_list)
        #cir_list += torch.randn_like(cir_list) * 0.001
        self.data =  cir_list
    
    
        

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        train = self.data[idx]
       
        
        return train
    
