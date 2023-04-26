import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from IPython import embed


class Datadata(torch.utils.data.Dataset):
    def __init__(self):
        cir_list = []
        angle = np.linspace(0, 2*np.pi,50)
        num = np.linspace(0, 49, 50)
        cos_x = np.cos(angle) 
        sin_y = np.sin(angle) 
        points = np.array([cos_x, sin_y]).T
        points = np.array([points[:49],points[1:]])
        # train_points = np.array([points[0][:49],points[1][:49]])
        # train_points = train_points.T
        # train_points = 
        # target_points = np.array([points[0][1:],points[1][1:]])
        # points = np.array([train_points, target_points])
        points = torch.tensor(points, dtype=torch.float32)
        
        cir_list.append(points)
        cir_list = torch.stack(cir_list)
        self.data =  cir_list
    
    
        

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        train = self.data[idx]
       
        
        return train
    
