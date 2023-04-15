import torch
import torchvision
import torchvision.transforms as transforms

class Datadata(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=True)
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)