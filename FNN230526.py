import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from custum_dataset_RNN import Datadata
import datetime


conflr = 0.01
confoptim = "Adam"
confnumhidden = 64
confact = "tanh"

datatrain = Datadata()
train_dataloader = DataLoader(datatrain,batch_size=5,shuffle=False) 

'''print(datatrain[0][0])
d = datatrain[0][0].T
plt.plot(d[0],d[1],marker = '.')
plt.show()'''


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(FNN, self).__init__()
        self.le1 = nn.Linear(input_size, hidden_size)
        self.le2 = nn.Linear(hidden_size, hidden_size)
        self.le3 = nn.Linear(hidden_size, hidden_size)
        self.le4 = nn.Linear(hidden_size, hidden_size)
        self.le5 = nn.Linear(hidden_size, output_size)
        


    def forward(self, x):
        x = F.tanh(self.le1(x))
        x = F.tanh(self.le2(x))
        x = F.tanh(self.le3(x))
        x = F.tanh(self.le4(x))
        x = self.le5(x)

        return x
        
        
    

net = FNN(2,confnumhidden,2,1)



criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=conflr)

net.train()
num_epoch = 10000

loss_record = []

ssscounter = 0
noise = 0.0001

for epoch in range(num_epoch):
    loss_sum = 0
    

    for inputs in train_dataloader: 
        #for i in range(len(inputs[0,0])):
        
        
        for i in range(len(inputs[0,0])):
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            #inputs[0,0,i] += torch.randn_like(inputs[0,0,i]) * 0.001
            #print(inputs[0,0])
            x = net(inputs[0,0,i]) + torch.randn_like(inputs[0,0,i]) * noise
            
            
            #print(x.shape)
            y = inputs[0,1,i]
                #+ torch.randn_like(inputs[0,1])
            
            
            loss = criterion(x, y)

            loss_sum += loss
            if i > 1000:
                if(loss_record[i-2] - loss_record[i-1] < 0):   #ノイズ減らす
                    ssscounter += 1
                    if ssscounter == 10:
                        noise /= 2
                        ssscounter = 0

        loss_sum /= len(inputs[0,0])
        loss_sum.backward()

        optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {loss_sum.item() / len(train_dataloader)}")

    loss_record.append(loss_sum.item() / len(train_dataloader))

    
s = np.linspace(0, num_epoch, num_epoch)
#s = s.reshape(1,num_epoch)
loss_record = np.array(loss_record)
print(s.shape)
print(loss_record.shape)
plt.yscale('log')
plt.plot(s,loss_record)
#plt.savefig(f'RNNfig/datarand0001_loss_{confnumhidden}_{confoptim}_{conflr}_{confact}_tanh.png')
plt.show()


torch.save(net, 'model_weight.pth')
model = torch.load('model_weight.pth')

predict = np.zeros((len(datatrain[0][0]), 2))
print(predict.shape)
input_mat = torch.tensor([4.,2.])
#input_mat = input_mat.unsqueeze(0)
print(datatrain[0,0,1])

for i in range (len(datatrain[0][0])):
    
    #print(input_mat.shape)
    #print(f"aaaaaaaa {datatrain[0][0].shape}")
    output = model(input_mat)
    #print(input_mat.shape)
    #output = model(datatrain[0,0,i])
    
    #print(datatrain[0][0][i].shape)
    #print(output.shape)

   
    predict[i] = output.detach().numpy()
    #print(output.shape)

    input_mat = output


print(predict)

predict = predict.T

print(predict.shape)

date = str(datetime.datetime.now())
date = date.split()

plt.plot(predict[0],predict[1],marker = '.')
plt.xlim(0,4)
plt.ylim(0,4)
plt.yscale('linear')
plt.savefig(f"FNNfig/FNN_result_{datatrain.cycle}_{date[0]}_{date[1]}.png")
plt.show()




