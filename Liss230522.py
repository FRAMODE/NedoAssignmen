import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from Lissajous_dataset import Datadata
import datetime


conflr = 0.01
confoptim = "Adam"
confnumhidden = 64
confact = "tanh"

datatrain = Datadata()
train_dataloader = DataLoader(datatrain,batch_size=5,shuffle=False) 

"""print(datatrain[0][0])
d = datatrain[0][0].T
plt.plot(d[0],d[1],marker = '.')
plt.show()"""

#---------------------------------モデル定義---------------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hx = None


    def forward(self, x):
        
        #output = [] #x.shape = [49,2]
        hx = None
        """for i in range(len(x)):

            hx = self.rnn(x[i],hx)
            output.append(hx)"""
        #x += torch.randn_like(x) * 0.001
        self.hx = self.rnn(x, self.hx)
        output = self.hx
            
        #output = torch.stack(output)
      
        output = self.fc(output)

        #output = 2 * F.tanh(output)

        #print(output.shape)
        
        return output
    

net = RNN(2,confnumhidden,2,1)



criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=conflr)

#---------------------------------学習---------------------------------

net.train()
num_epoch = 10000

loss_record = []


for epoch in range(num_epoch):
    loss_sum = 0
    

    for inputs in train_dataloader: 
        #for i in range(len(inputs[0,0])):
        
        
        for i in range(len(inputs[0,0])):
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            if i == 0:
                net.hx = None
                
            #inputs[0,0,i] += torch.randn_like(inputs[0,0,i]) * noise
            #print(inputs[0,0])
            x = net(inputs[0,0,i])
            
            #print(x.shape)
            y = inputs[0,1,i]
                #+ torch.randn_like(inputs[0,1])
            
            
            loss = criterion(x, y)

            loss_sum += loss

        loss_sum /= len(inputs[0,0])
        loss_sum.backward()

        optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {loss_sum.item() / len(train_dataloader)}")

    loss_record.append(loss_sum.item() / len(train_dataloader))


#---------------------------------学習結果出力---------------------------------
    
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

#---------------------------------推論---------------------------------

predict = np.zeros((len(datatrain[0][0]), 2))
print(predict.shape)
input_mat = torch.tensor([1.,0.])
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

#---------------------------------推論結果出力---------------------------------

date = str(datetime.datetime.now())
date = date.split()
plt.plot(predict[0], predict[1] ,marker = '.')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.yscale('linear')
plt.savefig(f"LissajousFig/Lissajous_result_{date[0]}_{date[1]}.png")
plt.show()




