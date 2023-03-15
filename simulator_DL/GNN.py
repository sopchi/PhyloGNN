import os
import torch
os.environ['TORCH'] = torch.__version__
#print(torch.__version__)
import numpy as np

import julia
julia.install()

import julia
j = julia.Julia()
simulator = j.include("simulator.jl")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as rd 
from torch.nn import Softmax
from torch_geometric.data import Data

def couple():
    y = rd.uniform(0,0.5) #p0
    x = rd.uniform(0,1)#p2
    while x+y > 1 or y >= x :
        y = rd.uniform(0,0.5) #p0
        x = rd.uniform(0,1)  #p2
    return (x,y)

from torch.nn import Linear
from torch.nn import Softmax
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        
        num_edge_features = 1
        output = 2
        # edge embedding
        
        self.conv1 = GraphConv(num_edge_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)

        # global embedding 
        self.lin5 = Linear(hidden_channels, hidden_channels)
        self.lin6 = Linear(hidden_channels, hidden_channels)
        self.lin7 = Linear(hidden_channels, hidden_channels)
        self.lin8 = Linear(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, output)


    def forward(self,edge_attr,edge_index, batch):
        # 1. Obtain edge embeddings 
        edge_attr = self.conv1(edge_attr, edge_index)
        edge_attr = edge_attr.relu()
        edge_attr = self.conv2(edge_attr, edge_index)
        edge_attr = edge_attr.relu()
        edge_attr = self.conv3(edge_attr, edge_index)
        edge_attr = edge_attr.relu()
        edge_attr = self.conv4(edge_attr, edge_index)

        # 2. Readout layer
        edge_attr = global_mean_pool(edge_attr, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        edge_attr = F.dropout(edge_attr, p=0.5, training=self.training)
        edge_attr = self.lin5(edge_attr)
        edge_attr = self.lin6(edge_attr)
        edge_attr = self.lin7(edge_attr)
        edge_attr = self.lin8(edge_attr)
        edge_attr = self.lin(edge_attr)
        #edge_attr = Softmax(dim=1)(edge_attr)
        edge_attr = torch.sigmoid(edge_attr)
        
        return edge_attr

from IPython.display import Javascript
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GNN(hidden_channels=50)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.L1Loss(reduction='sum')
nb_leafs = 20
t_max = 300
n=1000

def train():
    model.train()

    for k in range(n):  # Iterate in batches over the training dataset.
        p2,p0 = couple()
        simu = simulator(p0,p2, nb_leafs,t_max)
        if type(simu["dyck"]) != int : 
            values = simu["coo"][0] 
            row = simu["coo"][1] 
            col = simu["coo"][2] 
            coo = torch.LongTensor([row,col])
            data = Data(edge_index = coo, y= torch.tensor([[p0/p2,p2-p0]]), num_nodes = 2*nb_leafs+1, edge_attr=torch.tensor([values+[0]]).T) #[p0,1-(p2 +p0),p2]
            out = model(data.edge_attr,data.edge_index, data.batch) # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

def test():
    model.eval()

    MSE =0
    for k in range(n):  # Iterate in batches over the training dataset.
        p2,p0 = couple()
        simu = simulator(p0,p2, nb_leafs,t_max)
        if type(simu["dyck"]) != int: 
            values = simu["coo"][0] 
            row = simu["coo"][1] 
            col = simu["coo"][2] 
            coo = torch.LongTensor([row,col])
            data = Data(edge_index = coo, y= torch.tensor([[p0/p2,p2-p0]]), num_nodes = 2*nb_leafs+1, edge_attr=torch.tensor([values+[0]]).T) 
            out = model(data.edge_attr,data.edge_index, data.batch) # Perform a single forward pass.
            distance = criterion(out, data.y)  # Compute the loss.
            MSE += distance
    return MSE/n


report = open("report_GNN.txt","w+")
report.write(f'leafs:{nb_leafs}\n')
report.write(f'tmax:{t_max}\n')

for epoch in range(1, 50):
    train()
    test_acc = test()
    report.write(f'Epoch: {epoch:03d}, Test MSE: {test_acc:.4f} \n')
    print(f'Epoch: {epoch:03d}, Test MSE: {test_acc:.4f}')

report.close()

torch.save(model,"model_GNN") #model = torch.load('model_DNN') to load
