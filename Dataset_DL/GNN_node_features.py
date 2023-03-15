import os
import torch
os.environ['TORCH'] = torch.__version__
#print(torch.__version__)
import numpy as np
from random import shuffle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as rd 
from torch.nn import Softmax
from torch_geometric.data import Data


from torch.nn import Linear
from torch.nn import Softmax
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,n_hidden_convlayers):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        
        num_node_features = 4 #change
        output = 2
        self.n_hidden_convlayers = n_hidden_convlayers
        # edge embedding
        
        self.hidden_layers = nn.ModuleList()
        for k in range(n_hidden_convlayers):
            if k == 0:
                self.hidden_layers.append( GraphConv(num_node_features, hidden_channels))
            else:
                self.hidden_layers.append( GraphConv(hidden_channels, hidden_channels))

        # global embedding 
        self.lin5 = Linear(hidden_channels, hidden_channels)
        self.lin6 = Linear(hidden_channels, hidden_channels)
        self.lin7 = Linear(hidden_channels, hidden_channels)
        self.lin8 = Linear(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, output)


    def forward(self,node_attr,edge_index, batch):
        # 1. Obtain edge embeddings 
        for k in range(self.n_hidden_convlayers):
            node_attr = self.hidden_layers[k](node_attr,edge_index)
            node_attr = node_attr.relu()

        # 2. Readout layer
        node_attr = global_mean_pool(node_attr, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        node_attr = F.dropout(node_attr, p=0.5, training=self.training)
        node_attr = self.lin5(node_attr)
        node_attr = node_attr.relu()
        node_attr = self.lin6(node_attr)
        node_attr = node_attr.relu()
        node_attr = self.lin7(node_attr)
        node_attr = node_attr.relu()
        node_attr = self.lin8(node_attr)
        node_attr = node_attr.relu()
        node_attr = self.lin(node_attr)
        #node_attr = Softmax(dim=1)(node_attr)
        node_attr = torch.sigmoid(node_attr)
        
        return node_attr

from IPython.display import Javascript
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))


nb_leafs = 20
t_max = 300
model = GNN(hidden_channels=50, n_hidden_convlayers=int(0.4*(nb_leafs-1)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.L1Loss(reduction='sum')

import json

# Open the file and read the contents
with open('/workdir/chirraneso/dataset10k.txt', 'r') as file:
    contents = file.read()

# Parse the contents as JSON and convert it to a Python dictionary
data_dict = json.loads(contents)
n= len(data_dict['p0'])
print("n",n)
train_size = int(0.8*n)
print("train", train_size)
#shuffle index of the dataset 
index = [k for k in range(n)]
shuffle(index)

def train():
    model.train()

    for k in index[0:train_size]:  # Iterate in batches over the training dataset.
        p2,p0 = data_dict['p2'][k],data_dict['p0'][k]
        #simu = simulator(p0,p2, nb_leafs,t_max)
        if type(data_dict["dyck"][k]) != int : 
            values = data_dict["coo"][k][0]
            features = np.array(data_dict['node_features'][k])
            row = data_dict["coo"][k][1] 
            col = data_dict["coo"][k][2] 
            coo = torch.LongTensor([row,col])
            data = Data(edge_index = coo, y= torch.tensor([[p0/p2,p2-p0]]), num_nodes = 2*nb_leafs-1,edge_attr = torch.tensor([values+[0]]).T,  x=torch.tensor(np.vstack((features,np.zeros(4)))).float()) #[p0,1-(p2 +p0),p2]
            out = model(data.x,data.edge_index, data.batch) # Perform a single forward pass. #CHANGE
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

def test():
    model.eval()

    MSE =0
    error_delta = 0
    error_q =0
    error_p0 =0
    error_p2 = 0

    for k in index[train_size:n]:  # Iterate in batches over the training dataset.
        p2,p0 = data_dict['p2'][k],data_dict['p0'][k]
        #simu = simulator(p0,p2, nb_leafs,t_max)
        if type(data_dict["dyck"][k]) != int: 
            values = data_dict["coo"][k][0]
            features = np.array(data_dict['node_features'][k])
            row = data_dict["coo"][k][1] 
            col = data_dict["coo"][k][2] 
            coo = torch.LongTensor([row,col])
            data = Data(edge_index = coo, y= torch.tensor([[p0/p2,p2-p0]]), num_nodes = 2*nb_leafs-1, edge_attr = torch.tensor([values+[0]]).T, x=torch.tensor(np.vstack((features,np.zeros(4)))).float())
            out = model(data.x,data.edge_index, data.batch) # Perform a single forward pass. #CHANGE
            distance = criterion(out, data.y)  # Compute the loss.
            out = out[0]
            data.y = data.y[0]
            error_q += (out[0] -data.y[0])/data.y[0]
            error_delta += (out[1] -data.y[1])/data.y[1]
            error_p2 += (out[1]/(1-out[0]) - p2)/p2
            error_p0 += (out[0]*out[1]/(1-out[0]) - p0)/p0
            MSE += distance
    return MSE/(n-train_size), error_q/(n-train_size), error_delta/(n-train_size), error_p2/(n-train_size), error_p0/(n-train_size)


report = open("report_GNN_node_features.txt","w+")
report.write(f'leafs:{nb_leafs}\n')
report.write(f'tmax:{t_max}\n')
report.write(f'trainsize:{train_size}\n')

for epoch in range(1, 100):
    train()
    MSE, error_q, error_delta, error_p2, error_p0 = test()
    report.write(f'Epoch: {epoch:03d}, Test MSE: {MSE:.4f}, Test error q: {error_q:.4f},Test error delat: {error_delta:.4f},Test error p2: {error_p2:.4f}, Test error p0: {error_p0:.4f} \n')
    print(f'Epoch: {epoch:03d}, Test MSE: {MSE:.4f}, Test error q: {error_q:.4f},Test error delat: {error_delta:.4f},Test error p2: {error_p2:.4f}, Test error p0: {error_p0:.4f} ')

report.close()

torch.save(model,"model_GNN_nodefeat") #model = torch.load('model_DNN') to load
