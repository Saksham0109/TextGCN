import torch
import torch.nn as nn
class TextGCN(nn.Module):
    def __init__(self,input_dim, hidden_dim1,hidden_dim2, output_dim,dropout):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.weight1=torch.nn.Parameter(torch.FloatTensor(768, hidden_dim1)).to(device)
        self.bias1=torch.nn.Parameter(torch.FloatTensor(hidden_dim1)).to(device)
        self.weight2=torch.nn.Parameter(torch.FloatTensor(hidden_dim1, hidden_dim2)).to(device)
        self.bias2=torch.nn.Parameter(torch.FloatTensor(hidden_dim2)).to(device)
        self.weight3=torch.nn.Parameter(torch.FloatTensor(hidden_dim2, output_dim)).to(device)
        self.bias3=torch.nn.Parameter(torch.FloatTensor(output_dim)).to(device)
        #Initialize weights
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.zeros_(self.bias2)
        torch.nn.init.zeros_(self.bias3)
        self.dropout=torch.nn.Dropout(dropout)
        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax(dim=1)
    def forward(self,adj,features):
        output=torch.mm(torch.mm(adj,features),self.weight1)+self.bias1
        output=self.relu(output)
        output=self.dropout(output)
        output=torch.mm(torch.mm(adj,output),self.weight2)+self.bias2
        output=self.relu(output)
        output=self.dropout(output)
        output=torch.mm(torch.mm(adj,output),self.weight3)+self.bias3
        output=self.softmax(output)
        return output
