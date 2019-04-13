import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import math
import pdb
import os


# define GCN
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)#2048 2048
        self.gc2 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nfeat, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.BN_layer1 = nn.BatchNorm1d(nhid,affine=False)
        self.BN_layer2 = nn.BatchNorm1d(nhid,affine=False)
        self.BN_layer3 = nn.BatchNorm1d(nhid,affine=False)
    def forward(self, x, adj, frames, feature_dim):
        x = F.relu(self.gc1(x, adj))
        x = x.view(-1,feature_dim)
        x = self.BN_layer1(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = x.view(-1,frames,feature_dim)
        x = F.relu(self.gc2(x, adj))
        x = x.view(-1,2048)
        x = self.BN_layer2(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = x.view(-1,frames,feature_dim)
        x = F.relu(self.gc3(x, adj))
        return x

      

        
#define GCN layer
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
               
               
class GCN_apply(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class,batch_size):
        super(GCN_apply, self).__init__()
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        
        self.fc_att1 = nn.Linear(img_feature_dim,img_feature_dim)
        self.fc_att2 = nn.Linear(img_feature_dim,img_feature_dim)
        self.fc_W = nn.Linear(img_feature_dim,img_feature_dim)
        
        #nn.init.normal(self.fc_W.weight.data, 0, 0.01)
        #nn.init.constant(self.fc_W.bias.data, 0.01)
        self.gcn = GCN(nfeat=img_feature_dim,nhid=img_feature_dim,nclass=img_feature_dim,dropout=0.5)
        #self.classifier = self.fc_fusion()

    #graph define: f = G*X*W G:relation node X:input W  output.shape = input.shape
    #  G = x'x = H[batch,9,9] G[b,9,9] W[2048,2048]
    def forward(self, input):
        # batch,9,2048
        input = input.cuda()
        x = input.view(input.size(0), self.num_frames,self.img_feature_dim)
        # get [8,9,9]
        graph_H = torch.matmul(self.fc_att1(x)[:],self.fc_att2(x).permute(0,2,1)[:])  #[8,9,2048] 
        batch_graph_H = torch.nn.functional.softmax(graph_H,dim=2) #node value softmax
        x = self.gcn(input,batch_graph_H,self.num_frames,self.img_feature_dim)
        return x


if __name__ == "__main__":
	batch_size = 8
	num_frames = 9
	num_class =21
	img_feature_dim = 2048
	input_var = Variable(torch.randn(batch_size,num_frames,img_feature_dim))
	model = GCN_apply(img_feature_dim,num_frames,num_class,batch_size)
	model = model.cuda()
	output = model(input_var)
	print ('output.shape',output.shape)