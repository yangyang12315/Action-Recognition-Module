import torch
from torch import nn
input_var = torch.randn((8,2048,9,1,1))
def conv1x1(in_channels,out_channels):
    return nn.Conv3d(in_channels = in_channels,out_channels = out_channels,kernel_size = 1,stride = 1,padding =0) 

class NonLocalBlock(nn.Module):
    def __init__(self,input_dims):
        super(NonLocalBlock, self).__init__()
        self.theta = conv1x1(input_dims, input_dims//4)
        self.phalt = conv1x1(input_dims, input_dims//4)
        self.glob = conv1x1(input_dims, input_dims//4)
        self.up   = conv1x1(input_dims//4, input_dims)
        self.bn1 = nn.BatchNorm3d(input_dims//4, affine=False)
        self.bn2 = nn.BatchNorm3d(input_dims//4, affine=False)
        self.bn3 = nn.BatchNorm3d(input_dims//4, affine=False)
    def forward(self, maps):
        b,c,t,h,w = maps.shape
        theta_maps = self.theta(maps) #(b,c/2,t,h,w)
        #theta_maps = nn.BatchNorm3d(theta_maps)
        theta_maps = self.bn1(theta_maps)
        phalt_maps = self.phalt(maps) #(b,c/2,t,h,w)
        phalt_maps = self.bn2(phalt_maps)
        global_maps = self.glob(maps) #(b,c,t,h,w)
        global_maps = self.bn3(global_maps)
        """ inter_channel=512 """
        inter_channel = int(c/4) 
        theta_x = theta_maps.view(b,inter_channel,-1).permute(0,2,1)
        phalt_x = theta_maps.view(b,inter_channel,-1)
        relation= torch.matmul(theta_x, phalt_x)#(b,thw*twh)=(8,9,9)
        relation_normalize = torch.nn.functional.softmax(relation/relation.size(-1),dim=-1)

        #relation_normalize = relation/relation.size(-1)
        #print(relation_normalize.shape)
        global_x = global_maps.view(b,inter_channel,-1).permute(0,2,1)
        y = torch.matmul(relation_normalize, global_x).permute(0,2,1).contiguous()
        y = y.view(b,c//4,*maps.size()[2:])
        out = self.up(y) + maps
        print ('maps',maps[0,:5,0,0,0])
        print ('out',out[0,:5,0,0,0])
        
        return out
if __name__ == '__main__':
    input_var = torch.autograd.Variable(input_var)
    net = NonLocalBlock(2048)
    out = net(input_var)
    print ('out',out.size)