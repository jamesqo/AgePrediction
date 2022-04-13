import torch
import torch.nn as nn
import numpy as np

import math

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class GlobalAttention(nn.Module):
    def __init__(self, 
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class transformer_block(nn.Module):

    def __init__(self, dim, 
                 num_heads, 
                 mlp_ratio=4.,
                 drop=0., 
                 attn_drop=0.,
                 act_layer=GELU, 
                 norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = GlobalAttention(
                 transformer_num_heads=num_heads,
                 hidden_size=dim,
                 transformer_dropout_rate=attn_drop)
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(in_features=dim, 
                       hidden_features=mlp_hidden_dim, 
                       out_features=dim,
                       act_layer=act_layer, 
                       drop=drop)

    def forward(self, xquery,xcontext):
        xq = self.norm1(xquery)
        xc = self.norm2(xcontext)
        xres = self.attn(xq,xc)
        xres = xres+xq
        xquery = self.mlp(self.norm3(xres))+xquery
        return xquery
    

class Transformer(nn.Module):
    def __init__(self, 
                 in_dim, 
                 depth, 
                 heads, 
                 mlp_ratio=4.0,
                 drop_rate=.0, 
                 attn_drop_rate=.0):
        
        super().__init__()
        
        self.trnblock = nn.ModuleList()
        for d in range(depth):
            trn = transformer_block(
                    dim=in_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate)
            
            self.trnblock.append(trn)
            
    def forward(self,xquery,xcontext):

        for blk in self.trnblock:
            xquery = blk(xquery,xcontext)
        return xquery

class Patch2DembedPooling(nn.Module):
    def __init__(self,in_dim,out_dim,patch_size):
        super().__init__()
        
        self.conv = nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1,bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        
        self.pooling = nn.AvgPool2d(kernel_size=patch_size,stride=patch_size)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pooling(x)
        N,B,H,W = x.size()
        x = x.view(N,B,H*W)
        x = x.permute(0,2,1)
        
        return x

class Tensor2Tokens2d(nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self,x):
        N,B,H,W = x.size()
        x = x.view(N,B,H*W)
        x = x.permute(0,2,1)
        
        return x
    
class Tensor2Tokens3d(nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self,x):
        N,B,H,W,C = x.size()
        x = x.view(N,B,H*W*C)
        x = x.permute(0,2,1)
        
        return x

class MakeLayers(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,dim='2d'):
        super().__init__()
        
        if dim == '2d':
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=False)
            #self.bn = nn.InstanceNorm2d(out_channels)
            self.bn = nn.BatchNorm2d(out_channels)
        elif dim == '3d':
            self.conv = nn.Conv3d(in_channels,out_channels,kernel_size,padding=padding,bias=False)
            self.bn = nn.BatchNorm3d(out_channels)
            #self.bn = nn.InstanceNorm3d(out_channels)
        else:
            raise ValueError('dim %s does not in [2d,3d]'%dim)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Backbone(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 dim='2d',
                 firstPool=True):
        super().__init__()
        
        self.dim=dim
        self.layer = [32,64,128,256,256,64]

        
        self.conv1 = self._make_layers(in_channels,self.layer[0],1,isPool=firstPool,kernel_size=3,padding=1)
        self.conv2 = self._make_layers(self.layer[0],self.layer[1],1,kernel_size=3,padding=1)
        self.conv3 = self._make_layers(self.layer[1],self.layer[2],1,kernel_size=3,padding=1)
        self.conv4 = self._make_layers(self.layer[2],self.layer[3],1,kernel_size=3,padding=1)
        self.conv5 = self._make_layers(self.layer[3],self.layer[4],1,kernel_size=3,padding=1)
        self.conv6 = self._make_layers(self.layer[4],self.layer[5],1,isPool=False,kernel_size=1,padding=0)
        
        if self.dim == '2d':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        elif self.dim == '3d':
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            
        self.classifier = nn.Sequential(
            nn.Linear(self.layer[-1], num_classes),
        )
        
    
    def _make_layers(self,in_channels,out_channels,nlayers,isPool=True,kernel_size=3,padding=1):
        layers = []
        
        if isPool:
            if self.dim == '2d':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif self.dim == '3d':
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        
        for n in range(nlayers):
            lay = MakeLayers(in_channels,out_channels,kernel_size,padding,dim=self.dim)
            layers.append(lay)
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        '''
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(x4.shape)
        print(x5.shape)
        print(x6.shape)
        '''
        
        xfeat = torch.flatten(self.avgpool(x6),1)
        
        logits = self.classifier(xfeat)
        
        return logits,x6
    
def creat_model(in_channels=1,
                 num_classes=1,
                 dim='2d',
                 firstPool=True):
    
    model = Backbone(in_channels=in_channels,
                num_classes=num_classes,
                dim=dim,
                firstPool=firstPool)
    
    return model

class CompaireLearning(nn.Module):
    def __init__(self,
                 in_dim,
                 num_classes=1,
                 total_blocks=2,
                 drop_rate=0,
                 dim='3d',
                 firstPool=True,
                 shareBackone=True):
        
        super().__init__()
        
        if dim not in ['2d','3d']:
            raise ValueError('dim %s not in [2d,3d]'%dim)
        
        backbone = creat_model
        
        self.shareBackone = shareBackone
        
        self.featextraction1 = backbone(in_channels=in_dim,
                     num_classes=num_classes,
                     dim=dim,
                     firstPool=firstPool)
        if self.shareBackone:
            self.featextraction2 = self.featextraction1
        else:
            self.featextraction2 = backbone(in_channels=in_dim,
                     num_classes=num_classes,
                     dim=dim,
                     firstPool=firstPool)
        
        if dim == '2d':
            self.tokens = Tensor2Tokens2d()
        elif dim == '3d':
            self.tokens = Tensor2Tokens3d()
            
        self.channel_number = self.featextraction1.layer
        hidden_size = self.channel_number[-1]
        

        self.transformer = Transformer(
                 in_dim=hidden_size, 
                 depth=total_blocks, 
                 heads=8, 
                 mlp_ratio=4.0,
                 drop_rate=drop_rate, 
                 attn_drop_rate=drop_rate)
  
        self.add = nn.Linear(hidden_size,num_classes)
        self.sub = nn.Linear(hidden_size,num_classes)
        self.max = nn.Linear(hidden_size,num_classes)
        self.min = nn.Linear(hidden_size,num_classes)
        

    def forward(self,query,context):
        
        qlogits,qfeat = self.featextraction1(query)
        clogits,cfeat = self.featextraction2(context)
        
        queryfeat = self.tokens(qfeat)
        contfeat = self.tokens(cfeat)
        
        queryfeat = torch.cat([queryfeat,contfeat],1)
        
        #print(queryfeat.shape)
        
        queryfeat = self.transformer(queryfeat,queryfeat)
        
        logitslist = []
        logitslist.append(self.add(queryfeat[:,0]))
        logitslist.append(self.sub(queryfeat[:,1]))
        logitslist.append(self.max(queryfeat[:,2]))
        logitslist.append(self.min(queryfeat[:,3]))
        
        return logitslist
