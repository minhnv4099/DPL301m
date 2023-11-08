import math
from collections import OrderedDict
import torch
from torch import nn

class Position(nn.Module):
    def __init__(self,in_dim:int):
        super(Position, self).__init__()
        self.in_dim = in_dim
        self.norm = nn.LayerNorm(normalized_shape=self.in_dim)

    def forward(self,tensor:torch.Tensor):
        batch_size = tensor.shape[0]
        sequence_length = tensor.shape[-2]
        dim = tensor.shape[-1]

        position = torch.arange(end=sequence_length,dtype=torch.long)
        d = torch.arange(end=dim, dtype=torch.long)
        d = (2 * d / dim)

        position = torch.unsqueeze(input=position,dim=-1)
        position = position / (1e4 ** d)

        position[:,::2] = torch.sin(input=position[:,::2])
        position[:, 1::2] = torch.cos(input=position[:, 1::2])

        batch_position = torch.expand_copy(input=position,size=(batch_size,)+position.shape)
        return self.norm(batch_position+tensor)

class ResidualLayer(nn.Module):
    def __init__(self,module:nn.Module,weight:float=1.,*args,**kwargs):
        super(ResidualLayer, self).__init__(*args,**kwargs)
        self.module = module
        self.weight = weight

    def forward(self,tensor:torch.Tensor):
        output:torch.Tensor = self.module(tensor)
        shape_out = output.shape
        shape_tensor = tensor.shape
        if shape_tensor[-1] == shape_out[-1]:
            return output + self.weight*output
        else:
            self.convert_block = nn.Sequential(OrderedDict([
                ('conv2_1',nn.Conv2d(in_channels=shape_out[-1],out_channels=shape_tensor[-1])),
                ('batch_norm_1',nn.BatchNorm2d(num_features=shape_tensor[-1]))
            ]))
            return self.convert_block(output) + self.weight*tensor


class SelfAttention(nn.Module):
    def __init__(self,in_dim,out_dim,*args,**kwargs):
        super(SelfAttention, self).__init__(*args,**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        if 'value_dim' in kwargs:
            self.value_dim = kwargs.pop('value_dim')
        else:
            self.value_dim = self.out_dim

        if 'key_dim' in kwargs and 'query_dim' in kwargs:
            assert kwargs['key_dim'] == kwargs['query_dim'], '"key_dim" and "query_dim" must match'
            self.key_dim = self.query_dim = kwargs.pop('key_dim')
            kwargs.pop('query_dim')
        elif 'key_dim' in kwargs:
            self.key_dim = self.query_dim = kwargs.pop('key_dim')
        elif 'query_dim' in kwargs:
            self.key_dim = self.query_dim = kwargs.pop('query_dim')
        else:
            self.key_dim = self.query_dim = self.out_dim

        self.key_transform = nn.Linear(in_features=in_dim,out_features=self.key_dim)
        self.query_transform = nn.Linear(in_features=in_dim,out_features=self.query_dim)
        self.value_transform = nn.Linear(in_features=in_dim, out_features=self.value_dim)
        self.position_transform = Position(in_dim=self.in_dim)

    def forward(self,tensor:torch.Tensor,attention_mask:torch.Tensor=None):
        positional_tensor = self.position_transform(tensor)
        keys:torch.Tensor = self.key_transform(positional_tensor)
        queries:torch.Tensor = self.key_transform(positional_tensor)
        values:torch.Tensor = self.key_transform(positional_tensor)

        return scale_product_attention(queries,keys,values,attention_mask)

class Attention(nn.Module):
    def __init__(self,in_dim:int,out_dim:int,*args,**kwargs):
        super(Attention, self).__init__(*args,**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self,queries:torch.Tensor,keys:torch.Tensor=None,values:torch.Tensor=None,attention_mask:torch.Tensor=None):
        if keys is None: keys = queries
        if values is None: values = keys
        return scale_product_attention(queries,keys,values,attention_mask)

def scale_product_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor):
    assert q.shape[-1] == k.shape[-1], '"q_dim" and "k_dim" must match'
    product = torch.matmul(input=q, other=torch.transpose(k, dim0=-1, dim1=-2))
    d_dim = k.shape[-1]
    scaled = product / math.sqrt(d_dim)
    if mask is not None:
        scaled = torch.masked_fill(scaled, mask=mask, value=1e-9)

    align_weights = torch.softmax(input=scaled, dim=-1)
    context = torch.matmul(input=align_weights, other=v)

    return context

class MutilHead(nn.Module):
    def __init__(self,n_head:int,d_head:int,in_dim:int,out_dim:int,*args,**kwargs):
        super(MutilHead, self).__init__(*args,**kwargs)
        self.in_dim = in_dim
        self.n_head = n_head
        self.d_head = d_head
        self.out_dim = out_dim
        self.attention = SelfAttention(in_dim=self.in_dim,out_dim=self.d_head)
        self.dense = nn.Linear(in_features=self.n_head*self.d_head,out_features=self.out_dim)

    def _split_on_head(self,tensor:torch.Tensor) -> torch.Tensor:
        """
        :param tensor: Tensor size of [N,S,in_dim*n_head]
        :return: Split on head tensor size of [N*n_head,S,in_dim]
        """
        batch,sequence_length = tensor.shape[:2]
        x = torch.reshape(input=tensor,shape=(batch,sequence_length,self.n_head,-1))
        x = torch.permute(input=x,dims=(0,2,1,3))
        x = torch.reshape(input=x,shape=(-1,sequence_length,self.in_dim))
        return x

    def _split_on_output(self,tensor:torch.Tensor) -> torch.Tensor:
        """
        :param tensor: Tensor size of [N*n_head,S,d_head]
        :return: Concated tensor size of [N,S,n_head*d_head]
        """
        sequence_length = tensor.shape[1]
        x = torch.reshape(input=tensor,shape=(-1,self.n_head,sequence_length,self.d_head))
        x = torch.permute(input=x, dims=(0, 2, 1, 3))
        x = torch.reshape(input=x,shape=(-1,sequence_length,self.n_head*self.d_head))
        return x

    def forward(self,tensor:torch.Tensor,attention_mask:torch.Tensor=None) -> torch.Tensor:
        """
        :param tensor: Tensor size of [batch,sequence_length,in_dim]
        :param attention_mask: Tensor size of [batch,sequence_length] or [batch,sequence_length,in_dim]
        :return: Final output of multihead, tensor size of [batch,sequence_length,out_dim]
        """
        multi_tensor = tensor.repeat((1,1,self.n_head))
        multi_tensor_split = self._split_on_head(multi_tensor)
        multi_attention_mask = attention_mask.repeat((1,self.n_head)) if attention_mask is not None else attention_mask
        multi_alignments = self.attention(multi_tensor_split,multi_attention_mask)
        multi_alignments_split = self._split_on_output(multi_alignments)
        return self.dense(multi_alignments_split)

class ClipRelu(nn.Module):
    def __init__(self,min=0.0,max=100):
        super(ClipRelu, self).__init__()
        self.clip_min = min
        self.clip_max = max

    def forward(self,tensor:torch.Tensor):
        return torch.clamp(input=tensor,min=self.clip_min,max=self.clip_max)

class FeatureExactor(torch.nn.Module):
    IN_CHANNELS = 2

    def __init__(self,in_channels,out_channels,*args,**kwargs):
        torch.nn.Module.__init__(self,*args,**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block_1 = nn.Sequential(OrderedDict({
            'b1_conv2_1': torch.nn.Conv2d(in_channels=self.in_channels, out_channels=32, stride=(1, 1), padding='same',kernel_size=(7, 3)),
            'b1_cliprelu_1': ClipRelu(max=60),
            'b1_bn2_1' : torch.nn.BatchNorm2d(num_features=32),
            'b1_conv2_2': torch.nn.Conv2d(in_channels=32, out_channels=32, stride=(1, 1), padding='same',kernel_size=(9, 3)),
            'b1_cliprelu_2': ClipRelu(max=60),
            'b1_bn2_2': torch.nn.BatchNorm2d(num_features=32),
        }))

        self.block_2 = nn.Sequential(OrderedDict({
            'b2_conv2_1': torch.nn.Conv2d(in_channels=32, out_channels=64, stride=(1, 1), padding='same',kernel_size=(11, 3)),
            'b2_bn2_1': torch.nn.BatchNorm2d(num_features=64),
            'b2_cliprelu_1': ClipRelu(max=60),
            'b2_conv2_2': torch.nn.Conv2d(in_channels=64, out_channels=64, stride=(1, 1), padding='same',kernel_size=(11, 5)),
            'b2_cliprelu_2': ClipRelu(max=60),
            'b2_bn2_2': torch.nn.BatchNorm2d(num_features=64),
        }))

        self.block_3 = nn.Sequential(OrderedDict({
            'b3_conv2_1': torch.nn.Conv2d(in_channels=64, out_channels=self.out_channels, stride=(1, 1), padding='same',kernel_size=(11, 5)),
            'b3_bn2_1': torch.nn.BatchNorm2d(num_features=self.out_channels),
            'b3_cliprelu_1': ClipRelu(max=30),
            'b3_conv2_2': torch.nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, stride=(1, 1), padding='same',kernel_size=(11, 5)),
            'b3_cliprelu_2': ClipRelu(max=30),
            'b3_bn2_2': torch.nn.BatchNorm2d(num_features=self.out_channels),
        }))

        self.pool2_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop_out_1 = nn.Dropout2d(p=0.3)

        self.residual_block_1 = ResidualLayer(module=self.block_1)
        self.residual_block_2 = ResidualLayer(module=self.block_2)
        self.residual_block_3 = ResidualLayer(module=self.block_3)

    def forward(self,tensor:torch.Tensor):
        x = self.residual_block_1.forward(tensor=tensor)
        x = self.pool2_1(x)
        x = self.residual_block_2.forward(tensor=x)
        x = self.pool2_1(x)
        x = self.residual_block_3.forward(tensor=x)
        x = self.pool2_1(x)
        x = self.drop_out_1(x)
        x = torch.flatten(x,start_dim=-2)
        return x,x.shape[-1]

class Encoder(nn.Module):
    def __init__(self,in_dim:int,out_dim:int,*args,**kwargs):
        super(Encoder, self).__init__(*args,**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_feature1 = 32
        self.d_feature2 = 64
        self.d_feature3 = 128

        self.sequen_dense = nn.Sequential(OrderedDict([
            ('dense_1',nn.Linear(in_features=in_dim,out_features=self.d_feature1)),
            ('relu_1',nn.LeakyReLU(inplace=False)),
            ('dense_2', nn.Linear(in_features=self.d_feature1, out_features=self.d_feature2)),
            ('leaky_relu_1', nn.LeakyReLU(inplace=False,negative_slope=-0.9)),
            ('dense_3', nn.Linear(in_features=self.d_feature2, out_features=self.d_feature3)),
            ('gelu_1', nn.GELU()),
        ]))

        self.sequen_rnn = nn.Sequential(OrderedDict([
            ('lstm_1',nn.LSTM(input_size=self.d_feature3,hidden_size=self.out_dim,bidirectional=True,num_layers=2))
        ]))

    def forward(self,tensor:torch.Tensor):
        x = self.sequen_dense(tensor)
        x = self.sequen_rnn(x)[0]
        return x

class Labeler(nn.Module):
    def __init__(self,in_dim:int,out_dim:int,*args,**kwargs):
        super(Labeler, self).__init__(*args,**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_feature1 = 32
        self.d_feature2 = 64
        self.predictor = nn.Sequential(OrderedDict([
            ('dense_1',nn.Linear(in_features=self.in_dim,out_features=self.d_feature1)),
            ('relu_1',nn.RReLU(inplace=False)),
            ('dense_2', nn.Linear(in_features=self.d_feature1, out_features=self.d_feature2)),
            ('relu_2', nn.RReLU()),
            ('dense_3', nn.Linear(in_features=self.d_feature2, out_features=self.out_dim)),
        ]))

    def forward(self,tensor:torch.Tensor):
        return self.predictor(tensor)


class FullModel(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,encode_dim:int,n_head:int,d_head:int,out_dim:int,mode='train'):
        super(FullModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exactor_dim = None

        self.encode_dim = encode_dim
        self.n_head = n_head
        self.d_head = d_head
        self.out_dim = out_dim

        self.exactor = FeatureExactor(in_channels=self.in_channels,out_channels=self.out_channels)
        self.attention = None
        self.residual_attention = None
        self.encoder = None
        self.predictor = None

    def forward(self,tensor:torch.Tensor):

        x,exactor_dim = self.exactor(tensor)
        if self.exactor_dim is None:
            self.exactor_dim = exactor_dim

        if self.attention is None:
            self.attention: MutilHead = MutilHead(n_head=self.n_head,d_head=self.d_head,in_dim=self.exactor_dim,out_dim=self.exactor_dim)
        if self.residual_attention is None:
            self.residual_attention = ResidualLayer(module=self.attention)
        if self.encoder is None:
            self.encoder = Encoder(in_dim=self.exactor_dim, out_dim=self.encode_dim)
        if self.predictor is None:
            self.predictor = Labeler(in_dim=self.encode_dim * 2, out_dim=self.out_dim)

        x = self.residual_attention(x)
        x = self.encoder(x)
        x = self.predictor(x)
        return x

if __name__ == '__main__':
    pass