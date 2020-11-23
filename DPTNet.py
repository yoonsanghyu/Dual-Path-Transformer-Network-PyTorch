# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:06:44 2020

@author: yoonsanghyu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.modules.container import ModuleList
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride = L//2, bias=False)
        
    def forward(self, x):

        enc_out = F.relu(self.conv1d_U(x)) # [M, 1, T] -> [M, N, I]  
        return enc_out
    
class Decoder(nn.Module):
    def __init__(self, L, N):
        super(Decoder, self).__init__()
        
        self.L, self.N = L, N
        self.dconv1d_V = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2, bias=False)
        
    def forward(self, x):
        out = self.dconv1d_V(x)
        return out
  

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = LSTM(d_model, d_model*2, 1, bidirectional=True)
        self.linear2 = Linear(d_model*2*2, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        ## type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        self.linear1.flatten_parameters()
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)[0])))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class DPTBlock(nn.Module):
    def __init__(self, input_size, nHead, dim_feedforward):
        super(DPTBlock, self).__init__()

        #d_model, nhead, dim_feedforward, dropout=0, activation="relu"
        self.intra_transformer = TransformerEncoderLayer(d_model=input_size, nhead=nHead, dim_feedforward=dim_feedforward, dropout=0)
        self.inter_transformer = TransformerEncoderLayer(d_model=input_size, nhead=nHead, dim_feedforward=dim_feedforward, dropout=0)
        
    def forward(self,x):
        
        B, N, K, P = x.shape
        
        # intra DPT
        row_input =  x.permute(0, 3, 2, 1).contiguous().view(B*P, K, N) # [B, N, K, S] -> [B, P, K, N] -> [B*P, K, N]
        row_output = self.intra_transformer(row_input.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        row_output = row_output.view(B, P, K, N).permute(0, 3, 2, 1).contiguous()  # [B*P, K, N] -> [B, P, K, N]
        
        output = x + row_output

        #inter DPT
        col_input = output.permute(0, 2, 3, 1).contiguous().view(B*K, P, N) # [B, P, K, N] -> [B, K, P, N] -> [B*K, P, N]
        col_output = self.inter_transformer(col_input.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        col_output = col_output.view(B, K, P, N).permute(0, 3, 1, 2).contiguous() # [B*K, P, N] -> [B, K, P, N]
        
        output = output + col_output
        
        return output 
    

        
class Separator(nn.Module):
    def __init__(self, N, C, L, H, K, B):
        super(Separator, self).__init__()
        
        self.N = N
        self.C = C
        self.K = K
        self.B = B
        self.LN = nn.GroupNorm(1, N, eps=1e-8)
        
        self.DPT = nn.ModuleList([])
        for i in range(B):
            self.DPT.append(DPTBlock(N, H, 256))
        
        self.prelu = nn.PReLU()
        self.conv2d = nn.Conv2d(N, N*C, kernel_size=1)
        
        #self.output = nn.Sequential(nn.Conv1d(N, N, 1), nn.Tanh())
        #self.output_gate = nn.Sequential(nn.Conv1d(N, N, 1), nn.Sigmoid())

    def forward(self, x):
        out, gap = self.split_feature(x, self.K)  # [B, N, I] -> [B, N, K, S]
        out = self.LN(out) # [B, N, K, S] -> [B, N, K, S]

        for i in range(self.B):
            out = self.DPT[i](out) # [B, N, K, S] -> [B, N, K, S]

        out = self.conv2d(self.prelu(out)) # [B, N, K, S] -> [B, N*C, K, S]

        B, _, K, S = out.shape
        out = out.view(B, -1, self.C, K, S).permute(0, 2, 1, 3, 4).contiguous() # [B, N*C, K, S] -> [B, N, C, K, S] 
        out = out.view(B*self.C, -1, K, S)
        out = self.merge_feature(out, gap)  # [B*C, N, K, S]  -> [B*C, N, I]
        
        #out = F.relu(self.output(out)*self.output_gate(out))
        out = F.relu(out)
        return out
    
    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T
        
    
class DPTNet(nn.Module):
    
    """
        Args:
            C: Number of speakers
            N: Number of filters in autoencoder
            L: Length of the filters in autoencoder
            H: Multi-head
            K: segment size
            R: Number of repeats

    """

    def __init__(self, N=64, C=2, L=2, H=4, K=250, B=6):
        super(DPTNet, self).__init__()
        
        self.C = C # number of sources 
        
        self.N = N # encoder dimension
        self.L = L # encoder win length
        
        self.H = H # Multi-head
        
        self.K = K # segment size (=chunk size)
        
        self.B = B # repeats
        
        self.encoder = Encoder(L, N)
        self.separator = Separator(N, C, L, H, K, B)
        self.decoder = Decoder(L, N)
        
        
    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.L - (self.L//2 + nsample % self.L) % self.L
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.L//2)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest      
    
    def forward(self, x):
        
        # encoding
        x, rest = self.pad_signal(x)
        enc_out = self.encoder(x) # [B, 1, T] -> [B, N, I]
        
        # mask estimation
        masks = self.separator(enc_out) # [B, N, I] -> [B*C, N, I]
        _, N, I = masks.shape
        masks = masks.view(self.C, -1, N, I) #[C, B, N, I]
        
        # masking
        out = [masks[i]*enc_out for i in range(self.C)] # C*([B, N, I]) * [B, N, I]
        
        # decoding
        audio = [self.decoder(out[i]) for i in range(self.C)] # C*[B, 1, T]
        audio[0] = audio[0][:,:,self.L//2:-(rest+self.L//2)].contiguous()  # B, 1, T
        audio[1] = audio[1][:,:,self.L//2:-(rest+self.L//2)].contiguous()  # B, 1, T
        audio = torch.cat(audio,dim=1) #[B, C, T]

        return audio
    

if __name__ == "__main__":
    
   
    input = torch.rand(3,32000)
    model = DPTNet()
    #print(model)
    #out = model(input) 
    #print(out.shape)
    
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)

    

