import torch.nn as nn
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      xavier_init)
import torch
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
import numpy as np
from box_iou_rotated import obb_overlaps


class IOUfitModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(IOUfitModule, self).__init__()
        self.attention1 = IOUfitTransformer(in_channels=1, inter_channels=inter_channels)
        #self.bn11 = nn.BatchNorm1d(hidden_features)
        self.feedforward1 = FFN(inter_channels, inter_channels, act_cfg=dict(type='ReLU', inplace=True))
        #self.bn12 = nn.BatchNorm1d(hidden_features)
        self.out_fc1 = Linear(inter_channels, 1)
        self.out_fc2 = Linear(in_channels, 1)
        #self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.loss_weight = 10.0

    def forward(self, pred, gt):
        input = torch.cat([pred, gt], dim=-1).unsqueeze(-1) #N,16,1
        out = self.attention1(query=input, key=input, value=input) #N,16,128
        #out = self.bn11(out.transpose(1, 2)).transpose(1, 2)
        # N = out.size(0)
        # out = out.view(N, -1) #N,1024
        out = self.feedforward1(out) #N,16,128
        #out = self.bn12(out.transpose(1, 2)).transpose(1, 2)
        out = self.out_fc1(out) #N,16,1
        out = self.out_fc2(out.squeeze(-1)) # N,1
        return out

    def loss(self, rbboxes1, rbboxes2, iou_fit_value):
        IoU_targets = obb_overlaps(rbboxes1, rbboxes2.detach(), is_aligned=True).squeeze(
            1).clamp(min=1e-6, max=1)

        loss_fit = self.mse_loss(iou_fit_value, IoU_targets.detach()).sqrt()
        loss_fit = self.loss_weight * loss_fit
        # loss_fit = ((iou_fit_value - IoU_targets.detach()).square() + 1).log().sqrt()

        return loss_fit


class IOUfitTransformer(nn.Module):
    """A warpper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        dropout (float): A Dropout layer on attn_output_weights. Default 0.0.
    """

    def __init__(self, in_channels, inter_channels):    #, bn_layer=False
        super(IOUfitTransformer, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.q_project = Linear(in_channels, inter_channels)
        self.k_project = Linear(in_channels, inter_channels)
        self.v_project = Linear(in_channels, inter_channels)
        # if bn_layer:
        #     self.out_bn = nn.BatchNorm1d(self.inter_channels)
        
        self.out_project = Linear(inter_channels, inter_channels)
        # nn.init.constant_(self.out_project.weight, 0)
        # nn.init.constant_(self.out_project.bias, 0)

    def forward(self,
                query,
                key,
                value=None,
                #residual=None
                ):
        if value is None:
            value = key
        # if residual is None:
        #     residual = key
        
        query_emb = self.q_project(query)#N,16,64
        key_emb = self.k_project(key)#N,16,64
        value_emb = self.v_project(value)#N,16,64
        # if key_emb.dim()<3:
        #     print(key_emb.dim())
        similarity = torch.matmul(query_emb, key_emb.transpose(1,2)) / query_emb.size(-1) #N,16,16
        spft_similarity = F.softmax(similarity, dim=-1)  #N,16,16
        out = torch.bmm(spft_similarity, value_emb) / spft_similarity.size(-1) #N,16,64
        attn_out = self.out_project(out) #N,16,64
        #attn_out = self.out_bn(out_proj.transpose(1, 2)).transpose(1, 2)

        return attn_out
        #return residual + attn_out


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        add_residual (bool, optional): Add resudual connection.
            Defaults to True.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 add_residual=False):#dropout=0.0,
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        #self.dropout = dropout
        self.activate = build_activation_layer(act_cfg)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(Linear(in_channels, feedforward_channels), self.activate))#, nn.Dropout(dropout)
            in_channels = feedforward_channels
        layers.append(nn.Sequential(Linear(feedforward_channels, embed_dims), self.activate))
        self.layers = nn.Sequential(*layers)
        # self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + out #self.dropout(out)

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        #repr_str += f'dropout={self.dropout}, '
        repr_str += f'add_residual={self.add_residual})'
        return repr_str


class fitAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    # bias_k: Optional[torch.Tensor]
    # bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(fitAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            #self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(2 * embed_dim, embed_dim))
            self.register_parameter('qk_proj_weight', None)
            #self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(2 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.qk_proj_weight)
            #xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        # if not self._qkv_same_embed_dim:
        return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.qk_proj_weight, k_proj_weight=self.qk_proj_weight,
                v_proj_weight=self.v_proj_weight)
        # else:
        #     return F.multi_head_attention_forward(
        #         query, key, value, self.embed_dim, self.num_heads,
        #         self.in_proj_weight, self.in_proj_bias,
        #         self.bias_k, self.bias_v, self.add_zero_attn,
        #         self.dropout, self.out_proj.weight, self.out_proj.bias,
        #         training=self.training,
        #         key_padding_mask=key_padding_mask, need_weights=need_weights,
        #         attn_mask=attn_mask)



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

def get_model(in_channels, inter_channels=None):

    model = IOUfitModule(in_channels, inter_channels)  # 101
    return model