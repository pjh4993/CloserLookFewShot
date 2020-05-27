# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

CTMblock = backbone.CTMblock

class CTM(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(CTM, self).__init__( model_func,  n_way, n_support)
        self.feature = model_func(flatten=False)
        self.feat_dim = self.feature.final_feat_dim
        self.ctm_dim = self.feat_dim.copy()
        self.ctm_dim[0] = self.ctm_dim[0]//8
        self.concentrator = CTMblock(self.feat_dim[0], self.ctm_dim[0])
        outdim = self.concentrator.outdim[0]
        outdim *= self.n_way
        self.projector = CTMblock(outdim, self.ctm_dim[0], flatten=True)
        self.reshaper = CTMblock(self.feat_dim[0], self.ctm_dim[0],flatten=True)
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        # 1. concentrator
        z_support = z_support.reshape([self.n_way * self.n_support] + self.feat_dim)
        z_query = z_query.reshape([self.n_way * self.n_query] + self.feat_dim)
        z_common = self.concentrator(z_support)
        z_common = z_common.view([self.n_way, self.n_support] + self.ctm_dim ).mean(1)
        # 2. Projector
        z_common = z_common.view(1,self.n_way * self.concentrator.outdim[0], self.ctm_dim[1], self.ctm_dim[2])
        z_mask = self.projector(z_common)
        z_mask = F.softmax(z_mask, dim=1)
        
        # 3. reshaper
        z_support = self.reshaper(z_support)
        z_query = self.reshaper(z_query)

        z_support = z_support * z_mask.expand(self.n_way * self.n_support, z_support.shape[1])
        z_query = z_query * z_mask.expand(self.n_way * self.n_query, z_query.shape[1])
        
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
