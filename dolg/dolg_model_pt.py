import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .resnet_pt import GeneralizedMeanPoolingP, init_weights
import os 
from .utils.download_weights import get_checkpoint_path, download_checkpoint

""" Dolg models """

class DOLG(nn.Module):
    """ DOLG model """
    def __init__(self, backbone, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=81313, pretrained=None):
        super(DOLG, self).__init__()
        self.pool_l= nn.AdaptiveAvgPool2d((1, 1)) 
        self.pool_g = GeneralizedMeanPoolingP(norm=3.0) 
        self.fc_t = nn.Linear(s4_dim, s3_dim, bias=True)
        self.fc = nn.Linear(s3_dim*2, head_reduction_dim, bias=True)
        self.globalmodel = backbone
        self.localmodel = SpatialAttention2d(s3_dim, act_fn='relu', with_aspp=with_ma)
        self.s4_dim, self.s3_dim, self.s2_dim = s4_dim, s3_dim, s2_dim
        self.desc_cls = Arcface(head_reduction_dim, num_classes) if num_classes is not None and type(num_classes) == int else None
        
        if pretrained is not None:
            self.load_pretrained_weights(pretrained)
        
    def load_pretrained_weights(self, pretrained):
        path_weight = get_checkpoint_path()
        if pretrained=="r50":
            filename = "r50_dolg_512.pt"
            if not os.path.exists(path_weight) or filename not in os.listdir(path_weight):
                download_checkpoint(model_name=pretrained, backend="pt")
            ckpt = torch.load(os.path.join(path_weight, filename) , map_location="cpu")
        elif pretrained == "r101":
            filename = "r101_dolg_512.pt"
            if not os.path.exists(path_weight) or filename not in os.listdir(path_weight):
                download_checkpoint(model_name=pretrained, backend="pt")
            ckpt = torch.load(os.path.join(path_weight, filename) , map_location="cpu")
        else:
            raise(f"{pretrained} does not exist as a pretrained weight")
        
        model_keys = set(list(self.state_dict().keys()))
        ckpt_keys = set(list(ckpt.keys()))
        
        missing_key = model_keys - ckpt_keys 
        print("Number of keys missing in pretrained weights : ", len(missing_key))
        print(missing_key)
        
        key_not_exist = ckpt_keys - model_keys
        print("Number of key which can't be find in the model : ", len(key_not_exist))
        print(key_not_exist)
        self.load_state_dict(ckpt, strict=False)
        
    def forward(self, x, targets=None):
        """ Global and local orthogonal fusion """
        f3, f4 = self.globalmodel(x)
        fl, _ = self.localmodel(f3)
        
        fg_o = self.pool_g(f4)
        fg_o = fg_o.view(fg_o.size(0), self.s4_dim)
        
        fg = self.fc_t(fg_o)
        fg_norm = torch.norm(fg, p=2, dim=1)
        
        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(fl, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(fl.size())
        proj = proj / (fg_norm * fg_norm).view(-1, 1, 1, 1)
        orth_comp = fl - proj

        fo = self.pool_l(orth_comp)
        fo = fo.view(fo.size(0), self.s3_dim)

        final_feat=torch.cat((fg, fo), 1)
        global_feature = self.fc(final_feat)
        
        if self.desc_cls is not None:
            global_logits = self.desc_cls(global_feature, targets)
            return global_feature, global_logits
        return global_feature



class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    @inc_c = s3 dim
    '''
    def __init__(self, in_c, act_fn='relu', with_aspp=False, bn_eps=1e-5, bn_mom=0.1):
        super(SpatialAttention2d, self).__init__()
        
        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(in_c)
        self.conv1 = nn.Conv2d(in_c, in_c, 1, 1)
        self.bn = nn.BatchNorm2d(in_c, eps=bn_eps, momentum=bn_mom)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_c, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

        for conv in [self.conv1, self.conv2]: 
            conv.apply(init_weights)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        if self.with_aspp:
            x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        return self.__class__.__name__


class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling Module 
    '''
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2d(in_c, 512, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2d(in_c, 512, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.ModuleList(self.aspp)

        self.im_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_c, 512, 1, 1),
                                     nn.ReLU())
        conv_after_dim = 512 * (len(self.aspp)+1)
        self.conv_after = nn.Sequential(nn.Conv2d(conv_after_dim, 1024, 1, 1), nn.ReLU())
        
        for dilation_conv in self.aspp:
            dilation_conv.apply(init_weights)
        for model in self.im_pool:
            if isinstance(model, nn.Conv2d):
                model.apply(init_weights)
        for model in self.conv_after:
            if isinstance(model, nn.Conv2d):
                model.apply(init_weights)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h,w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = torch.cat(aspp_out, 1)
        x = self.conv_after(aspp_out)
        return x
    
    
class Arcface(nn.Module):
    """ Additive Angular Margin Loss """
    def __init__(self, in_feat, num_classes, scale=128, margin=0.15):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets=None):
        # get cos(theta)
        
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight)) # (bs, num_class)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        if targets is None:
            return cos_theta * self._s
        
        target_logit = cos_theta[torch.arange(0, features.size(0)), targets.view(-1)].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )
