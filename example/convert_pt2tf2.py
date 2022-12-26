# -*- coding: utf-8 -*-
import sys, os
from dolg.dolg_model_tf2 import DOLG
from dolg.resnet_tf2 import ResNet
from dolg.utils.convert_pt_to_tf2 import assign_globalmodel, assign_localmodel
from tensorflow.keras import Input, Model
import cv2
import numpy as np 
import torch
import tensorflow as tf 

if __name__ == "__main__":
    depth = 50
    data = torch.load(f"../weights/r{depth}_dolg_512.pt")
    backbone = ResNet(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform", name="globalmodel")
    model = DOLG(backbone, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=True, num_classes=81313, pretrained=None)

    for l in model.layers:
        print(l.name)
        #print(l.name, l.output_shape)
        if l.name =="localmodel":
            l, data = assign_localmodel(l, data)
        elif l.name == "fc":
            l.kernel.assign(data["fc.weight"].transpose(1,0))
            l.bias.assign(data["fc.bias"])
            del data["fc.weight"], data["fc.bias"]
        elif l.name == "fc_t":
            l.kernel.assign(data["fc_t.weight"].transpose(1,0))
            l.bias.assign(data["fc_t.bias"])
            del data["fc_t.weight"], data["fc_t.bias"]
        elif l.name =="globalmodel":
            l, data = assign_globalmodel(l, data)
        elif l.name == "pool_g":
            l.weights[0].assign(data["pool_g.p"])
            del data["pool_g.p"]
        elif l.name == "desc_cls":
            l.weight.assign(data["desc_cls.weight"].transpose(1,0))
            l.t.assign(data["desc_cls.t"])
            del data["desc_cls.weight"], data["desc_cls.t"]
        else:
            print(l.name)
    
    model.save_weights( f"../weights/r{depth}_dolg_512_from_pt.h5")