# -*- coding: utf-8 -*-

from dolg.dolg_model_tf2 import DOLG
from dolg.resnet_tf2 import ResNet
from tensorflow.keras import Input, Model
import cv2
import numpy as np 
import torch
from dolg.utils.extraction import _MEAN, _SD # BGR 



if __name__ == "__main__":
    _MEAN = np.array(_MEAN).reshape(1,1,-1)
    _SD = np.array(_SD).reshape(1,1,-1)
    depth = 101
    
    root_dir = "../images"
    
    
    backbone = ResNet(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform", name="globalmodel")
    model = DOLG(backbone, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=81313, pretrained=f"r{depth}")
    
    #model.load_weights(f"../weights/r{depth}_dolg_512_from_paddle.h5", skip_mismatch=True, by_name=True)
    img = cv2.resize(cv2.imread("../images/landmark.png"), (512,512))/255.0
    img = (img - np.array(_MEAN).reshape(1,1,-1))/np.array(_SD).reshape(1,1,-1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    output, output2 = model.predict(img)