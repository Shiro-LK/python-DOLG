# -*- coding: utf-8 -*-
import dolg
from dolg.dolg_model_pt import DOLG
from dolg.resnet_pt import ResNet
import cv2
from torchvision import transforms
import numpy as np 
import torch 
from dolg.utils.extraction import _MEAN, _SD

if __name__ == "__main__":

    root_dir = "../images"
    
    depth = 101
    backbone = ResNet(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform")
    model = DOLG(backbone, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=81313, pretrained=f"r{depth}")

    model.eval()
    #model.load_state_dict(torch.load(f"../weights/r{depth}_dolg_512_from_paddle.pt"), strict=False)
    img = np.transpose(cv2.resize(cv2.imread("../images/landmark.png"), (512,512)), (2,0,1))/255.0
    img = (img - np.array(_MEAN).reshape(-1,1,1))/np.array(_SD).reshape(-1,1,1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = torch.as_tensor(img).float()


    with torch.no_grad():
        output, output2  = model(img )
        