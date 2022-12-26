# -*- coding: utf-8 -*-
"""
    Compare pytorch cls from arcface vs TF cls from arcface re-implementation

"""
import torch
import tensorflow as tf 
import cv2
import os, sys
from torchvision import transforms
from dolg.dolg_model_pt import DOLG as DOLG_pt
from dolg.resnet_pt import ResNet as ResNet_pt
from dolg.dolg_model_tf2 import DOLG as DOLG_tf 
from dolg.resnet_tf2 import ResNet as ResNet_tf
from dolg.utils.extraction import process_data, compute_similarity
import numpy as np 

if __name__ == "__main__":
    
    root_dir = "../images"
    images_names = ["notre_dame_2.jpg", "notre_dame.jpg", "landmark.png"]
    depth = 101
    
    # pytorch
    backbone_pt = ResNet_pt(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform")
    model_pt = DOLG_pt(backbone_pt, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=81313, pretrained=f"r{depth}")
    
    model_pt.eval()
    target = np.array([2, 5, 12], dtype=np.int32).reshape(-1,1)
    img = [process_data(images_names[i], root_dir, mode="pt") for i in range(len(images_names))]
    img = torch.stack(img)
    with torch.no_grad():
        _, output_pt_cls = model_pt(img)
        _, output_pt_cls2 = model_pt(img, targets=torch.as_tensor(target).long())
        
        
    # tensorflow 
    
    backbone_tf = ResNet_tf(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform", name="globalmodel")
    model_tf = DOLG_tf(backbone_tf, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=81313, pretrained=f"r{depth}")
    img = np.stack([process_data(images_names[i], root_dir, mode="tf") for i in range(len(images_names))])

    _, output_tf_cls = model_tf(img)
    _, output_tf_cls2 = model_tf(img, targets=tf.constant(target))
    
    # similarity between pt and tf should be one or close to one.
    similarity = np.concatenate([compute_similarity(output_pt_cls[i].numpy().reshape(1,-1), output_tf_cls[i].numpy().reshape(1,-1)) for i in range(len(output_pt_cls))])
    similarity2 = np.concatenate([compute_similarity(output_pt_cls2[i].numpy().reshape(1,-1), output_tf_cls2[i].numpy().reshape(1,-1)) for i in range(len(output_pt_cls))])
    assert np.round(similarity.sum(), 2) == len(images_names)
    assert np.round(similarity2.sum(), 2)  == len(images_names)
    print("output cls similar between torch and tf")
    