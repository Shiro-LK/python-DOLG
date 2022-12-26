# -*- coding: utf-8 -*-
"""
    Compare pytorch embeddings vs TF embeddings re-implementation

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
from dolg.utils.extraction import extract_embedding_tf, extract_embedding_pt, compute_similarity
import numpy as np 


if __name__ == "__main__":
    
    root_dir = "../images"
    images_names = ["notre_dame_2.jpg", "notre_dame.jpg", "landmark.png"]
    depth = 50
    
    # pytorch
    backbone_pt = ResNet_pt(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform")
    model_pt = DOLG_pt(backbone_pt, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=None, pretrained=f"r{depth}")
    model_pt.eval()

    embeddings_pt = [extract_embedding_pt(filename, root_dir, model_pt) for filename in images_names]
    
    sim1_pt = compute_similarity(embeddings_pt[0], embeddings_pt[1])
    sim2_pt = compute_similarity(embeddings_pt[2], embeddings_pt[1])
    
    # tensorflow 
    
    backbone_tf = ResNet_tf(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform", name="globalmodel")
    model_tf = DOLG_tf(backbone_tf, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=None, pretrained=f"r{depth}")

    embeddings_tf = [extract_embedding_tf(filename, root_dir, model_tf) for filename in images_names]
    
    sim1_tf = compute_similarity(embeddings_tf[0], embeddings_tf[1])
    sim2_tf = compute_similarity(embeddings_tf[2], embeddings_tf[1])
    
    
    assert np.round(compute_similarity(sim1_tf.reshape(1,-1), sim1_pt.reshape(1,-1)), 2) == 1
    assert np.round(compute_similarity(sim2_tf.reshape(1,-1), sim2_pt.reshape(1,-1)), 2) == 1