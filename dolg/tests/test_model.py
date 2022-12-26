# -*- coding: utf-8 -*-
import pytest
from dolg.dolg_model_pt import DOLG as DOLG_pt
from dolg.resnet_pt import ResNet as ResNet_pt 
from dolg.dolg_model_tf2 import DOLG as DOLG_tf 
from dolg.resnet_tf2 import ResNet as ResNet_tf 

@pytest.mark.unittest
def test_resnet_tf():
    depth = 50
    model = ResNet_tf(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform", name="globalmodel")
    assert model is not None 
  
@pytest.mark.unittest
def test_resnet_pt():
    depth = 50
    model = ResNet_pt(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform")
    assert model is not None 
    
@pytest.mark.unittest    
def test_dolg_tf():
    depth = 50
    backbone_tf = ResNet_tf(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform", name="globalmodel")
    model = DOLG_tf(backbone_tf, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=None, pretrained=None)
    
    assert model is not None

@pytest.mark.unittest 
def test_dolg_pt():
    depth = 50
    backbone_pt = ResNet_pt(depth=depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform" )
    model = DOLG_pt(backbone_pt, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=None, pretrained=None)
    
    assert model is not None
    
    
if __name__ == "__main__":
    test_resnet_tf()
    test_resnet_pt()
    test_dolg_tf()
    test_dolg_pt()