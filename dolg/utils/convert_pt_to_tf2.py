# -*- coding: utf-8 -*-
"""
@author: Shiro-LK

Convert pt weights into tf for ResNet (DOLG)
"""
import tensorflow as tf


def assign_localmodel(layer, weights):
    
    layer.conv1.weights[0].assign(tf.transpose(weights["localmodel.conv1.weight"] , perm=[2,3,1,0]))
    layer.conv1.weights[1].assign(weights["localmodel.conv1.bias"])
    
    layer.conv2.weights[0].assign(tf.transpose(weights["localmodel.conv2.weight"] , perm=[2,3,1,0]))
    layer.conv2.weights[1].assign(weights["localmodel.conv2.bias"])
    
    # BN
    
    layer.bn.weights[0].assign(weights["localmodel.bn.weight"])
    layer.bn.weights[1].assign(weights["localmodel.bn.bias"])
    layer.bn.weights[2].assign(weights["localmodel.bn.running_mean"])
    layer.bn.weights[3].assign(weights["localmodel.bn.running_var"])
    
    del weights["localmodel.conv1.weight"], weights["localmodel.conv1.bias"]
    del weights["localmodel.conv2.weight"], weights["localmodel.conv2.bias"]
    del weights["localmodel.bn.weight"], weights["localmodel.bn.bias"]
    del weights["localmodel.bn.running_mean"], weights["localmodel.bn.running_var"]
    
    # if there is aspp
    if 'localmodel.aspp.aspp.0.weight' in weights.keys():
        for i in range(4):
            name_w = f'localmodel.aspp.aspp.{i}.weight' # even
            name_b = f'localmodel.aspp.aspp.{i}.bias' # not even number
            layer.aspp.aspp.weights[i*2].assign(tf.transpose(weights[name_w] , perm=[2,3,1,0]))
            layer.aspp.aspp.weights[i*2+1].assign(weights[name_b])
            del weights[name_w], weights[name_b]
        layer.aspp.im_pool.weights[0].assign(tf.transpose(weights["localmodel.aspp.im_pool.1.weight"] , perm=[2,3,1,0]))
        layer.aspp.im_pool.weights[1].assign(weights["localmodel.aspp.im_pool.1.bias"])
        
        layer.aspp.conv_after.weights[0].assign(tf.transpose(weights["localmodel.aspp.conv_after.0.weight"] , perm=[2,3,1,0]))
        layer.aspp.conv_after.weights[1].assign(weights["localmodel.aspp.conv_after.0.bias"])
        del weights["localmodel.aspp.im_pool.1.weight"], weights["localmodel.aspp.im_pool.1.bias"]
        del weights["localmodel.aspp.conv_after.0.weight"], weights["localmodel.aspp.conv_after.0.bias"]
    return layer, weights

def assign_globalmodel(layer, weights):

    to_delete = []
    for key, value in weights.items():
        if "globalmodel" not in key:
            continue
        key = key.split(".")
        new_key = ".".join(key[1:-1]).replace("block", "b")
        if "stem" not in new_key:
            new_key = new_key.replace("conv", "s")
        new_key = new_key.split(".")
        num = len(new_key)
        if key[-1] == "weight":
            if key[-2] in ["conv", "a", "b", "c", "proj"]:
                
                for i in range(num):
                    if i == 0:
                        curr_module = getattr(layer, new_key[i])
                    else:
                        curr_module = getattr(curr_module, new_key[i] )
                curr_module.weights[0].assign(tf.transpose(value , perm=[2,3,1,0]))
                to_delete.append(".".join(key))
            elif key[-2] in ["bn", "a_bn", "b_bn", "c_bn"]:
                curr_module = layer
                for i in range(num):
                    if i == 0:
                        curr_module = getattr(layer, new_key[i])
                    else:
                        curr_module = getattr(curr_module, new_key[i] )
                curr_module.weights[0].assign(value)
                to_delete.append(".".join(key))
        elif key[-1] == "bias":
            curr_module = layer
            for i in range(num):
                if i == 0:
                    curr_module = getattr(layer, new_key[i])
                else:
                    curr_module = getattr(curr_module, new_key[i] )
            curr_module.weights[1].assign(value)
            to_delete.append(".".join(key))
        elif key[-1] == "running_mean":
            curr_module = layer
            for i in range(num):
                if i == 0:
                    curr_module = getattr(layer, new_key[i])
                else:
                    curr_module = getattr(curr_module, new_key[i] )
            curr_module.weights[2].assign(value)
            to_delete.append(".".join(key))
        elif key[-1] == "running_var":
            curr_module = layer
            for i in range(num):
                if i == 0:
                    curr_module = getattr(layer, new_key[i])
                else:
                    curr_module = getattr(curr_module, new_key[i] )
            curr_module.weights[3].assign(value)
            to_delete.append(".".join(key))

    for k in to_delete:
        del weights[k]
    return layer, weights
