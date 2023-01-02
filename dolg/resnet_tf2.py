#!/usr/bin/env python3

"""ResNe(X)t model backbones.
    Convert paddle to tf2   
"""

import math 
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, ZeroPadding2D
from tensorflow.keras.layers import MaxPool2D, Activation
# Stage depths for ImageNet models
_IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]
        
def adaptative_avg_pool2d(x, output_dim=1):
    # Dim batch = (N, H, W, C),
    # from stackoverflow : https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
    if type(output_dim) == int:
        output_dim = (output_dim, output_dim)
    shape = x.shape 
    
    stride = (shape[1]//output_dim[0], shape[2]//output_dim[1]) 
    
    h, w = shape[1]- (output_dim[0]-1) * stride[0],  shape[2] -(output_dim[1]-1)*stride[1]
    return tf.cast(tf.nn.avg_pool2d(input=x, ksize=(h, w), strides=stride, padding="VALID" ), x.dtype)
    
class AdaptiveAvgPool2d(tf.keras.layers.Layer):
    def __init__(self, output_dim=1, **kwargs):
        super(AdaptiveAvgPool2d, self).__init__(**kwargs)
        self.output_dim = output_dim
        if type(output_dim) == int:
            self.output_dim = (output_dim, output_dim)

    def call(self, x):
        shape = x.shape 
        stride = (shape[1]//self.output_dim[0], shape[2]//self.output_dim[1]) 
        h, w = shape[1]- (self.output_dim[0]-1) * stride[0],  shape[2] -(self.output_dim[1]-1)*stride[1]
        return tf.nn.avg_pool2d(input=x, ksize=(h, w), strides=stride, padding="VALID" )

class GeneralizedMeanPooling(tf.keras.layers.Layer):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6, **kwargs):
        """ just for infer """
        super(GeneralizedMeanPooling, self).__init__(**kwargs)
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def call(self, x):
        """ """
        out = tf.math.pow(tf.clip_by_value(x, clip_value_min=self.eps, clip_value_max=x.dtype.max),  y=tf.cast(self.p,x.dtype))
        return tf.math.pow(adaptative_avg_pool2d(out, self.output_size), tf.cast(1. / self.p, out.dtype))
    def __repr__(self):
        """ """
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
    
    



class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6,   **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps,  **kwargs)
        self.p = tf.Variable(initial_value= tf.ones(1) * norm, trainable=True, name="p")
        
class BasicTransform(Layer):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1,
                 bn_eps=1e-5 , bn_mom=0.1, **kwargs):
        """ """
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__(**kwargs)
        self.a_pad = ZeroPadding2D(1, name="a_pad")
        self.a = Conv2D(w_out, kernel_size=3, strides=stride, padding="valid", use_bias=False, name="a")
        self.a_bn = BatchNormalization(epsilon=bn_eps, momentum=bn_mom, name="a_bn")
        self.a_relu = Activation("relu", name="a_relu")
        self.b = Conv2D(w_out, kernel_size=3, strides=1, padding=1, use_bias=False, name="b")
        self.b_bn = BatchNormalization(epsilon=bn_eps, momentum=bn_mom, name="b_bn")
        self.b_bn.final_bn = True

    def call(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckTransform(Layer):
    """Bottleneck transformation: 1x1, BN, ReLU, 3x3, BN, ReLU, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, num_gs, bn_eps=1e-5, bn_mom=0.1, 
                 stride_1x1=True):
        """ 
            @stride_1x1 :  Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
            @bn_mom : BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
            @bn_epsilon : BN epsilon
        """
        super(BottleneckTransform, self).__init__()
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (s1, s3) = (stride, 1) if stride_1x1 else (1, stride)
        self.a = Conv2D(w_b, 1, strides=s1, padding="valid", use_bias=False, name="a")
        self.a_bn = BatchNormalization(epsilon=bn_eps, momentum=bn_mom, name="a_bn")
        self.a_relu = Activation("relu", name="a_relu")
        self.b_pad = ZeroPadding2D(1, name="b_pad")
        self.b = Conv2D(w_b, 3, strides=s3, padding="valid", groups=num_gs, use_bias=False, name="b")
        self.b_bn = BatchNormalization(epsilon=bn_eps, momentum=bn_mom, name="b_bn")
        self.b_relu =Activation("relu", name="b_relu")
        self.c = Conv2D(w_out, 1, strides=1, padding="valid", use_bias=False, name="c")
        self.c_bn = BatchNormalization( epsilon=bn_eps, momentum=bn_mom, name="c_bn")
        self.c_bn.final_bn = True

    def call(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b_pad(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(Layer):
    """Residual block: x + F(x).
    
        @bn_mom : BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
        @bn_epsilon : BN epsilon
    """

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1, bn_eps=1e-5, 
                 bn_mom=0.1):
        """ """
        super(ResBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = Conv2D(w_out, 1, strides=stride, padding="valid", use_bias=False, name="proj")
            self.bn = BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name="bn")
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs)
        self.relu = Activation("relu", name="relu")

    def call(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStage(Layer):
    """Stage of ResNet."""
    # ResStage(256, 512, stride=2, d=4, w_b=64 * 2, num_gs=1)
    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1, 
                 trans_func="bottleneck_transform"):
        """ """
        super(ResStage, self).__init__()
        self.d = d
        self.layers = []
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(trans_func)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            #self.add_module("b{}".format(i + 1), res_block)
            #self.layers.append(res_block)
            setattr(self, "b{}".format(i + 1), res_block)
    def call(self, x):
        """ """
        #for block in self.children():
        #    x = block(x)
        
        for i in range(self.d):
            #x = self.layers[i](x)
            x = getattr(self, "b" + str(i+1))(x)
            #if i == 1:
            #print(x)
        return x


class ResStemIN(Layer):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out, bn_eps=1e-5, bn_mom=0.1):
        """ """
        super(ResStemIN, self).__init__()
        self.conv_pad = ZeroPadding2D(3, name="conv_pad")
        self.conv = Conv2D(w_out, 7, strides=2, padding="valid", use_bias=False, name="conv")
        self.bn = BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name="bn")
        self.relu = Activation("relu", name="relu")
        self.pad_pool = ZeroPadding2D(1, name="pad_pool")
        self.pool = MaxPool2D(3, strides=2, padding="valid", name="pool")

    def call(self, x):
        """ """
        x = self.conv_pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pad_pool(x)
        x = self.pool(x)
        
        return x


class ResNet(tf.keras.Model):
    """ResNet model."""

    def __init__(self, depth, num_groups=1, width_per_group=64, bn_eps=1e-5, 
                 bn_mom=0.1, trans_fun="bottleneck_transform", shape=(512,512,3), **kwargs):
        """ """
        super(ResNet, self).__init__(**kwargs)
        self._construct(depth, num_groups, width_per_group, bn_eps, bn_mom, trans_fun)
        #self = self.build_graph(shape)

    def _construct(self, depth, num_groups, width_per_group, bn_eps, bn_mom, trans_fun):
        """ """
        g, gw = num_groups, width_per_group
        (d1, d2, d3, d4) = _IN_STAGE_DS[depth]
        w_b = gw * g
        # d2  = 4,  w_b=64 g = 1
        self.stem = ResStemIN(3, 64, bn_eps=bn_eps, bn_mom=bn_mom)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g, trans_func=trans_fun)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g, trans_func=trans_fun)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g, trans_func=trans_fun)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g, trans_func=trans_fun)

    def call(self, x):
        """ """
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        return x3, x4
    
    def build_graph(self, shape=(224, 224, 3)):
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))