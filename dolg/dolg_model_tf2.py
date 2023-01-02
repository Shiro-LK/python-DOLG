# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, ReLU
from tensorflow.keras.layers import Activation , BatchNormalization, LeakyReLU
from .resnet_tf2 import AdaptiveAvgPool2d, GeneralizedMeanPoolingP
from .utils.download_weights import get_checkpoint_path, download_checkpoint
import math 

class Softplus(tf.keras.layers.Layer):
    def __init__(self, beta=1, threshold=20, **kwargs):
        super(Softplus, self).__init__(**kwargs)
        self.beta = beta 
        self.threshold = threshold

    def call(self, x):
        
        linear_ids = tf.math.greater(x, self.threshold)
        softplus_ids = tf.math.logical_not(linear_ids)
        
        linear_value = x * tf.cast(linear_ids, x.dtype)
        softplus_value = (1.0/self.beta) * tf.math.log( 1 + tf.math.exp(self.beta * x) ) * tf.cast(softplus_ids, x.dtype)
        
        return linear_value + softplus_value

def load_pretrained_weights(model, pretrained ):
    path_weight = get_checkpoint_path()
    if pretrained=="r50":
        filename = "r50_dolg_512_from_pt.h5"
        if not os.path.exists(path_weight) or filename not in os.listdir(path_weight):
            download_checkpoint(model_name=pretrained, backend="tf")
        model.load_weights(os.path.join(path_weight,  
                                        filename), by_name=True, skip_mismatch=True)
    elif pretrained == "r101":
        filename = "r101_dolg_512_from_pt.h5"
        if not os.path.exists(path_weight) or filename not in os.listdir(path_weight):
            download_checkpoint(model_name=pretrained, backend="tf")
        model.load_weights(os.path.join(path_weight, 
                                        filename), by_name=True, skip_mismatch=True )
    else:
        raise(f"{pretrained} does not exist as a pretrained weight")
        
class DOLG(tf.keras.Model):
    """ DOLG model """
    def __init__(self, backbone, s4_dim=2048, s3_dim=1024, s2_dim=512, head_reduction_dim=512,
                 with_ma=False, num_classes=81313, pretrained=None, shape=(512,512,3), **kwargs):
        """ """
        super(DOLG, self).__init__(**kwargs)
        
        self.MODEL_S4_DIM = s4_dim
        self.MODEL_S3_DIM = s3_dim
        self.MODEL_S2_DIM = s2_dim
        self.num_class = num_classes
        
        self.pool_l = AdaptiveAvgPool2d((1,1), name="pool_l") 
        self.pool_g = GeneralizedMeanPoolingP(norm=3.0, name="pool_g")
        self.fc_t = Dense(self.MODEL_S3_DIM, name="fc_t", use_bias=True)  
        self.fc = Dense(head_reduction_dim, name="fc", use_bias=True) 
        self.globalmodel = backbone
        self.localmodel = SpatialAttention2d(self.MODEL_S3_DIM, act_fn='relu', with_aspp=with_ma, name="localmodel")
        self.desc_cls = Arcface(head_reduction_dim, num_classes, name="desc_cls") if num_classes is not None and type(num_classes) == int else None
        
        
        self.build((1, *shape))
        
        if pretrained is not None:
            load_pretrained_weights(self, pretrained)
        
        self = self.build_graph(shape)
        
       
    def call(self, x, targets=None):
        """ Global and local orthogonal fusion """
        f3, f4 = self.globalmodel(x)
        fl, _ = self.localmodel(f3)

        fg_o = self.pool_g(f4)
        
        fg_o = tf.reshape(fg_o,[-1, self.MODEL_S4_DIM]) 
        
        fg = self.fc_t(fg_o)
        fg_norm =tf.norm(fg, ord=2, axis=-1)  

        proj = tf.matmul(tf.expand_dims(fg, axis=1), tf.transpose(tf.reshape(fl, [-1, fl.shape[1]* fl.shape[2], fl.shape[-1]]), [0,2,1])) 
        proj = tf.transpose(tf.reshape(tf.matmul(tf.expand_dims(fg, axis=2), proj), [-1, fl.shape[3], fl.shape[1], fl.shape[2]]) , [0,2,3, 1]) 
        proj = proj / tf.reshape(fg_norm * fg_norm, [-1, 1, 1, 1])
        orth_comp = fl - proj
        fo = self.pool_l(orth_comp)
        
        fo =   tf.reshape(fo,[-1, self.MODEL_S3_DIM])
        global_feat = tf.concat((fg, fo), axis=-1)
        global_feat = self.fc(global_feat)
        if targets is not None or self.desc_cls is not None:
            global_logits = self.desc_cls(global_feat, targets)
            return global_feat, global_logits
        return global_feat
    
    def build_graph(self, shape=(224, 224, 3)):
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name="dolg")
    
   

class SpatialAttention2d(tf.keras.layers.Layer):
    """ SpatialAttention2D """
    def __init__(self, in_c, act_fn='relu', with_aspp=False, bn_eps=1e-5, 
                 bn_mom=0.1, **kwargs): # 
        """ """
        super(SpatialAttention2d, self).__init__(**kwargs)

        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(in_c, name="aspp")
        self.conv1 =  Conv2D(in_c, 1, 1, name="conv1" )
        self.bn = BatchNormalization( momentum=bn_mom, epsilon=bn_eps, name="bn")
        if act_fn.lower() in ['relu']:
            self.act1 = Activation("relu")
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = LeakyReLU()
        self.conv2 =  Conv2D( 1, 1, 1, name="conv2" )
        self.softplus = Softplus(beta=1, threshold=20, name="softplus") 

    def call(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        if self.with_aspp:
            x = self.aspp(x)
            
        x = self.conv1(x)
        x = self.bn(x)
        feature_map_norm = tf.linalg.normalize(x, ord=2, axis=-1)[0] 
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = tf.tile(att_score, [1,1,1, feature_map_norm.shape[-1] ])
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        """ """
        return self.__class__.__name__
    
    
class ASPP(tf.keras.layers.Layer):
    """ Atrous Spatial Pyramid Pooling Module """
    def __init__(self, out_dim, **kwargs):
        """ """
        super(ASPP, self).__init__(**kwargs)

        self.aspp = []
        self.aspp.append(Conv2D(512, 1, 1))

        for dilation in [6, 12, 18]:
            self.aspp.append(Conv2D(512, 3, 1, padding="same", dilation_rate=dilation)) # self.aspp.append(nn.Conv2D(in_c, 512, 3, 1, padding=_padding, dilation=dilation))

        self.im_pool = tf.keras.Sequential([AdaptiveAvgPool2d(1),
                                     Conv2D(512, 1, 1),
                                     ReLU()])
        self.conv_after = tf.keras.Sequential( [Conv2D(1024, 1, 1),  ReLU()])

    def call(self, x):
        """  """
        h, w = x.shape[1], x.shape[2]
        aspp_out =[tf.image.resize(images=self.im_pool(x), size=(h, w), method="bilinear", preserve_aspect_ratio=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = tf.concat(aspp_out, -1)
        x = self.conv_after(aspp_out)
        return x

class Arcface(tf.keras.layers.Layer):
    """ Additive Angular Margin Loss """
    def __init__(self, in_feat, num_classes, scale=128, margin=0.15, **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = tf.Variable(tf.zeros((in_feat, num_classes)), trainable=True, name="weight")
        self.t = tf.Variable([0.], trainable=False, name="t")
        
    def call(self, features, targets=None, training=False):
        """
            @targets : (batch dim, 1)
        """
        # get cos(theta)
        cos_theta = tf.tensordot(tf.linalg.normalize(features, axis=-1)[0] , tf.linalg.normalize(self.weight, axis=0)[0], axes=1)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)  # for numerical stability
        if targets is None:
            return cos_theta * self._s

        target_logit = tf.gather(cos_theta, targets, axis=1, batch_dims=1)

        sin_theta = tf.math.sqrt(1.0 - tf.math.pow(target_logit, tf.cast(2, target_logit.dtype)))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = tf.greater(cos_theta , cos_theta_m)
        final_target_logit = tf.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = tf.boolean_mask(cos_theta, mask)

        self.t.assign(tf.reduce_mean(target_logit) * 0.01 + (1 - 0.01) * self.t)
        hard_example = hard_example * (self.t + hard_example)
        cos_theta = tf.tensor_scatter_nd_update(cos_theta, tf.where(mask), hard_example)

        idx = tf.concat([tf.reshape(tf.range(len(targets)), (-1,1)), targets], axis=-1)
        cos_theta = tf.tensor_scatter_nd_update(cos_theta, idx , tf.squeeze(final_target_logit, axis=-1))
        pred_class_logits = cos_theta * self._s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )