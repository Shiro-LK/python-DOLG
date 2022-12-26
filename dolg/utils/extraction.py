# -*- coding: utf-8 -*-
from torchvision import transforms
import os, sys
import cv2
import torch
import numpy as np


# Per-channel mean and SD values in RGB order paddle

_MEAN_paddle = [0.48145466, 0.4578275, 0.40821073]
_SD_paddle = [0.26862954, 0.26130258, 0.27577711]

# Per-channel mean and SD values in BGR order pt/tf
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

def process_data(filename, root_dir, mode="pt"):
    if mode == "pt":
        trf =  transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize(_MEAN, _SD)])
        img = cv2.resize(cv2.imread(os.path.join(root_dir,filename)), (512,512))
        img = trf(img)
        return img
    elif mode == "tf":
        img = cv2.resize(cv2.imread(os.path.join(root_dir,filename)), (512,512))
        img = img/255.0
        img = (img - np.array(_MEAN).reshape(1,1,-1))/np.array(_SD).reshape(1,1,-1)
        return img 
    elif mode == "paddle":
        img = cv2.resize(cv2.imread(os.path.join(root_dir,filename)), (512,512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        img = (img - np.array(_MEAN_paddle).reshape(1,1,-1))/np.array(_SD_paddle).reshape(1,1,-1)
        return img 
    else:
        raise("mode not recognize, choose pt or tf ")
def extract_embedding_pt(filename, root_dir, model):
    img = process_data(filename, root_dir, mode="pt")
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        output  = model(img )
    if type(output) == tuple:
        return output[0].cpu().numpy(), output[1].cpu().numpy()
    return output.cpu().numpy()


def extract_embedding_tf(filename, root_dir, model):
    img = process_data(filename, root_dir, mode="tf")
    img = np.expand_dims(img, axis=0)

    output  = model(img ) # model.predict(img) #
    if type(output) == tuple:
        return output[0].numpy(), output[1].numpy()
    return output.numpy()

def compute_similarity(a, b):
    a = a/np.linalg.norm(a, ord=2, axis=1)
    b = b/np.linalg.norm(b, ord=2, axis=1)

    return np.sum(a*b, axis=1)