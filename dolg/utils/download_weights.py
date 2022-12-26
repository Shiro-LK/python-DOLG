# -*- coding: utf-8 -*-

from huggingface_hub import hf_hub_url, cached_download
import requests
import os 

def get_checkpoint_path():
    p = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
    p = os.path.join(p, "weights")
    return p 

def download_checkpoint(model_name, backend="pt", folder=get_checkpoint_path()):
    REPO_ID = "Shiro/DOLG"
    
    if model_name not in ["r101", "r50"]:
        raise("model name wrong, accept only r101 or r50")
    
    if backend not in ["pt", "tf" ]:
        raise("backend accept only 'pt' or 'tf', change backend! ")
        
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    if model_name == "r50":
        if backend == "pt":
            FILENAME = "r50_dolg_512.pt"
        elif backend == "tf":
            FILENAME = "r50_dolg_512_from_pt.h5"
    elif model_name == "r101":
        if backend == "pt":
            FILENAME = "r101_dolg_512.pt"
        elif backend == "tf":
            FILENAME = "r101_dolg_512_from_pt.h5"
        
       
    response = requests.get(hf_hub_url(REPO_ID, FILENAME))
    
    open(os.path.join(folder, FILENAME), "wb").write(response.content)
    print(f"checkpoint {FILENAME} downloaded in {folder}")