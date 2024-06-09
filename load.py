import os
import orjson
import torch
import numpy as np
from model import TMR_textencoder
import mld
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from collections import OrderedDict

ckpt = "textencoder.ckpt"
modelpath = 'distilbert-base-uncased'

EMBS = "data/unit_motion_embs"


def load_json(path):
    with open(path, "rb") as ff:
        return orjson.loads(ff.read())


def load_keyids(split):
    path = os.path.join(EMBS, f"{split}.keyids")
    with open(path) as ff:
        keyids = np.array([x.strip() for x in ff.readlines()])
    return keyids


def load_keyids_splits(splits):
    return {
        split: load_keyids(split)
        for split in splits
    }


def load_unit_motion_embs(device):
    # path = os.path.join(EMBS, f"{split}_motion_embs_unit.npy")
    path = "./embeddings/motion_embedding.npy"
    tensor = torch.from_numpy(np.load(path)).to(device)
    path = "./TEST_embeddings/motion_embedding.npy"
    TEST_tensor = torch.from_numpy(np.load(path)).to(device)
    return {"all": tensor, "test":TEST_tensor}


# def load_unit_motion_embs_splits(splits, device):
#     return {
#         split: load_unit_motion_embs(split, device)
#         for split in splits
#     }


def load_model(device):
    # text_params = {
    #     'latent_dim': 256, 'ff_size': 1024, 'num_layers': 6, 'num_heads': 4,
    #     'activation': 'gelu', 'modelpath': 'distilbert-base-uncased'
    # }
    # "unit_motion_embs"
    # model = TMR_textencoder(**text_params)
    # state_dict = torch.load("data/textencoder.pt", map_location=device)
    # # load values for the transformer only
    # model.load_state_dict(state_dict, strict=False)
    # state_dict = torch.load(ckpt, map_location=torch.device(device))["state_dict"]
    state_dict = torch.load(ckpt, map_location=torch.device(device))
    vae_dict = OrderedDict()
        
    model = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, num_heads=4)
    # print(model)
    for k, v in state_dict.items():
        # print(k)
        if k.split(".")[0] == "textencoder":
            name = k.replace("textencoder.", "")
            vae_dict[name] = v
    # import pdb; pdb.set_trace()
    model.load_state_dict(vae_dict, strict=True)
    
    model = model.eval()
    return model
