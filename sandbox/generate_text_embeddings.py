
from openclip_encoder import OpenCLIPNetwork
import torch 
import numpy as np
import json

def encode_queries(queries, out_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = OpenCLIPNetwork(device)
    
    prompt_embed = clip_model.encode_text(queries, "cuda").detach().cpu().numpy()
    embed_dict = {q: list(map(list, prompt_embed[i].astype(float).reshape(-1,1))) for i, q in enumerate(queries)}
    with open(out_file, 'w') as f:
        json.dump(embed_dict, f)
    

if __name__ == "__main__":
    encode_queries(["building", "trees, bushes or vegetation", "sports field", "bridge", "water (lake or sea)", "road", "skyscraper"], "prompts.json")
    encode_queries(["object", "things", "stuff", "texture"], "canonical.json")