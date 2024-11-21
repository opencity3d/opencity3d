import numpy as np
import sys
sys.path.append("..")
import torch
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms

def get_encode_fn(model_type):
    if model_type == "clip":
        checkpoint = 'laion2b_s34b_b88k'
        model_type = "ViT-B-16"
        model, _, preprocess = create_model_and_transforms(
                "ViT-B-16",
                pretrained=checkpoint,
                precision="fp16",
        ) 
        tokenizer = get_tokenizer(model_type)
    elif model_type == "siglip":
        checkpoint = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
        model, preprocess = create_model_from_pretrained(checkpoint)
        tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    def encode_text(texts, model, tokenizer):
        model.eval()
        texts_tok = tokenizer(texts, context_length=model.context_length)
        with torch.no_grad(), torch.cuda.amp.autocast():
            return model.encode_text(texts_tok).cpu().detach().numpy().astype(np.float16).T
    return model, tokenizer, encode_text