import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
from PIL import Image

import torch
import torchvision
from torch import nn

import pickle
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms # works on open-clip-torch>=2.23.0, timm>=0.9.8
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


class SigLipNetwork:
    """class for SigLip model"""
    def __init__(self, device):
        self.checkpoint = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
        self.model, self.preprocess = create_model_from_pretrained(self.checkpoint)
        self.tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
        self.model = self.model.to(device)
        self.model.half().eval()
        self.device = device
        self.clip_n_dims = 1152

    def encode_text(self, texts):
        with torch.no_grad(), torch.cuda.amp.autocast():
            return self.model.encode_text(self.tokenizer(texts, context_length=model.context_length))
    
    def encode_image(self,images, batch_size=128):
        pil_images =  [Image.fromarray((i*255).detach().cpu().numpy().transpose(1,2, 0).astype(np.uint8)) for i in images] # ugly af
        torch_images = torch.stack(list(map(self.preprocess, pil_images)), axis=0).half().to(self.device)

        embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in range(0, len(images), batch_size):
                embeddings.append(self.model.encode_image(torch_images[i:i+batch_size]))
        return torch.cat(embeddings, axis=0)

@dataclass
class OpenCLIPNetworkConfig:
    """Configuration for OpenCLIPNetwork"""
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    """class for OpenCLIP model"""
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input, batch_size=128):
        processed_input = self.process(input).half()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(input), batch_size):
                embeddings.append(self.model.encode_image(processed_input[i:i + batch_size]))
        return torch.cat(embeddings, axis=0)





def create(image_list, data_list, save_folder, use_cached_masks=False, overwrite=False, mode="bbox"):
    """
    Creates OpeCity features for a list of images.

    Args:
        image_list (list): A list of images.
        data_list (list): A list of data corresponding to each image.
        save_folder (str): The folder to save the generated features.
        use_cached_masks (bool, optional): Whether to use cached masks. Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing features. Defaults to False.
        mode (str, optional): The mode for embedding images. Defaults to "bbox".

    Returns:
        None
    """
    assert image_list is not None, "image_list must be provided to generate features"
    # embed_size=512
    # seg_maps = []
    total_lengths = []
    timer = 0
    # img_embeds = torch.zeros((len(image_list), 300, embed_size))
    # seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", total=len(image_list)):
        
        print(f"Embedding {data_list[i]} ----------------------------")
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        if not overwrite and os.path.exists(save_path + '_s.npy') and os.path.exists(save_path + '_f.npy'):
            print(f"Skipping {data_list[i]}")
            continue
        timer += 1
        try:
            if img.dim() == 3:
                _img = img.unsqueeze(0)
            else:
                _img = img
            if use_cached_masks:
                feature_dir = save_folder.split("/")[-1]
                img_cache_dir = save_path.replace(feature_dir, f"cached_masks_{feature_dir}")
                # @TODO: actually delete cached masks, add flag 
                # if overwrite: # remove old cache
                #     if os.path.exists(img_cache_dir):
                #         os.system(f"rm -rf {img_cache_dir}")
            else:
                img_cache_dir = None

            img_embed, seg_map = _embed_clip_sam_tiles(_img, sam_encoder, level=mode, img_cache_dir=img_cache_dir)      

            lengths = [len(v) for k, v in img_embed.items()]
            total_length = sum(lengths)
            total_lengths.append(total_length)
            
            # if total_length > img_embeds.shape[1]:
            #     pad = total_length - img_embeds.shape[1]
            #     img_embeds = torch.cat([
            #         img_embeds,
            #         torch.zeros((len(image_list), pad, embed_size))
            #     ], dim=1)

            img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
            assert img_embed.shape[0] == total_length
            # img_embeds[i, :total_length] = img_embed
            
            seg_map_tensor = []
            lengths_cumsum = lengths.copy()
            for j in range(1, len(lengths)):
                lengths_cumsum[j] += lengths_cumsum[j-1]
            for j, (k, v) in enumerate(seg_map.items()):
                if j == 0:
                    seg_map_tensor.append(torch.from_numpy(v))
                    continue
                assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
                v[v != -1] += lengths_cumsum[j-1]
                seg_map_tensor.append(torch.from_numpy(v))
            seg_map = torch.stack(seg_map_tensor, dim=0)
            # seg_maps[i] = seg_map

            curr = {
                'feature': img_embed,
                'seg_maps': seg_map
            }
            sava_numpy(save_path, curr)
        except Exception as e:
            print(f"Error in {data_list[i]}: {e}")

    mask_generator.predictor.model.to('cpu')
        

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder, img_cache_dir=None, level="bbox"):
    """
    Embeds the given image patch using the CLIP model.

    Args:
        image (torch.Tensor): The input image to embed.
        sam_encoder: The SAM encoder model.
        img_cache_dir (str, optional): The directory to cache the segmented images. Defaults to None.
        level (str, optional): The level of segmentation. Defaults to "bbox".

    Returns:
        tuple: A tuple containing the CLIP embeddings for different levels of segmentation and the segmentation map.
    """
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs, level, img_cache_dir=img_cache_dir)

    clip_embeds = {}
    for level in tqdm(['default', 's', 'm', 'l'], desc="applying CLIP to crops"):
        tiles = seg_images[level]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[level] = clip_embed.detach().cpu().half()        
    return clip_embeds, seg_map

def get_seg_img(mask, image, mode="bbox", pad=25):
    """
    Get the segmented image patch, cut-out based on the given mask and image.

    Parameters:
    - mask: A dictionary containing the segmentation mask information.
    - image: The original image.
    - mode: The mode for cropping the image. Possible values are "bbox", "highlight", and "highlight_negative". Default is "bbox".
    - pad: The padding value for cropping the image. Default is 25.

    Returns:
    - seg_img: The segmented image based on the given mask and mode.
    """
    if mode == "bbox":
        # bbox-based cropping of images
        image = image.copy()
        image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
        x,y,w,h = np.int32(mask['bbox'])
        seg_img = image[y:y+h, x:x+w, ...]
        return seg_img
    elif mode == "highlight":
        image = image.copy()
        # highlight the mask in the image
        image[mask['segmentation']==0] = ((image[mask['segmentation']==0].astype(np.float32) + 2*255) / 3).astype(np.uint8)
        
        # crop to roughly the bbox + a lot of padding
        x,y,w,h = np.int32(mask['bbox'])
        seg_img = image[max(0, y-pad):min(image.shape[0], y+h+pad), max(0, x-pad):min(image.shape[1], x+w+pad), ...]
        seg_mask = (mask['segmentation']==0)[max(0, y-pad):min(image.shape[0], y+h+pad), max(0, x-pad):min(image.shape[1], x+w+pad), ...]
        contours, hierarchy = cv2.findContours((255-seg_mask*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        seg_img = cv2.drawContours(seg_img, contours, -1, (255,0,0), 3)
        
        # if (mask['segmentation']!=0).sum() / (mask['segmentation']==0).sum() > 0.1:
        #     pil_img = Image.fromarray(seg_img)
        #     pil_img = pil_img.convert("RGB")
        #     ratio = (mask['segmentation']!=0).sum() / (mask['segmentation']==0).sum()
        #     pil_img.save(f"highlight_{ratio}.png")
        return seg_img
    elif mode == "highlight_negative":
        image = image.copy()    
        # crop to roughly the bbox + a lot of padding
        x,y,w,h = np.int32(mask['bbox'])
        seg_img = image[max(0, y-pad):min(image.shape[0], y+h+pad), max(0, x-pad):min(image.shape[1], x+w+pad), ...]
        return seg_img
    

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result, min_size = 0.0025) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        seg = m["segmentation"]
        if i in keep and seg.sum() > seg.shape[0] * seg.shape[1] * min_size: result_keep.append(m)
    return result_keep

def bboxes_intersect(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1)

def mask_nms(masks, scores, bboxes=None, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, quick=True, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        bboxes (torch.Tensor, optional): The bounding boxes of the masks, has shape (num_masks, 4).
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)
    if bboxes is not None:
        bboxes_ord = bboxes[idx.view(-1), :]

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)

    for i in range(num_masks):
        for j in range(i, num_masks):
            # skip if bboxes don't intersect
            if quick and bboxes_ord is not None and not bboxes_intersect(bboxes_ord[i], bboxes_ord[j]):
                continue
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou


    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    """
    Update masks by removing redundant masks based on the scores and overlap rate between masks.

    Args:
        *args: Variable length argument list containing the masks at different levels.
        **kwargs: Keyword arguments for additional parameters.

    Returns:
        Tuple: Updated masks after removing redundant masks.

    """
    masks_new = ()
    for masks_lvl in tqdm(args, desc="postprocessing masks (NMS + filtering)"):
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
        bboxes = torch.from_numpy(np.stack([m['bbox'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, bboxes=bboxes, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)
        for mask in masks_lvl:
            for key in list(masks_lvl[0].keys()):
                if key not in ['segmentation', 'predicted_iou', 'stability_score', 'bbox']:
                    del mask[key]

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image, mode="bbox", img_cache_dir=None):
    """
    Encode the given image into segmented images and maps using the Selective Attention Masking (SAM) algorithm.

    Parameters:
        image (numpy.ndarray): The input image to be encoded.
        mode (str, optional): The mode for encoding. Defaults to "bbox".
        img_cache_dir (str, optional): The directory to cache pre-computed masks. Defaults to None.

    Returns:
        tuple: A tuple containing the segmented images and maps.
            - seg_images (dict): A dictionary of segmented images, where the keys are the sizes of the masks ('default', 's', 'm', 'l').
            - seg_maps (dict): A dictionary of segmented maps, where the keys are the sizes of the masks ('default', 's', 'm', 'l').
    """
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    if img_cache_dir is not None and os.path.exists(img_cache_dir):
        masks_default, masks_s, masks_m, masks_l = pickle.load(open(os.path.join(img_cache_dir, "masks.pkl"), "rb"))
    else:
        # pre-compute masks
        masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
        # pre-compute postprocess
        masks_default, masks_s, masks_m, masks_l = \
            masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        if img_cache_dir is not None and not os.path.exists(img_cache_dir):
            os.makedirs(img_cache_dir, exist_ok=True)
            pickle.dump((masks_default, masks_s, masks_m, masks_l), open(os.path.join(img_cache_dir, "masks.pkl"), "wb"))


    def mask2segmap(masks, image, mode):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in tqdm(range(len(masks)), desc="generating mask crops"):
            mask = masks[i]
            seg_img = get_seg_img(mask, image, mode=mode)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)
            # @TODO rewrite this to be more storage efficient (one single image of indices)
            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map
    
    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image, mode=mode)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image, mode=mode)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image, mode=mode)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image, mode=mode)

    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--mode', type=str, default="highlight")
    parser.add_argument('--use_cached_masks', type=bool, default=False)
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--model', type=str, default="siglip")
    # parser.add_argument('--overwrite_cache', type=bool, default=False)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    mode = args.mode
    use_cached_masks = args.use_cached_masks
    overwrite = args.overwrite
    # overwrite_cache = args.overwrite_cache
    if use_cached_masks:
        print("[ INFO ] Using cached masks, caching new seg results on cache miss")
        if overwrite:
            print("[ INFO ] Overwriting existing mask cache")
    else:
        print("[ INFO ] Recomputing masks")
    if overwrite:
        print("[ INFO ] Overwriting existing features")
    else:
        print("[ INFO ]  Reusing features")
    print(f"[ INFO ] Embedding mode: {mode}")
    img_folder = os.path.join(dataset_path, 'images')
    if not os.path.isdir(img_folder):
        img_folder = os.path.join(dataset_path, 'color')
    data_list = [f for f in sorted(os.listdir(img_folder))]
    data_list.sort()
    # random order
    data_list = np.random.permutation(data_list).tolist()

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig) if args.model == "clip" else SigLipNetwork("cuda")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    #imgs = torch.cat(images)

    random_perm = np.random.permutation(len(images))
    images = [images[i] for i in random_perm]
    data_list = [data_list[i] for i in random_perm]


    seed_num = 42
    seed_everything(seed_num)
    if mode == 'bbox':
        save_folder = os.path.join(dataset_path, 'language_features')
    else:
        save_folder = os.path.join(dataset_path, f'language_features_{mode}')
    os.makedirs(save_folder, exist_ok=True)
    create(images, data_list, save_folder, use_cached_masks=use_cached_masks, overwrite=overwrite, mode=mode)