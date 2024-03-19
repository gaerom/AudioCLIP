import torch 
import timm
import clip # 1024로 바꾸기 위해
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from tqdm import tqdm
import open_clip

import json

class FrozenCLIPTextEmbedder(torch.nn.Module):
    def __init__(self, version='RN50', device="cuda", max_length=77, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, device=device)
        self.max_length = max_length
        self.normalize = normalize

    def forward(self, text: Union[str, List[str]]):
        device = next(self.model.parameters()).device
        if isinstance(text, list):
            embeddings = [] # list 처리
            for t in text:
                tokens = clip.tokenize([t], context_length=self.max_length).to(device)  # 각 text -> tokenization
                embedding = self.model.encode_text(tokens)  
                if self.normalize:
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding)
            embeddings = torch.cat(embeddings, dim=0)  # 모든 embedding을 하나의 tensor로 결합
        else:
            tokens = clip.tokenize([text], context_length=self.max_length).to(device)
            embeddings = self.model.encode_text(tokens)
            if self.normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def encode(self, text: Union[str, List[str]]):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = z.repeat(1, self.max_length, 1) # 두 번째 인자가 n.repeat 이었던 것을 max_length로 수정
        return z
    
class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Codes from VideoComposer
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", pretrained="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        
        model, _, _ = open_clip.create_model_and_transforms(arch, device=device, pretrained=pretrained)
        
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text) 
        # tokens.shape: torch.Size([1, 77])
        z = self.encode_with_transformer(tokens.to(self.device)) 
        # z.shape: torch.Size([1, 77, 1024])
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
        

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

max_length = 77
input_dim = 1024
output_dim = 1024


class Mapping_Model(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, max_length=77):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim * 2)  
        self.linear2 = nn.Linear(output_dim * 2, output_dim * max_length) 
        self.act = nn.ReLU()  
        self.drop = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.drop(self.act(self.linear1(x)))
        x = self.linear2(x)
        return x.view(-1, max_length, output_dim) 


    
# def load_annotations(json_file):
#     with open(json_file, 'r') as f:
#         annotations = json.load(f)
#     return annotations



# def process_annotations(annotations, clip_text_embedder):
#     for video_id, video_info in tqdm(annotations['database'].items(), desc="Processing GT annotations"):
#         for annotation in video_info['annotations']:
#             start_time, end_time = annotation['segment']
#             label = annotation['label']
#             text_description = label 
#             text_embedding = clip_text_embedder.encode(text_description)
#             # print(f'Label: {label}, Embedding: {text_embedding.shape}') # 어느 동영상에 해당되는 label인지도 알 수 있게 코드 수정
    
    
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_text_embedder = FrozenCLIPTextEmbedder(device=device)

# json_file = '/home/broiron/Desktop/TPoS/dataset/unav100_annotations.json' # UnAV-100 annotations.json 가져오기
# annotations = load_annotations(json_file)
# process_annotations(annotations, clip_text_embedder)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


