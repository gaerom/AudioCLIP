# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, AutoTokenizer, CLIPVisionModel


model_id = "stabilityai/stable-diffusion-2-1-base"

text = ""

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False,)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
image_encoder = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# text
text_inputs = tokenizer(["a photo of a cat"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
text_features = text_encoder(**text_inputs)   # [1, 77, 1024]
print(f'text embedding shape: {text_features.last_hidden_state.shape}')

# image
# image_inputs = image_processor(images=image, return_tensors="pt")
# image_features = image_encoder(**image_inputs)    # [1, 257, 1280]
# print(image_features.last_hidden_state.shape)

# image_features.save("astronaut_rides_horse.png")