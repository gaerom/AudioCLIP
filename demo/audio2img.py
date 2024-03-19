""" 저장한 output embedding load해서 diffusion 생성 결과 확인
diffusion model:  https://huggingface.co/stabilityai/stable-diffusion-2#examples """
import torch
import numpy as np
from diffusion import EmbeddingToImageDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

diffusion_model_id = "stabilityai/stable-diffusion-2"
embedding_to_image = EmbeddingToImageDiffusion(diffusion_model_id, device)


# load output embeddings
embeddings = '/home/broiron/Desktop/AudioCLIP/results_output/output_embeddings.npy' # evaluation에서 추출한 embedding
# embeddings = '../results_output/output_embeddings.npy'

loaded_embeddings = np.load(embeddings)
# print(f'loaded_embeddings shape: {loaded_embeddings.shape}')
embedding_tensor = torch.tensor(loaded_embeddings).to(device)  # tensor로 변환


generated_images = embedding_to_image.generate_images(embedding_tensor)

# save images
output_dir = '../results_output/output_embeddings'
embedding_to_image.save_images(generated_images, output_dir)