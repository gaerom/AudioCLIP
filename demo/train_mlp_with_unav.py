""" UnAV-100으로 test """


import os
import sys
import glob

import librosa
import librosa.display
import wandb

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model import AudioCLIP # audio encoder
from model.model_final import FrozenCLIPTextEmbedder, FrozenOpenCLIPEmbedder, Mapping_Model # text encoder
from dataset import AudioDataset # dataset

from utils.transforms import ToTensor1D

torch.set_grad_enabled(True)
device = "cuda" if torch.cuda.is_available() else "cpu"

max_length = 77 # 이 부분 나중에 parser로 정리
input_dim = 1024
output_dim = 1024
epochs = 50

model = Mapping_Model(input_dim, output_dim, max_length).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
SAMPLE_RATE = 44100 


# audio encoder
audio_encoder = AudioCLIP(pretrained=f'../assets/{MODEL_FILENAME}').eval().to(device) # 학습시키는게 아니니까
# ViT-14
text_encoder = FrozenOpenCLIPEmbedder().eval().to(device)



# 아래 경로는 각각의 10초 raw audio를 2초 segment로 잘라서 저장해놓은 경로
audio_segments_path = '/home/broiron/Desktop/AudioCLIP/data/segments/*.wav' # 10개로 일단 test 
audio_paths = glob.glob(audio_segments_path)

# audio segments(.wav)를 numpy 배열로 load
audio_data = [] # ndarray
for path in tqdm(audio_paths, desc='Audio loading: '):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, dtype=np.float32)
    audio = audio.reshape(1, -1)
    audio_data.append(audio)



# # audio = list()
# audio_transforms = ToTensor1D()
# audio_data = [audio_transforms(librosa.load(path, sr=SAMPLE_RATE, dtype=np.float32)[0].reshape(1, -1)) for path in tqdm(audio_paths, desc='Audio loading: ')]


# Load labels from file
# dataset 파일명에서 추출한 label
labels_path = '/home/broiron/Desktop/AudioCLIP/data/label/labels_38423.txt' # test를 위한 label
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file]



### train / validation 분리
audio_data_train, audio_data_val, labels_train, labels_val = train_test_split(audio_data, labels, test_size=0.2, random_state=42)
print(f'train: {len(audio_data_train)}')

# Audio encoding function
def encode_audio_data(audio_data_list):
    encoded_features = []
    for audio_sample in tqdm(audio_data_list, desc='Audio encoding: '):
        with torch.no_grad():
            ((audio_features, _, _), _), _ = audio_encoder(audio=audio_sample) # demo에서 사용한 방식이랑 일치
            # audio_features = audio_encoder(audio_sample.unsqueeze(0).to(device))[0]
            # print(f'encoder 통과한 audio embedding shape: {audio_features.shape}') # [1, 1024]
            encoded_features.append(audio_features)
    return torch.stack(encoded_features)

# train, validation data 각각 encoding 처리
audio_features_train = encode_audio_data(audio_data_train) # embedding
audio_features_val = encode_audio_data(audio_data_val) # embedding
 torch.Size([1, 2048, 1, 1])
(tpos) broiron@broiron-desktop:~/Desktop/TPoS_last/audio_encoder$ 


# """ classification 결과 확인 """
# ### 1. Noralization of Embeddings
# audio_features_train = audio_features_train / torch.linalg.norm(audio_features_train, dim=-1, keepdim=True) # audio
# #print(f'nomalized audio embedding shape: {audio_features_train.shape}') # normalization 한다고 해서 차원이 달라지진 않음
# text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True) # text
# #print(f'text_features 확인: {text_features}')


# ### 2. Obtaining Logit Scales
# scale_audio_text = torch.clamp(audio_encoder.logit_scale_at.exp(), min=1.0, max=100.0)



# ### 3. Computing Similarities
# # audio와 text 간의 similality 계산
# logits_audio_text = scale_audio_text * audio_features_train @ text_features.T
# # print(logits_audio_text)
# print(f'logits_audio_text 차원 확인: {logits_audio_text.dim()}') # 왜 3차원..?, 2차원이어야 하는데


# ### 4. Classification

# print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')

# # calculate model confidence
# paths_to_audio = glob.glob('audio_segment/*.wav')
# confidence = logits_audio_text.softmax(dim=1) # confidence score부터가 1로 통일되어 있음 -> 여기서부터 문제

# print(f'confidence score: {confidence}')
# for audio_idx in range(len(paths_to_audio)):
#     # acquire Top-3 most similar results
#     conf_values, ids = confidence[audio_idx].topk(3)
    
#     ids_list = ids.tolist()  # 텐서를 리스트로 변환
#     conf_values_list = conf_values.tolist()  # 텐서를 리스트로 변환
    
#     # print(ids_list)
#     print(conf_values_list)

#     # format output strings
#     query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
#     results = ', '.join([f'{labels[i]:>15s} ({v:06.2%})' for i, v in zip(ids_list[0], conf_values_list[0])])
#     print(query + results)

# wandb 연동
wandb.init(project="audio2video")
cfg = {
    "learning_rate": 0.001,
    "epochs" : 100,
    "batch_size" : 32,
    "dropout": 0.2
}
wandb.config.update(cfg)

batch_size = 32
shuffle = True
num_workers = 0 # default: 0

# Data Loader
train_dataset = AudioDataset(audio_features_train, labels_train)
val_dataset = AudioDataset(audio_features_val, labels_val)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=shuffle, num_workers=num_workers)

val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers)


# validation
def evaluate(model, data_loader, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad(): 
        for audio_features, labels_batch in data_loader:
            audio_features = audio_features.to(device).float()
            text_embeddings = torch.stack([text_encoder.encode([label]).to(device).float() for label in labels_batch])

            predictions = model(audio_features)
            loss = loss_function(predictions, text_embeddings.squeeze(1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss



""" Audio embedding과 Text embedding MSE 적용"""
patience = 20
cnt = 0
best_loss = float('inf')  # 최소 손실 값을 저장하기 위한 변수 초기화
model_save_path = '/home/broiron/Desktop/AudioCLIP/assets/test.pth'  

# Measure time in PyTorch
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
    

# training
model.train()
for epoch in range(epochs):
    total_loss = 0
    for audio_features, labels_batch in tqdm(train_loader, desc='Training: '):
        # start time
        start.record()
        audio_features = audio_features.to(device).float() 

        # text embedding 추출
        text_embeddings = torch.stack([text_encoder.encode([label]).to(device).float() for label in labels_batch])
        # print(f'text embedding shape: {text_embeddings.squeeze(1).shape}')

        optimizer.zero_grad()
        predictions = model(audio_features) # MLP 통과한 embedding
        loss = loss_function(predictions, text_embeddings.squeeze(1)) # 앞서 stack으로 쌓아서 차원이 하나 더 생김(?) -> squeeze로 제거
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # end time 
    end.record()
    torch.cuda.synchronize()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = evaluate(model, val_loader, loss_function)

    print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}')

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved at epoch: {epoch+1}, loss: {best_loss}")
        cnt = 0
    else:
        cnt += 1

    if cnt >= patience:
        print(f"Early stopping {epoch+1}")
        break
    
    wandb.log({'train_loss':avg_train_loss, 'val_loss':avg_val_loss, 'time': start.elapsed_time(end)})
wandb.finish()








