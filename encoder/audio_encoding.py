# 원본
# import sys
# import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# import numpy as np
# import torch
# from glob import glob
# from model.audioclip import AudioCLIP 
# from utils.transforms import ToTensor1D

# mel_dir = '/home/broiron/Desktop/AudioCLIP/data/UnAV/data/unav100/mel_train'  # 일단 mel_train만 적용 
# npy_files = glob(mel_dir + '/*.npy')
# num_frames = 5 

# soundclip = AudioCLIP(pretrained='../assets/AudioCLIP-Full-Training.pt') 

# def segment_and_encode(mel_spectrogram):
#     total_frames = mel_spectrogram.shape[1]
#     frame_per_segment = total_frames // num_frames
#     embeddings = []

#     for i in range(num_frames):
#         start_frame = i * frame_per_segment
#         end_frame = start_frame + frame_per_segment
#         segment = mel_spectrogram[:, start_frame:end_frame]

#         # # 세그먼트의 차원을 확인하고 필요한 경우 패딩 추가
#         # required_frames = 44  # 오디오 인코더에 들어가기 전 필요한 프레임 수
#         # if segment.shape[1] < required_frames:
#         #     # 부족한 프레임 수만큼 0으로 패딩
#         #     padding = np.zeros((segment.shape[0], required_frames - segment.shape[1]))
#         #     segment = np.concatenate((segment, padding), axis=1)
    
        
#         # audio encoder input으로 쓰게 torch tensor로 변환, 차원 변경
#         segment_tensor = torch.tensor(segment).unsqueeze(0).unsqueeze(0)  # 차원을 [1, 1, 224, 44]로 조정
        
#         # audio embedding 추출
#         with torch.no_grad():
#             # embedding = soundclip.encode_audio(segment_tensor.float())
#             # embeddings.append(embedding.cpu().numpy())
#             audio_features = soundclip(segment_tensor)
#             audio_features = audio_features.mean(axis=0, keepdim=True)
#             audio_features /= audio_features.norm(dim=-1, keepdim=True)
#             print("audio_features dim : ", audio_features.ndim)
#             print("audio_features shape : ", audio_features.shape)
            

#     return np.array(embeddings)


# # 각각의 mel_spectrogram에 대해 처리
# for npy_file in npy_files:
#     mel_spectrogram = np.load(npy_file)
#     embeddings = segment_and_encode(mel_spectrogram) 
#     print(f'File: {npy_file}, Embeddings shape: {embeddings.shape}')

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import torch
from glob import glob
from model.audioclip import AudioCLIP

mel_dir = '/home/broiron/Desktop/AudioCLIP/data/UnAV/data/unav100/mel_train'
npy_files = glob(mel_dir + '/*.npy')
num_frames = 5

soundclip = AudioCLIP(pretrained='../assets/AudioCLIP-Full-Training.pt')

def segment_and_encode(mel_spectrogram):
    total_frames = mel_spectrogram.shape[-1]  # 마지막 차원이 프레임 수
    frame_per_segment = total_frames // num_frames
    embeddings = []

    for i in range(num_frames):
        start_frame = i * frame_per_segment
        end_frame = start_frame + frame_per_segment
        # print(end_frame - start_frame)
        segment = mel_spectrogram[:, :, :, start_frame:end_frame]  # [1, 1, 224, segment_length]

        # audio encoder input으로 쓰게 torch tensor로 변환
        segment_tensor = torch.tensor(segment).float()  # 이미 [1, 1, 224, segment_length] 형태

        # audio embedding 추출
        with torch.no_grad():
            embedding = soundclip.encode_audio(segment_tensor)
            embeddings.append(embedding.cpu().numpy())

    return np.array(embeddings)

# 각각의 mel_spectrogram에 대해 처리
for npy_file in npy_files:
    mel_spectrogram = np.load(npy_file)
    mel_spectrogram = mel_spectrogram.reshape(1, 1, 224, 224)  # 원본 차원 조정
    embeddings = segment_and_encode(mel_spectrogram)
    print(f'File: {npy_file}, Embeddings shape: {embeddings.shape}')

