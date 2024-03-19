# audio를 mel-spectrogram으로 변환
import librosa
import numpy as np
import random
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import torch
import cv2 
from model.audioclip import AudioCLIP 
from utils.transforms import ToTensor1D

audio_dir = '/home/broiron/Desktop/AudioCLIP/data/UnAV/data/unav100/raw_audios'
mel_dir = '/home/broiron/Desktop/AudioCLIP/data/UnAV/data/unav100/mel'
os.makedirs(mel_dir, exist_ok=True)

soundclip = AudioCLIP(pretrained='../assets/AudioCLIP-Full-Training.pt')


'''
for audio_path in audio_files:
    y, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mel_spectrogram_db_resized = cv2.resize(mel_spectrogram_db, (224, 224))


    base_name = os.path.basename(audio_path)
    file_name = os.path.splitext(base_name)[0]
    mel_output_path = os.path.join(mel_dir, file_name + '_resized.npy')

    np.save(mel_output_path, mel_spectrogram_db_resized)

mel_files = glob(os.path.join(mel_dir, '*_resized.npy'))
mel_db_resized = np.load(mel_files[44]) 
mel_tensor = torch.from_numpy(mel_db_resized).float().unsqueeze(0).unsqueeze(0) 

print(f'Tensor shape: {mel_tensor.shape}')  # [1, 1, 224, 224]
'''


audio_transforms = ToTensor1D()
SAMPLE_RATE = 44100
n_mels = 128
time_length = 864
audio_files = glob(audio_dir + '/*.wav')

for audio_path in tqdm(audio_files, desc='Processing audio files'):
    track, _ = librosa.load(audio_path, sr=SAMPLE_RATE, dtype=np.float32)

    spec = soundclip.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.detach().numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
    audio_inputs = audio_transforms(track)
    local_audio_inputs = audio_inputs
    
    # localization audio input preprocessing 
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    local_audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    local_audio_inputs = librosa.power_to_db(local_audio_inputs, ref=np.max) / 80.0 + 1
    local_audio_inputs = local_audio_inputs 

    zero = np.zeros((n_mels, time_length))
    resize_resolution = 512
    h, w = local_audio_inputs.shape
    if w >= time_length:
        j = 0
        j = random.randint(0, w-time_length)
        audio_inputs = local_audio_inputs[:,j:j+time_length]
    else:
        zero[:,:w] = local_audio_inputs[:,:w]
        audio_inputs = zero

    local_audio_inputs = cv2.resize(local_audio_inputs, (224, 224))
    local_audio_inputs = np.array([local_audio_inputs])
    local_audio_inputs = torch.from_numpy(local_audio_inputs.reshape((1, 1, 224, 224))).float().cuda()
    print(local_audio_inputs.shape) # tensor shape : 1,1,224,224
    
    base_name = os.path.basename(audio_path)
    file_name = os.path.splitext(base_name)[0] + '.npy' 
    mel_output_path = os.path.join(mel_dir, file_name)
    np.save(mel_output_path, local_audio_inputs.cpu().numpy())

print('Done')


