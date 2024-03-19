import numpy as np
import matplotlib.pyplot as plt
import librosa
from glob import glob
import os 

mel_dir = '/home/broiron/Desktop/AudioCLIP/data/UnAV/data/unav100/mel'
mel_files = glob(os.path.join(mel_dir, '*.npy'))
mel_db = np.load(mel_files[43]) 

n_mels = 128 # 실제 mel 계산 시 사용한 값
time_frames = mel_db.shape[0] // n_mels
mel_db_reshaped = mel_db.reshape(n_mels, time_frames)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db_reshaped, sr=44100, fmax=8000, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.tight_layout()
plt.show()