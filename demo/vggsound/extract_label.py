import os
from tqdm import tqdm

seg_dir = '/home/broiron/Desktop/AudioCLIP/data/test_2'
label_dir = '/home/broiron/Desktop/AudioCLIP/data/label'

# Ensure the label directory exists
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

# file 순서대로 label 추출
sorted_filenames = sorted(os.listdir(seg_dir))
labels_file_path = os.path.join(label_dir, f'labels_{len(sorted_filenames)}.txt')


with open(labels_file_path, 'w') as labels_file:
    for filename in tqdm(sorted_filenames, desc='Extracting labels'):
        label = filename.split('_')[0]
        labels_file.write(label + '\n')