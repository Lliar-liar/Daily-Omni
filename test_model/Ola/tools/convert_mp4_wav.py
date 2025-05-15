import glob
import os
import argparse

parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--idx', type=int, help='index of the patch', default=0)
args = parser.parse_args()
idx = args.idx

path = '/path/to/video_folder'

filelist = []
for home, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith('.mp4') or filename.endswith('.mkv'):
            filelist.append(os.path.join(home, filename))

print(len(filelist))

from tqdm import tqdm
for file in tqdm(filelist):

    if file.endswith('.mp4'):
        target_path = file.replace('.mp4', '.wav')
    elif file.endswith('.mkv'):
        target_path = file.replace('.mkv', '.wav')
    else:
        raise NotImplementedError
    
    target_path = target_path.replace('/path/to/video_folder', '/path/to/audio_folder')

    if os.path.exists(target_path):
        continue

    # create target dir
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # convert
    os.system(f'ffmpeg -i {file} -ac 1 -ar 16000 -vn {target_path}')