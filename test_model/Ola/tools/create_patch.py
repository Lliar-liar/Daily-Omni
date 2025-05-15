import glob
import os, io
import json
import random
from PIL import Image
from tqdm import tqdm
import argparse

json_out_dir = '/path/to/json'
image_in_dir = '/path/to/image_in_dir'
image_out_dir = '/path/to/image_out_dir'

file_list = []
for root, dirs, files in os.walk(image_in_dir):
    for file in files:
        file_list.append(os.path.join(root, file))
file_list = sorted(file_list)

file_count = 0
now_file_name = image_out_dir + f'patch_{file_count:06d}'
file = open(now_file_name, 'wb')
now_btytes = file.tell()
count_image = 0
new_data = {}

for image_path in tqdm(file_list):
    image_info = {}
    btyes = 0
    sizes = []
    with open(image_path, 'rb') as img:
        img_data = img.read()
        btyes = file.write(img_data)
        sizes.append(btyes)
    patch_info = {
        "patch": now_file_name,
        "start_num": now_btytes, "size": sizes,
    }
    count_image += 1
    now_btytes = file.tell()
    if count_image == 10000:
        file.close()
        file_count += 1
        now_file_name = image_out_dir + f'patch_{file_count:06d}'
        file = open(now_file_name, 'wb')
        now_btytes = 0
        count_image = 0

    image_info['original_image_path'] = os.path.join(image_path)
    image_info['image'] = patch_info
    new_data[image_path.split('/')[-1]] = image_info
    if count_image == 1:
        print(new_data)
file.close()

with open(f'{json_out_dir}/patch_mapping.json', 'w') as f:
    json.dump(new_data, f, indent=4)


