import os
import subprocess

for image_path in os.listdir('dataset/images'):
    image_id = image_path.split('.')[0]
    try:
        if int(image_id) < 12500:
            subprocess.run(['mv', 'dataset/images/' + image_path, 'dataset/images/Cat/' + image_path])
        else:
            subprocess.run(['mv', 'dataset/images/' + image_path, 'dataset/images/Dog/' + image_path])
    except ValueError:
        print('Invalid ID')