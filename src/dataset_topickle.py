from datasets import load_dataset
import pickle

# ONLY USED FOR TRAINING
# Dumps dataset as pickle file for easier handling during training.

dataset = load_dataset('imagefolder', data_dir='dataset/images')

with open('dataset/data.pkl', 'wb') as picklefile:
    pickle.dump(dataset, picklefile)