from datasets import load_dataset
import pickle

dataset = load_dataset('imagefolder', data_dir='dataset/images')

with open('dataset/data.pkl', 'wb') as picklefile:
    pickle.dump(dataset, picklefile)