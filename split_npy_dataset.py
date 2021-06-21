import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', required=True, type=str)
parser.add_argument('--split_dataset_save_path', required=True, type=str)
parser.add_argument('--batch_size', required=True, type=int)

args = parser.parse_args()

def split_npy_dataset(npy_dataset_path, split_npy_dataset_path, batch_size, test_size=0.2):
    X = np.load(npy_dataset_path+'/X.npy')
    y = np.load(npy_dataset_path+'/y.npy')
    
    X, X_test, y, y_test = train_test_split(X, y, test_size = test_size)
    
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = test_size)
    
    print(y.shape, y_train.shape, y_validation.shape)
    
    if not os.path.exists(split_npy_dataset_path):
        os.makedirs(split_npy_dataset_path)
    
    if not os.path.exists(split_npy_dataset_path + '_validation'):
        os.makedirs(split_npy_dataset_path + '_validation')

    for batch_i in range(0, y_train.shape[0], batch_size):
        np.save(split_npy_dataset_path+'/X_{0}.npy'.format(batch_i), X_train[batch_i:batch_i+batch_size])
        np.save(split_npy_dataset_path+'/y_{0}.npy'.format(batch_i), y_train[batch_i:batch_i+batch_size])
    
    for batch_i in range(0, y_validation.shape[0], batch_size):
        np.save(split_npy_dataset_path+'_validation/X_{0}.npy'.format(batch_i), X_validation[batch_i:batch_i+batch_size])
        np.save(split_npy_dataset_path+'_validation/y_{0}.npy'.format(batch_i), y_validation[batch_i:batch_i+batch_size])

    
    print('Splitted NPY Dataset saved!')

split_npy_dataset(args.dataset_path, args.split_dataset_save_path, batch_size = args.batch_size)
