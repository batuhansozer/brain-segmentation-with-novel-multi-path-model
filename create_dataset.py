import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import nibabel as nib
import pylibjpeg
from tqdm import tqdm
from albumentations import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_height', required=True, type=int)
parser.add_argument('--input_width', required=True, type=int)
parser.add_argument('--dataset_path', required=True, type=str)
parser.add_argument('--dataset_save_path', required=True, type=str)

args = parser.parse_args()

INPUT_SHAPE = (args.input_width, args.input_height, 1) # (width, height, depth)
dataset_path = args.dataset_path
dataset_save_path = args.dataset_save_path

def get_scan(dicom_path, scan_size):
    voxel_ndarray = nib.load(dicom_path)

    voxel_ndarray = np.array(voxel_ndarray.dataobj)
    
    voxel_ndarray = voxel_ndarray.astype(float)
    voxel_ndarray = (np.maximum(voxel_ndarray, 0) / voxel_ndarray.max()) * 255.0
    voxel_ndarray = np.uint16(voxel_ndarray)

    voxel_ndarray = cv2.resize(voxel_ndarray, scan_size, interpolation=cv2.INTER_NEAREST)

    return voxel_ndarray

def get_seg_img(data_path, img_size):
    seg_img = []

    imgs = nib.load(data_path)
        
    imgs = np.array(imgs.dataobj)
    
    seg_img = cv2.resize(imgs, img_size, interpolation=cv2.INTER_NEAREST)
    
    seg_img[seg_img == 15] = 13
    seg_img[seg_img == 21] = 14
    seg_img[seg_img == 22] = 15

    return seg_img

def scan_pading(flair_scan, t2_scan, t1_scan, seg_img, input_shape):
    # this function fills the gaps with zero valued arrays. (it changes the depth)
    pad_size = input_shape - (flair_scan.shape[-1] % input_shape)
    if pad_size != input_shape:
        padded_flair = np.pad(flair_scan, ((0,0), (0,0), (0,pad_size)), 'constant')
        padded_t2 = np.pad(t2_scan, ((0,0), (0,0), (0,pad_size)), 'constant')
        padded_t1 = np.pad(t1_scan, ((0,0), (0,0), (0,pad_size)), 'constant')
        try:
            padded_seg_img = np.pad(seg_img, ((0,0), (0,0), (0, pad_size)), 'constant')
        except:
            padded_seg_img = None
    else:
        padded_flair = flair_scan
        padded_t2 = t2_scan
        padded_t1 = t1_scan
        padded_seg_img = seg_img

    return padded_flair, padded_t2, padded_t1, padded_seg_img

def split_scan_imgs(flair_scan, t2_scan, t1_scan, seg_imgs, input_shape):
    # since the depth is multiplier of expected depth we can split the image into expected_d / d
    split_flair = []
    for i in range(0, flair_scan.shape[-1]-(input_shape-1)):
        split_flair.append(flair_scan[:, :, i:i+input_shape])

    split_t2 = []
    for i in range(0, t2_scan.shape[-1]-(input_shape-1)):
        split_t2.append(t2_scan[:, :, i:i+input_shape])

    split_t1 = []
    for i in range(0, t1_scan.shape[-1]-(input_shape-1)):
        split_t1.append(t1_scan[:, :, i:i+input_shape])
    
    split_seg_imgs = []
    for i in range(0, seg_imgs.shape[-1]-(input_shape-1)):
        split_seg_imgs.append(seg_imgs[:, :, i:i+input_shape])

    split_flair = np.array(split_flair)
    split_t2 = np.array(split_t2)
    split_t1 = np.array(split_t1)
    split_seg_imgs = np.array(split_seg_imgs)
    
    return split_flair, split_t2, split_t1, split_seg_imgs

def do_augmentation(scans, seg_imgs):
    final_scans, final_seg_imgs = [], []

    for i in tqdm(range(scans.shape[0]), total=scans.shape[0]):
        x = scans[i, :, :, :]
        y = seg_imgs[i, :, :, :]
        
        aug = ElasticTransform(p=1.0)
        augmented = aug(image=x, mask=y)
        x1 = augmented["image"]
        y1 = augmented["mask"]
        
        aug = RandomRotate90(p=1.0)
        augmented = aug(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']

        aug = GridDistortion(p=1.0)
        augmented = aug(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']

        aug = HorizontalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']

        aug = VerticalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']

        save_images = [x, x1, x2, x3, x4, x5]
        save_masks =  [y, y1, y2, y3, y4, y5]
        
        for one_scan in save_images:
            final_scans.append(one_scan)

        for one_seg_img in save_masks:
            final_seg_imgs.append(one_seg_img)

    return np.array(final_scans), np.array(final_seg_imgs)

def get_dataset(dataset_path, input_shape, dataset_save_path, test_size=0.2, save=True, do_augment=True):
    count = 1
    passed_folders = []

    flair_scans, t2_scans, t1_scans, seg_imgs = [], [], [], []
    samples = os.listdir(dataset_path + '/sourcedata')

    for sample_id in tqdm(samples, total=len(samples)):
        count += 1

        flair_path = dataset_path + '/sourcedata/' + sample_id + '/anat/' + sample_id + '_FLAIR.nii.gz'
        t2_path = dataset_path + '/sourcedata/' + sample_id + '/anat/' + sample_id + '_T2w.nii.gz'
        t1_path = dataset_path + '/sourcedata/' + sample_id + '/anat/' + sample_id + '_T1w.nii.gz'
        mask_path = dataset_path + '/derivatives/segmentation/' + sample_id + '/anat/' + sample_id + '_dseg.nii.gz'

        flair_scan = get_scan(flair_path, scan_size=input_shape[0:2])
        t1_scan = get_scan(t1_path, scan_size=input_shape[0:2])
        t2_scan = get_scan(t2_path, scan_size=input_shape[0:2])
        seg_img = get_seg_img(mask_path, img_size=input_shape[0:2])
        
        if type(flair_scan) == type(False) or type(t2_scan) == type(False):
            passed_folders.append(sample_id)
            continue
        
        if flair_scan.shape[-1] != t2_scan.shape[-1] or flair_scan.shape[-1] != t1_scan.shape[-1] or t1_scan.shape[-1] != t2_scan.shape[-1]:
            passed_folders.append(sample_id)
            continue
        
        flair_scan, t2_scan, t1_scan, seg_img = scan_pading(flair_scan, t2_scan, t1_scan, seg_img, input_shape = input_shape[2]) # 128
        flair_scan, t2_scan, t1_scan, seg_img = split_scan_imgs(flair_scan, t2_scan, t1_scan, seg_img, input_shape = input_shape[2]) # 128

        for one_scan in flair_scan:
            flair_scans.append(one_scan)
        
        for one_scan in t2_scan:
            t2_scans.append(one_scan)    

        for one_scan in t1_scan:
            t1_scans.append(one_scan)
        

        for one_seg_img in seg_img:
            seg_imgs.append(one_seg_img)

    flair_scans = np.array(flair_scans, dtype='float32')
    t2_scans = np.array(t2_scans, dtype='float32')
    t1_scans = np.array(t1_scans, dtype='float32')
    seg_imgs = np.array(seg_imgs).astype('float32')
    
    flair_scans = (flair_scans - np.min(flair_scans)) / (np.max(flair_scans) - np.min(flair_scans))
    t2_scans = (t2_scans - np.min(t2_scans)) / (np.max(t2_scans) - np.min(t2_scans))
    t1_scans = (t1_scans - np.min(t1_scans)) / (np.max(t1_scans) - np.min(t1_scans))

    print(flair_scans.shape, t2_scans.shape, t1_scans.shape, seg_imgs.shape)

    print(np.min(flair_scans), np.max(flair_scans))
    print(np.min(t2_scans), np.max(t2_scans))
    print(np.min(t1_scans), np.max(t1_scans))
    print(np.min(seg_imgs), np.max(seg_imgs))
    
    scans = np.concatenate((flair_scans, t2_scans, t1_scans), axis=-1)
    
    scans, X_test, seg_imgs, y_test = train_test_split(scans, seg_imgs, test_size=0.2, random_state=6)
    
    if do_augment:
        scans, seg_imgs = do_augmentation(scans, seg_imgs)
    
    print("Scans Data Shape: ", str(scans.shape))
    print("Segmentation Data Shape: ", str(seg_imgs.shape))
    print("Passed Folder List: ", str(len(passed_folders)), passed_folders)
    print("Train labels: ", np.unique(seg_imgs))
    print("Test labels: ", np.unique(y_test))
    
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
    
    
    np.save(dataset_save_path + '/X_test.npy', X_test)
    np.save(dataset_save_path + '/y_test.npy', y_test)

    np.save(dataset_save_path + '/X.npy', scans)
    np.save(dataset_save_path + '/y.npy', seg_imgs)

    print('NPY Dataset Saved')

if __name__ == "__main__":
    # dataset_path, input_shape, test_size=0.2, dataset_save_path, save=True
    get_dataset(dataset_path=dataset_path, input_shape=INPUT_SHAPE, dataset_save_path=dataset_save_path)
