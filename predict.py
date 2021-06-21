import os
import numpy as np
from numpy.lib import unique
import tensorflow as tf
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import pydicom
import dicom_numpy
import nibabel as nib
import pylibjpeg
import PIL
from keras import backend as K
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', required=True, type=bool)
parser.add_argument("--baseline_model", required=True, type=str)
parser.add_argument('--multi_channel_model', required=True, type=str)
parser.add_argument('--multi_path_model', required=True, type=str)
parser.add_argument('--dataset_path', required=True, type=str)

args = parser.parse_args()

n_classes = 16
lr = 1e-4

class_colors = [(0,0,0),(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,239,213),
                (0,0,205),(205,133,63),(210,180,140),(102,205,170),(0,0,128),(7,36,0),(218,219,112),(218,112,214)]

if args.use_gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    seg_arr = seg_arr[:, :, 0]

    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    # seg_img = np.zeros((output_height, output_width, output_depth, 3))
    seg_img = np.zeros((output_height, output_width, 3)) # (128, 128, 32, 3)

    for c in range(n_classes):

        seg_img[:, :, 0] += ((seg_arr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = seg_img.astype(np.int32)

    return seg_img

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def class_dice(y_true, y_pred, n_classes=16):
    dice_list = []

    y_pred = get_one_hot(np.argmax(y_pred, axis=-1), n_classes)

    # print(y_true.shape, y_pred.shape)

    for i in range(n_classes):
        y_true_temp, y_pred_temp = y_true[:, :, :, i], y_pred[:, :, :, i]
        dice_list.append(dice_coef(y_true_temp, y_pred_temp).numpy())

    return sum(dice_list)/len(dice_list)

if __name__ == '__main__':
    X_test = np.load(os.path.join(args.dataset_path, 'X_test.npy'))
    y_test = np.load(os.path.join(args.dataset_path, 'y_test.npy'))

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)

    print(X_test.shape, y_test.shape, np.max(X_test), np.max(y_test))

    multi_encoder_model = tf.keras.models.load_model(args.multi_path_model, custom_objects={"dice_coef":dice_coef, "class_dice":class_dice})
    multi_channel_model = tf.keras.models.load_model(args.multi_channel_model, custom_objects={"dice_coef":dice_coef, "class_dice": class_dice})
    classic_resnet_model = tf.keras.models.load_model(args.baseline_model, custom_objects={"dice_coef":dice_coef, "class_dice": class_dice})

    print("X_test shape, y_test shape: ", X_test.shape, y_test.shape, np.max(X_test), np.max(y_test))
    print("Classes: ", np.unique(y_test).astype(np.int32))
    
    count = 0
    for img, mask in tqdm(zip(X_test, y_test), total=len(X_test)):
        name = "results/results_" + str(count) + ".png"
        count +=1

        flair = img[:, :, 0]
        t2 = img[:, :, 1]
        t1 = img[:, :, 2]

        flair = np.expand_dims(flair, axis=-1)
        t2 = np.expand_dims(t2, axis=-1)
        t1 = np.expand_dims(t1, axis=-1)

        flair = np.expand_dims(flair, axis=0)
        t2 = np.expand_dims(t2, axis=0)
        t1 = np.expand_dims(t1, axis=0)

        img = np.expand_dims(img, axis=0)

        # print(flair.shape, t2.shape, t1.shape, img.shape)

        multi_encoder_pred = multi_encoder_model.predict([flair, t2, t1], verbose=0)
        multi_channel_pred = multi_channel_model.predict(img, verbose=0)
        classic_resnet_pred = classic_resnet_model.predict(flair, verbose=0)

        multi_encoder_pred = np.expand_dims(np.argmax(multi_encoder_pred, axis=-1), axis=-1)
        multi_channel_pred = np.expand_dims(np.argmax(multi_channel_pred, axis=-1), axis=-1)
        classic_resnet_pred = np.expand_dims(np.argmax(classic_resnet_pred, axis=-1), axis=-1)

        multi_encoder_pred = np.squeeze(multi_encoder_pred, axis=0)
        multi_channel_pred = np.squeeze(multi_channel_pred, axis=0)
        classic_resnet_pred = np.squeeze(classic_resnet_pred, axis=0)

        mask_colored = get_colored_segmentation_image(mask, n_classes)
        multi_channel_pred_colored = get_colored_segmentation_image(multi_channel_pred, n_classes)
        multi_encoder_pred_colored = get_colored_segmentation_image(multi_encoder_pred, n_classes)
        classic_resnet_pred_colored = get_colored_segmentation_image(classic_resnet_pred, n_classes)

        fig, axs = plt.subplots(2, 4)
        
        axs[0, 0].imshow(img[0, :, :, 0], cmap='gray')
        axs[0, 0].set_title('Flair')

        axs[0, 1].imshow(img[0, :, :, 1], cmap='gray')
        axs[0, 1].set_title('T2')

        axs[0, 2].imshow(img[0, :, :, 2], cmap='gray')
        axs[0, 2].set_title('T1')

        axs[0, 3].imshow(mask_colored)
        axs[0, 3].set_title('GT')

        axs[1, 0].imshow(classic_resnet_pred_colored)
        axs[1, 0].set_title('Baseline Model')

        axs[1, 1].imshow(multi_channel_pred_colored)
        axs[1, 1].set_title('Multi-Channel Model')

        axs[1, 2].imshow(multi_encoder_pred_colored)
        axs[1, 2].set_title("Proposed Model")
        
        empty_white = np.empty_like(multi_encoder_pred_colored)
        empty_white.fill(255)
        axs[1, 3].imshow(empty_white)
        
        plt.tight_layout()
        
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
        plt.box(False)

        plt.savefig(name)
        plt.close()
