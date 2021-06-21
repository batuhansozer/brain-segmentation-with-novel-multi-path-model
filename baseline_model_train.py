import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import argparse
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', required=True, type=bool)
parser.add_argument('--input_height', required=True, type=int)
parser.add_argument('--input_width', required=True, type=int)
parser.add_argument('--split_npy_dataset_path', required=True, type=str)
parser.add_argument('--epochs', required=True, type=int)

args = parser.parse_args()

if(args.use_gpu):
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

def plot_metrics(save_path, history):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss = history["loss"]
    val_loss = history["val_loss"]

    dice = history['dice_coef']
    val_dice = history['val_dice_coef']

    # loss
    plt.figure()
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(save_path + "/loss.png")

    # dice
    plt.figure()
    plt.plot(dice, label="Training Dice")
    plt.plot(val_dice, label="Validation Dice")
    plt.title("Dice")
    plt.legend()
    plt.savefig(save_path + "/dice.png")

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def organize_batch_dirs(batch_dirs):
    batch_list = []
    for i in batch_dirs:
        if i[0] == "X":
            temp = []

            temp.append(i)

            i = list(i)
            i[0] = "y"
            i = "".join(i)
            temp.append(i)

            batch_list.append(temp)
        else:
            break

    return batch_list

def data_gen(splited_npy_dataset_path):
    batch_dirs = os.listdir(splited_npy_dataset_path)
    batch_dirs.sort()
    batch_dirs = organize_batch_dirs(batch_dirs)
    while True:
        shuffle(batch_dirs)

        for batch_path in batch_dirs:
            X_batch = np.load(splited_npy_dataset_path + '/' + batch_path[0])
            Y_batch = np.load(splited_npy_dataset_path + '/' + batch_path[1])

            X_batch = X_batch.astype(np.float32)
            Y_batch = Y_batch.astype(np.int32)

            Y_batch = get_one_hot(np.squeeze(Y_batch, axis=-1), n_classes)

            flair_validation, t2_validation, t1_validation = X_batch[:, :, :, 0], X_batch[:, :, :, 1], X_batch[:, :, :, 2]

            flair_validation = np.expand_dims(flair_validation, axis=-1)
            t2_validation = np.expand_dims(t2_validation, axis=-1)
            t1_validation = np.expand_dims(t1_validation, axis=-1)

            yield flair_validation, Y_batch # for classic res u-net

def valid_data_gen(split_npy_dataset_path):
    split_npy_dataset_path = split_npy_dataset_path + "_validation"

    batch_dirs = os.listdir(split_npy_dataset_path)
    batch_dirs.sort()
    batch_dirs = organize_batch_dirs(batch_dirs)

    while True:
        shuffle(batch_dirs)
        for batch_path in batch_dirs:
            X_validation = np.load(split_npy_dataset_path + '/' + batch_path[0])
            y_validation = np.load(split_npy_dataset_path + '/' + batch_path[1])

            X_validation = X_validation.astype(np.float32)
            y_validation = y_validation.astype(np.int32)

            y_validation = get_one_hot(np.squeeze(y_validation, axis=-1), n_classes)

            flair_validation, t2_validation, t1_validation = X_validation[:, :, :, 0], X_validation[:, :, :, 1], X_validation[:, :, :, 2]

            flair_validation = np.expand_dims(flair_validation, axis=-1)
            t2_validation = np.expand_dims(t2_validation, axis=-1)
            t1_validation = np.expand_dims(t1_validation, axis=-1)

            yield flair_validation, y_validation # for classic res u-net

def dice_coef(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    smooth = 1.

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)

    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def mean_class_dice(y_true, y_pred, n_classes=16):
    dice_list = []

    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), n_classes)

    for i in range(n_classes):
        y_true_temp, y_pred_temp = y_true[:, :, :, i], y_pred[:, :, :, i]
        dice_list.append(dice_coef(y_true_temp, y_pred_temp).numpy())

    return sum(dice_list)/len(dice_list)

if __name__ == '__main__':
    input_shape = (args.input_width, args.input_height, 1)
    epochs = args.epochs
    n_classes = 16
    lr = 1e-4

    model = get_single_input_resnet_model(input_shape, n_classes)

    metrics = [dice_coef, mean_class_dice]

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr), metrics=metrics, run_eagerly=True)

    callbacks = [
        ModelCheckpoint("baseline_model.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    ]

    steps_per_epoch = (len(os.listdir(args.split_npy_dataset_path)) // 2)
    validation_steps = (len(os.listdir(args.split_npy_dataset_path + '_validation')) // 2)

    model.fit(data_gen(args.split_npy_dataset_path),
              epochs=epochs,
              steps_per_epoch = steps_per_epoch,
              validation_steps = validation_steps,
              validation_data = valid_data_gen(args.split_npy_dataset_path),
              callbacks=callbacks
              )

    history = model.history.history

    plot_metrics('baseline_model_graphs', history)
    