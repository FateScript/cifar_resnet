import urllib
import tarfile
import numpy as np
import pickle
import sys
import os
import cv2
from IPython import embed

data_dir = 'cifar10_data'
full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
validation_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

image_size = 32
image_depth = 3
num_classes = 10

train_random_label = False
validation_random_label = False
num_train_batch = 5 # from 0 to 5


def download_and_unzip():
    dest_dir = data_dir
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, blocksize, total_size):
            sys.stdout.write('\r>> Downloading %s %1.f%%' % (filename, float(count*blocksize)/float(total_size) *100.0 ))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)


def _read_one_batch(path, is_random_label):
    fo = open(path, 'rb')
    dicts = pickle.load(fo, encoding='latin1')
    fo.close()

    data = dicts['data']
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label


def whitening_image(image_np):
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(image_size*image_size*image_depth)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    cropped_batch = np.zeros(len(batch_data) * image_size * image_size * image_depth).reshape(
            (-1, image_size, image_size, image_depth))
    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2*padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2*padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+image_size, y_offset:y_offset+image_size, :]
        cropped_batch[i, ...] = horizontal_flip(cropped_batch[i, ...], axis=1)
    return cropped_batch


def read_in_all_images(address_list, shuffle=True, is_random_label=False):
    data = np.array([]).reshape([0, image_size*image_size*image_depth])
    label = np.array([])

    for address in address_list:
        print("Reading images from "+address)
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))
    num_data = len(label)
    data = data.reshape((num_data, image_size*image_size, image_depth), order='F')
    data = data.reshape((num_data, image_size, image_size, image_depth))

    if shuffle is True:
        print("Shuffling")
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]
    data = data.astype(np.float32)
    return data, label


def horizontal_flip(image, axis):
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)
    return image


def prepare_train_data(padding_size, shuffle=True):
    path_list = []
    for i in range(1, num_train_batch+1):
        path_list.append(full_data_dir+str(i))
    data, label = read_in_all_images(path_list, shuffle, is_random_label=train_random_label)
    pad_width = ( (0,0), (padding_size, padding_size), (padding_size, padding_size), (0,0) )
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    return data, label


def prepare_drop_train_data(padding_size, drop_idx, shuffle=True):
    path_list = []
    for i in range(1, num_train_batch+1):
        path_list.append(full_data_dir+str(i))
    data, label = read_in_all_images(path_list, shuffle, is_random_label=train_random_label)
    pad_width = ( (0,0), (padding_size, padding_size), (padding_size, padding_size), (0,0) )
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    keep_idx = np.setdiff1d(np.arange(len(label)), drop_idx)
    return data[keep_idx], label[keep_idx]


def prepare_mislabel_train_data(padding_size, num_mislabel, shuffle=False):
    path_list = []
    idx = np.random.choice(np.arange(0, 50000), num_mislabel)
    mislabel_idx = np.array(list(set(idx)))
    mislabel_idx.sort()
    for i in range(1, num_train_batch+1):
        path_list.append(full_data_dir+str(i))
    data, label = read_in_all_images(path_list, shuffle, is_random_label=train_random_label)
    pad_width = ( (0,0), (padding_size, padding_size), (padding_size, padding_size), (0,0) )
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    add = np.random.randint(1, 10, len(mislabel_idx))
    label[mislabel_idx] = (label[mislabel_idx] + add) % 10
    return data, label, mislabel_idx


def read_validation_data(shuffle=True):
    validation_array, validation_labels = read_in_all_images([validation_dir], shuffle, 
            is_random_label=validation_random_label)
    validation_array = whitening_image(validation_array) 
    return validation_array, validation_labels

'''
def read_validation_data(shuffle=True):
    validation_array, validation_labels = read_in_all_images([validation_dir], shuffle, 
            is_random_label=validation_random_label)
    validation_array = whitening_image(validation_array) 
    return validation_array, validation_labels

'''
if __name__ == "__main__":
    data, labels, idx = prepare_mislabel_train_data(0, num_mislabel=504)
    #d,l = prepare_train_data(0, False)
    embed()
