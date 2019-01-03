import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from IPython import embed
from tqdm import tqdm
import cifar
from resnet_modify import ResnetModel

x_train, y_train = cifar.prepare_train_data(padding_size=0, shuffle=False)
x_test, y_test = cifar.read_validation_data(shuffle=False)
num_data = len(y_train)
num_test = len(y_test)
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = ResnetModel(load=True)
test_idx_list = np.arange(10000//4*3,10000)
#predicted_loss_diffs = np.zeros(2700)
for test_idx in tqdm(test_idx_list):
    predicted_loss_diffs = model.get_influence_on_test_loss(
        [test_idx], np.arange(num_data), x_train, y_train, x_test, y_test
    )
    label = y_test[test_idx]
    np.savez(
        'approx_relu_mislabel/index{}.npz'.format(test_idx),
        test_idx = test_idx,
        label = label,
        predicted_loss_diffs = predicted_loss_diffs
    )


