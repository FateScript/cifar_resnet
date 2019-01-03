from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import math
import copy
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import scipy
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

from load_animals import load_animals

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from sklearn.metrics.pairwise import rbf_kernel

from influence.inceptionModel import BinaryInceptionModel
from influence.smooth_hinge import SmoothHinge
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.dataset_poisoning import generate_inception_features
from influence.multiclass_nn import MulticlassNN
from IPython import embed

def get_Y_pred_correct_inception(model):
    Y_test = model.data_sets.test.labels
    if np.min(Y_test) < -0.5:
        Y_test = (np.copy(Y_test) + 1) / 2        
    Y_pred = model.sess.run(model.preds, feed_dict=model.all_test_feed_dict)
    Y_pred_correct = np.zeros([len(Y_test)])
    for idx, label in enumerate(Y_test):
        Y_pred_correct[idx] = Y_pred[idx, int(label)]
    return Y_pred_correct

num_classes = 10

from tqdm import tqdm

tf.reset_default_graph()

train = np.load("cifar_train_features.npz")
x_train = train['features']
y_train = train['labels']

test = np.load("cifar_test_features.npz")
x_test = test['features']
y_test = test['labels']

train = DataSet(x_train, y_train)
test = DataSet(x_test, y_test)

validation = None
data_sets = base.Datasets(train=train, validation=validation, test=test)

dataset_name = 'cifar'
input_dim = 64
batch_size = 100
weight_decay = 0.0001
initial_learning_rate = 0.1
decay_epochs = [100, 1000]

multiNN_model = MulticlassNN(
    input_dim=input_dim,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    model_name='%s_MultiClassNN' % dataset_name)

#embed(header='before training')
#epoch = 300
#multiNN_model.train(epoch=1500, batch_size=100)

test_idx_list = np.arange(0,10000)
#predicted_loss_diffs = np.zeros(2700)
for test_idx in tqdm(test_idx_list):
    predicted_loss_diffs = multiNN_model.get_influence_on_test_loss(
        [test_idx], 
        np.arange(len(multiNN_model.data_sets.train.labels)),
        force_refresh=True)
    label = y_test[test_idx]
    np.savez(
        'cifar_data/index{}_label{}.npz'.format(test_idx, label),
        test_idx = test_idx,
        label = label,
        predicted_loss_diffs = predicted_loss_diffs
    )

'''
x_test = X_test[test_idx, :]
y_test = Y_test[test_idx]


#distances = dataset.find_distances(x_test, X_train)
#flipped_idx = Y_train != y_test
#rbf_margins_test = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_test_feed_dict)
#rbf_margins_train = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_train_feed_dict)
inception_Y_pred_correct = get_Y_pred_correct_inception(inception_model)


np.savez(
    'output/rbf_results', 
    test_idx=1,
    #distances=distances,
    #flipped_idx=flipped_idx,
    #rbf_margins_test=rbf_margins_test,
    #rbf_margins_train=rbf_margins_train,
    #inception_Y_pred_correct=inception_Y_pred_correct,
    #rbf_predicted_loss_diffs=rbf_predicted_loss_diffs,
    multinn_predicted_loss_diffs = predicted_loss_diffs
)
'''
