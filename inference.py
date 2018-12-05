import tensorflow as tf
from IPython import embed
from resnet import ResnetModel
import cifar
import numpy as np

num_classes = 10
x_train, y_train = cifar.prepare_train_data(padding_size=0, shuffle=False)
x_test, y_test = cifar.read_validation_data(shuffle=False)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
batch_size = 200


model = ResnetModel()
saver = tf.train.Saver()
saver.restore(model.sess, "./model/epoch-99")
graph = tf.get_default_graph()

# get resnet features from trained model
def generateFeatures(model, graph, x_train, y_train, batch_size):
    features = graph.get_tensor_by_name("fc/Reshape:0")
    num_data = len(y_train)
    for i in range(num_data//batch_size):
        fd = model.fill_feed_dict_with_batch(x_train, y_train, lr=0, augment=False, is_training=False, batch_size=batch_size)
        train_features = model.sess.run(features, feed_dict=fd)
        label = np.argmax(fd[model.label_placeholder], axis=1)
        if i == 0:
            res_features = train_features
            res_labels = label
        else:
            res_features = np.concatenate((res_features, train_features), axis=0)
            res_labels = np.concatenate((res_labels, label), axis=0)
    return res_features, res_labels


train_features, train_labels = generateFeatures(model, graph, x_train, y_train, batch_size)
np.savez("cifar_train_features.npz",
        features=train_features,
        labels=train_labels)

test_features, test_labels = generateFeatures(model, graph, x_test, y_test, batch_size)
np.savez("cifar_test_features.npz",
        features=test_features,
        labels=test_labels)
print("Done")

'''
w = graph.get_tensor_by_name("fc/fc_weights:0")
temp_w = model.sess.run(w)

b = graph.get_tensor_by_name("fc/fc_bias:0")
temp_b = model.sess.run(b)

np.savez("fc_weights.npz", value=temp_w)
np.savez("fc_bias.npz", value=temp_b)
'''

