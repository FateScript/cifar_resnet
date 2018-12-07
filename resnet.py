import numpy as np
import tensorflow as tf
import layer
from tensorflow.examples.tutorials.mnist import input_data
import cifar
from IPython import embed
from config import *

image_width = 32
image_height = 32
image_depth = 3
num_classes = 10

#embed(header="dataset check")

def scalar_writer(writer, name, value, step):
    v = tf.Summary.Value(tag=name, simple_value=value)
    s = tf.Summary(value=[v])
    writer.add_summary(s, step)
    return


class ResnetModel(object):

    def __init__(self, train=False, save=False, load=False):
        self.index = 0
        self.input_placeholder, self.label_placeholder, self.is_training, self.lr = self.input_placeholder()
        #self.logits = self.buildResNet(self.input_placeholder, FLAGS.num_blocks, self.is_training)
        self.logits = self.infer(self.input_placeholder, FLAGS.num_blocks, self.is_training, load)
        self.loss, self.total_loss = self.loss(self.logits, self.label_placeholder)
        self.acc = self.accuracy_op(self.logits, self.label_placeholder)
        self.train_op = self.train_op(self.lr)
        #self.train_op = self.train_op()
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)
        self.writer = self.boardWrite()
        #self.load_model()
        if train:
            self.train()
        if save:
            self.saver = self.save_model()


    def buildResNet(self, inputs, n, is_training):
        filters = [16, 32, 64]
        inputs = tf.reshape(inputs, shape=(-1, image_width, image_height, image_depth))
        #Conv1_x
        x = layer.conv(inputs, [3, 3, 3, 16], strides=1, name="Conv1")
        #Conv2_x
        #x = layer.maxpool(x, win_size=3, strides=2, name="Conv2_Pool1")
        for i in range(n):
            x = layer.res_block(x, 16, is_training=is_training, name="Conv2_Res"+str(i) )
        #Conv3_x
        for i in range(n):
            x = layer.res_block(x, 32, is_training=is_training, name="Conv3_Res"+str(i) )
        #Conv4_x
        for i in range(n):
            x = layer.res_block(x, 64, is_training=is_training, name="Conv4_Res"+str(i) )
        #x = layer.avgpool(x, win_size=7, strides=7, name="Global_avgpool")
        x = layer.avgpool(x, win_size=8, strides=8, name="Global_avgpool")
        reshaped_x = tf.reshape(x, [-1, filters[2]])
        x = layer.fc(reshaped_x, output_dim=num_classes, name="FC")
        return x


    def input_placeholder(self):
        input_placeholder = tf.placeholder(shape=(None, image_width, image_height, image_depth), dtype=tf.float32, name="input_placeholder")
        label_placeholder = tf.placeholder(shape=(None, num_classes), dtype=tf.float32, name="label_placeholder")
        lr_placeholder = tf.placeholder(shape=(None), dtype=tf.float32, name="lr_placeholder")
        is_training = tf.placeholder(shape=(None), dtype=tf.bool, name="is_training")
        return input_placeholder, label_placeholder,  is_training, lr_placeholder


    def fill_feed_dict_with_batch(self, x_data, y_data, lr=0, augment=True, is_training=True, batch_size=128):
        if self.index + batch_size > x_data.shape[0]:
            end_index = x_data.shape[0]
        else:
            end_index = self.index + batch_size
        input_feed, labels_feed = x_data[self.index : end_index], y_data[self.index : end_index]
        #input_feed = self.sess.run(tf.cast(tf.reshape(input_feed, (-1,28,28,1)), tf.float32))
        if augment:
            input_feed = cifar.random_crop_and_flip(input_feed, padding_size=2)
            input_feed = cifar.whitening_image(input_feed)
        feed_dict = {
                self.input_placeholder : input_feed,
                self.label_placeholder : labels_feed,
                self.lr                : lr,
                self.is_training       : is_training
        }
        self.index = end_index % x_data.shape[0]
        return feed_dict

    def loss(self, logits, labels):
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        loss = tf.reduce_mean(losses)
        regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regu_loss) + loss
        return loss, total_loss

    def train_op(self, learning_rate):
        train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(self.total_loss)
        return train_op

    def accuracy_op(self, preds, labels, top_k=1):
        correct = tf.nn.in_top_k(preds, tf.arg_max(labels,1), top_k)
        return tf.reduce_sum(tf.cast(correct, tf.int32))


    def boardWrite(self):
        #tf.summary.scalar('loss', self.loss)
        #tf.summary.scalar('accuracy', self.acc)
        #self.merged = tf.summary.merge_all()
        return tf.summary.FileWriter("./graph/cifar_resnet", self.sess.graph)


    def save_model(self, epoch):
        saver = tf.train.Saver()
        path = "./model/epoch-" + str(epoch)
        save_path = saver.save(self.sess, path)
        return saver


    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


    def get_learningrate(self, epoch):
        decay_epoch = [60, 80]
        if epoch < decay_epoch[0]:
            lr = 0.1
        elif epoch < decay_epoch[1]:
            lr = 0.01
        else:
            lr = 0.001
        return lr


    def train(self, epoch=100, batch_size=128):
        global x_train, y_train
        for i in range(epoch):
            lr = self.get_learningrate(i)
            loss = 0.0
            acc = 0.0
            train_iter = num_data//batch_size if num_data%batch_size == 0 else num_data//batch_size+1
            for j in range(train_iter):
                fd = self.fill_feed_dict_with_batch(x_train, y_train, lr, batch_size=batch_size)
                _, temp_loss, temp_acc = self.sess.run([self.train_op, self.total_loss, self.acc], feed_dict=fd)
                loss += temp_loss
                acc += temp_acc
            avg_loss = loss/(num_data//batch_size)
            avg_acc = acc/(num_data)
            #self.writer.add_summary(avg_loss, i)
            scalar_writer(self.writer, "train_loss", avg_loss, i)
            scalar_writer(self.writer, "train_acc", avg_acc, i)
            print("epoch{}'s loss: {}".format(i, avg_loss))
            print("train_acc: {}".format(avg_acc))
            test_acc = 0.0
            test_loss = 0.0
            test_iter = num_test//batch_size if num_test%batch_size else num_test//batch_size+1
            for j in range(test_iter):
                fd = self.fill_feed_dict_with_batch(x_test, y_test, lr, augment=False, is_training=False, batch_size=batch_size)
                temp_loss, temp_acc = self.sess.run([self.total_loss, self.acc], feed_dict=fd)
                test_acc += temp_acc
                test_loss += temp_loss
            avg_test_acc = test_acc/(num_test)
            avg_test_loss = test_loss/(num_test//batch_size)
            scalar_writer(self.writer, "test_acc", avg_test_acc, i)
            scalar_writer(self.writer, "test_loss", avg_test_loss, i)
            print("test loss:{}".format(avg_test_loss))
            print("test_acc: {}\n".format(avg_test_acc))

            #shuffle data
            order = np.random.permutation(num_data)
            x_train = x_train[order, ...]
            y_train = y_train[order]

            if i%20 == 0 or i == epoch-1:
                self.save_model(i)


    def inference(self, data, labels, size=1):
        fd = self.fill_feed_dict_with_batch(data, labels, lr=0, augment=True, is_training=False, batch_size=size)
        prob = tf.nn.softmax(self.logits)
        probs = self.sess.run(prob, feed_dict=fd)
        return probs, fd


    def infer(self, inputs, n, is_training, reuse=False):
        with tf.variable_scope('stage0', reuse=reuse):
            x = layer.conv_bn_relu(inputs, [3, 3, 3, 16], 1, is_training, name='stage0')
        for i in range(n):
            with tf.variable_scope('stage1_res_%d' %i, reuse=reuse):
                if i == 0:
                    x = layer.res_block(x, 16, is_training, name='stage1_res%d' %i , first_block=True)
                else:
                    x = layer.res_block(x, 16, is_training, name='stage1_res%d' %i)
        
        for i in range(n):
            with tf.variable_scope('stage2_res_%d' %i, reuse=reuse):
                x = layer.res_block(x, 32, is_training, name='stage2_res%d' %i)
        for i in range(n):
            with tf.variable_scope('stage3_res_%d' %i, reuse=reuse):
                x = layer.res_block(x, 64, is_training, name='stage3_res%d' %i)

        with tf.variable_scope('fc', reuse=reuse):
            x = layer.batchNorm(x, is_training, 'fc_batchNorm')
            x = tf.nn.relu(x)
            feature = tf.reshape(layer.avgpool(x, 8, 8, name='global_avg_pool'), shape=(-1, 64))
            x = layer.fc(feature, 10, name='fc')
        return x


if __name__ == "__main__":
    x_train, y_train = cifar.prepare_train_data(padding_size=2)
    x_test, y_test = cifar.read_validation_data()
    num_data = len(y_train)
    num_test = len(y_test)
    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train),(x_test, y_test) = mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = ResnetModel()
    model.train(epoch=100, batch_size=128)
