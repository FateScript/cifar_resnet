import numpy as np
import tensorflow as tf
import layer
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
import cifar
from hessians import hessian_vector_product
from IPython import embed
from config import *
from tqdm import tqdm

image_width = 32
image_height = 32
image_depth = 3
num_classes = 10

drop_data_dir = "/unsullied/sharefs/wangfeng02/logs/cifar_resnet/div_by_label/drop_0_5000.npz"
boardWrite_dir = "./graph/all_net/mislabel/baseline_approx"
#boardWrite_dir = "/unsullied/sharefs/wangfeng02/logs/cifar_resnet/graph/all_net/mislabel/baseline_approx"
load_model_dir = "./model/all_net/mislabel/baseline_approx"
#load_model_dir = "/unsullied/sharefs/wangfeng02/logs/cifar_resnet/model/all_net/mislabel/baseline_approx"
params_dir = "/unsullied/sharefs/wangfeng02/logs/savez"

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
        self.logits = self.infer(self.input_placeholder, FLAGS.num_blocks, self.is_training)
        self.loss, self.total_loss = self.loss(self.logits, self.label_placeholder)
        self.acc = self.accuracy_op(self.logits, self.label_placeholder)
        self.train_op = self.train_op(self.lr)
        #self.train_op = self.train_op()
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)
        self.writer = self.boardWrite()
        #self.load_model()
       
        self.train_dir = 'output'
        self.model_name = 'resnet'
        self.params = self.get_all_params()
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_loss_no_reg_op = tf.gradients(self.loss, self.params)
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)
        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)
        self.vec_to_list = self.get_vec_to_list_fn()
        if train:
            self.train()
        if save:
            self.saver = self.save_model()
        if load:
            self.load_model(load_model_dir + "99")
        print("create train_grad_loss_list")
        self.train_grad_loss_list = []
        for i in tqdm(range(50000)):
            temp = np.load(params_dir + "/train_{}.npz".format(i))['loss_val']
            self.train_grad_loss_list.append(temp)



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


    def fill_feed_dict_with_batch(self, x_data, y_data, lr=0, augment=True, is_training=True, batch_size=128, update_index=True):
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
        if update_index:
            self.index = end_index % x_data.shape[0]
        return feed_dict
   

    def fill_feed_dict_with_some_ex(self, x_data, y_data, test_indices, lr=0, augment=True, is_training=False):
        input_feed, labels_feed = x_data[test_indices], y_data[test_indices]
        #input_feed = self.sess.run(tf.cast(tf.reshape(input_feed, (-1,28,28,1)), tf.float32))
        if augment:
            #input_feed = cifar.random_crop_and_flip(input_feed, padding_size=2)
            input_feed = cifar.whitening_image(input_feed)
        feed_dict = {
                self.input_placeholder : input_feed,
                self.label_placeholder : labels_feed,
                self.lr                : lr,
                self.is_training       : is_training
        }
        return feed_dict

    def fill_feed_dict_manual(self, x_data, y_data, lr=0, augment=True, is_training=False):
        X, Y = np.array(x_data), np.array(y_data)
        input_feed, labels_feed = X.reshape(len(Y), -1), Y.reshape(-1)
        if augment:
            #input_feed = cifar.random_crop_and_flip(input_feed, padding_size=2)
            input_feed = cifar.whitening_image(input_feed)
        feed_dict = {
                self.input_placeholder : input_feed,
                self.label_placeholder : labels_feed,
                self.lr                : lr,
                self.is_training       : is_training
        }
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
        return tf.summary.FileWriter(boardWrite_dir, self.sess.graph)


    def save_model(self, epoch):
        saver = tf.train.Saver()
        path = load_model_dir  + str(epoch)
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
            train_iter = num_data//batch_size if num_data%batch_size == 0 else num_data // batch_size+1
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
            test_iter = num_test//batch_size if num_test%batch_size==0 else num_test//batch_size+1
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


    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))        
        print('Total number of parameters: %s' % self.num_params)
        
        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list

    def get_inverse_hvp(self, v, x_train, y_train, approx_type='lissa', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, x_train, y_train)
            #return self.get_inverse_hvp_lissa(v, **approx_params)
        #elif approx_type == 'cg':
            #return self.get_inverse_hvp_cg(v, verbose)


    def get_inverse_hvp_lissa(self, v, x_train, y_train, 
                              batch_size=10,
                              scale=10, damping=0.0, num_samples=10, recursion_depth=100):
        """
        This uses mini-batching; uncomment code for the single sample case.
        """    
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)
           
            cur_estimate = v
            norm = 1

            for j in range(recursion_depth):
                
                test_idx = np.random.randint(0, len(y_train), batch_size)
                #feed_dict = self.fill_feed_dict_with_batch(x_train, y_train, is_training=False, batch_size=1, update_index=False)
                feed_dict = self.fill_feed_dict_with_some_ex(x_train, y_train, test_idx)

                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                
                temp_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)]
                temp_norm = np.linalg.norm(np.concatenate(cur_estimate))
                if temp_norm > 100:
                    break
                else:
                    temp = temp_norm
                    cur_estimate = temp_estimate
                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    #print("Cur_estimate: {}".format(cur_estimate))
                    print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(np.concatenate(cur_estimate))))
                    feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b/scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, cur_estimate)]  

        inverse_hvp = [a/num_samples for a in inverse_hvp]
        return inverse_hvp


    def get_all_params(self):
        graph = tf.get_default_graph()
        all_params = []
        temp_tensor = graph.get_tensor_by_name("stage0/conv:0")
        all_params.append(temp_tensor)
        for stage in ['stage1_res_0', 'stage1_res_1', 'stage1_res_2', 'stage2_res_0', 'stage2_res_1', 'stage2_res_2', 'stage3_res_0', 'stage3_res_1', 'stage3_res_2']:
            for block in ['conv1_in_block', 'conv2_in_block']:
                temp_tensor = "{}/{}/conv:0".format(stage, block)
                tensor = graph.get_tensor_by_name(temp_tensor)
                all_params.append(tensor)
        all_params.append( graph.get_tensor_by_name("fc/fc_weights:0") )
        all_params.append( graph.get_tensor_by_name("fc/fc_bias:0") )
        #temp_tensor = graph.get_tensor_by_name("stage1_res_2/conv1_in_block/conv:0")
        #all_params.append(temp_tensor)
        return all_params 


    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block        
        return feed_dict


    def get_test_grad_loss_no_reg_val(self, test_indices, x_test, y_test, batch_size=100, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        else:
            raise ValueError('Loss must be specified').with_traceback()

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(x_test, y_test, test_indices[start:end], augment=False)
                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in zip(test_grad_loss_no_reg_val, temp)]
            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]
        
        return test_grad_loss_no_reg_val


    def get_influence_on_test_loss(self, test_indices, train_idx, x_train, y_train, 
            x_test, y_test, approx_type='lissa', approx_params=None, force_refresh=True, 
            test_description=None, loss_type='normal_loss', X=None, Y=None):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order

        if train_idx is None: 
            if (X is None) or (Y is None): raise ValueError('X and Y must be specified if using phantom points.').with_traceback()
            if X.shape[0] != len(Y): raise ValueError('X and Y must have the same length.').with_traceback()
        else:
            if (X is not None) or (Y is not None): raise ValueError('X and Y cannot be specified if train_idx is specified.').with_traceback()

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, x_test, y_test, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val, x_train, y_train, 
                approx_type,
                approx_params)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)

        start_time = time.time()
        if train_idx is None:
            num_to_remove = len(Y)
            predicted_loss_diffs = np.zeros([num_to_remove])            
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :], [Y[counter]])      
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples

        else:            
            num_to_remove = len(train_idx)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in tqdm(enumerate(train_idx)):
                #savez code, if you want get_params, uncomment it
                '''
                single_train_feed_dict = self.fill_feed_dict_with_some_ex(x_train, y_train, [idx_to_remove])
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                np.savez(params_dir+"/train_{}".format(counter), loss_val=train_grad_loss_val)
                #embed(header="load np")
                #train_grad_loss_val = np.load(params_dir+"train_{}.npz".format(counter))['loss_val']
                #predicted_loss_diffs[counter] = np.sum( np.multiply(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val))) / len(y_train)
                #predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / len(y_train)
                '''
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(self.train_grad_loss_list[counter])) / len(y_train)
                
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs


if __name__ == "__main__":
    x_train, y_train = cifar.prepare_train_data(padding_size=2, shuffle=True)
    #drop_idx = np.load("/unsullied/sharefs/wangfeng02/logs/inf_copy/drop_idx/drop_0_to_5000.npz")['idx']
    #drop_idx = np.random.randint(0, 50000, 9700)
    #drop_idx = np.array( list(set(drop_idx)) )

    #x_train, y_train = cifar.prepare_drop_train_data(padding_size=2, drop_idx=drop_idx, shuffle=False)
    x_test, y_test = cifar.read_validation_data(shuffle=True)
    num_data = len(y_train)
    num_test = len(y_test)
    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train),(x_test, y_test) = mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = ResnetModel(load=False)
    #model.train(epoch=100, batch_size=128)
    #embed(header="load all model")
