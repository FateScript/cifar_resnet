import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('weight_decay', 0.0001, '''weight decay value''')
tf.app.flags.DEFINE_integer('num_blocks', 3, '''block number in one stage''')

