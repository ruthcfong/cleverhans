"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import time

import numpy as np
from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, vgg

from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize

from PIL import ImageFilter, Image

slim = tf.contrib.slim

DEFAULT_IMAGE_HEIGHT = 299
DEFAULT_IMAGE_WIDTH = 299
DEFAULT_BATCH_SIZE = 128
#DEFAULT_EPOCHS = 300
#DEFAULT_LEARNING_RATE = 0.1

DEFAULT_NOISE_STD = 0.2
DEFAULT_CHECKPOINT_PATH = './inception_v3.ckpt'
DEFAULT_NET_TYPE = 'googlenet'

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', DEFAULT_CHECKPOINT_PATH, 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_string(
    'net_type', DEFAULT_NET_TYPE, 'Choose from ["googlenet", "vgg19"]')

tf.flags.DEFINE_integer(
    'image_width', DEFAULT_IMAGE_WIDTH, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', DEFAULT_IMAGE_WIDTH, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', DEFAULT_BATCH_SIZE, 'How many images process at one time.')

tf.flags.DEFINE_boolean(
    'using_docker', False, 'Using a self-contained docker container.')

tf.flags.DEFINE_integer(
    'gpu', None, 'What GPU device to use.')

tf.flags.DEFINE_float(
    'noise_std', DEFAULT_NOISE_STD, 'Additive Gaussian noise std.')

tf.flags.DEFINE_integer(
    'start', 0, 'start index.')

tf.flags.DEFINE_integer(
    'end', None, 'end index.')

FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape, start = 0, end = None):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  filepaths = np.sort(tf.gfile.Glob(os.path.join(input_dir, '*.png')))
  if end is not None:
      filepaths = filepaths[start:end]
  for filepath in filepaths:
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images

def prepare_graph(noise_std = DEFAULT_NOISE_STD, batch_shape = [DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, 3],
        num_classes = 1001, checkpoint_path = DEFAULT_CHECKPOINT_PATH, net_type = DEFAULT_NET_TYPE):
    graph = tf.Graph()
    with graph.as_default():
        # Prepare graph
        images = tf.placeholder(tf.float32, shape=batch_shape, name='images')

        noise_gate = tf.Variable(1.0, name='noise_gate')
        noise_off_op = tf.assign(noise_gate, 0.0)
        noise_on_op = tf.assign(noise_gate, 1.0)
        #noise_std = tf.Variable(0.2, name='noise_std')
        noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=noise_std, name='noise')

        inputs = images + noise_gate * noise

        if net_type == 'googlenet':
            arg_scope = inception.inception_v3_arg_scope()
            net = inception.inception_v3
        elif net_type == 'vgg19':
            arg_scope = vgg.vgg_arg_scope()
            net = vgg.vgg_19
        else:
            assert(False)

        with slim.arg_scope(arg_scope):
            (logits, end_points) = net(inputs, num_classes=num_classes,
                    is_training=False)

            # Allocate memory based on need
            config = tf.ConfigProto()
            if not FLAGS.using_docker:
                config.gpu_options.allow_growth = True

            # Restore the checkpoint
            sess = tf.Session(config=config, graph=graph)
            saver = tf.train.Saver(var_list = slim.get_model_variables())
            saver.restore(sess, checkpoint_path)

        # Construct the scalar neuron tensor
        neuron_selector = tf.placeholder(tf.int32, name='neuron_selector')
        y = logits[0][neuron_selector]

        # Construct tensor for predictions.
        prediction = tf.argmax(logits, 1)

        # Construct tensor for softmax scores
        if net_type == 'googlenet':
            softmax = end_points['Predictions']
        elif net_type == 'vgg19':
            softmax = slim.softmax(logits, scope='Predictions')

        # Define loss terms
        #tot_loss = score_loss + l1_lambda * l1_loss + tv_lambda * tv_norm_loss

        sess.run(noise_gate.initializer)

        # Set-up mask optimizer
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        #train = optimizer.minimize(tot_loss, var_list=[mask])

        #optimizer_slots = [
        #    optimizer.get_slot(mask, name)
        #    for name in optimizer.get_slot_names()
        #]
        #optimizer_slots.extend([optimizer._beta1_power, optimizer._beta2_power])

        #init_optimizer_op = tf.variables_initializer(optimizer_slots)

    tensors_dict = {
        'images': images,
        'noise_gate': noise_gate,
        'noise': noise,
        'neuron_selector': neuron_selector,
        'prediction': prediction,
        'softmax': softmax,
    }

    ops_dict = {
        'noise_on_op': noise_on_op,
        'noise_off_op': noise_off_op
    }

    return graph, sess, tensors_dict, ops_dict

def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.using_docker:
        gpu = FLAGS.gpu

        cuda = True if gpu is not None else False
        use_mult_gpu = isinstance(gpu, list)
        if cuda:
            if use_mult_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    (graph, sess, tensors_dict, ops_dict) = prepare_graph(
        noise_std = FLAGS.noise_std,
        batch_shape = batch_shape, 
        num_classes = num_classes, 
        checkpoint_path = FLAGS.checkpoint_path,
        net_type = FLAGS.net_type)
 
    #softmax = tensors_dict['softmax']
    #noise_on_op = ops_dict['noise_on_op']
    #noise_off_op = ops_dict['noise_off_op']
    #init_optimizer_op = ops_dict['init_optimizer_op']
    #train = ops_dict['train']

    prediction = tensors_dict['prediction']
    images = tensors_dict['images']

    c = FLAGS.start 
    for filenames, images_v in load_images(FLAGS.input_dir, batch_shape, FLAGS.start, FLAGS.end):
        start = time.time()
        labels = sess.run(prediction, feed_dict={images: images_v})
        
        out_file = open(FLAGS.output_file, 'a')
        for filename, label in zip(filenames, labels):
            out_file.write('{0},{1}\n'.format(filename, label))
        out_file.close()

        print('batch %d took %.4f secs' % (c, time.time() - start))
        c += batch_shape[0] 

if __name__ == '__main__':
  tf.app.run()
