"""Implementation of sample defense.
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time

import numpy as np
from scipy.misc import imread

from pyunlocbox import functions, solvers # TODO: NEED TO BE ADDED TO DOCKER

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception #, vgg 

slim = tf.contrib.slim

DEFAULT_IMAGE_HEIGHT = 299
DEFAULT_IMAGE_WIDTH = 299
DEFAULT_BATCH_SIZE = 100
DEFAULT_BATCH_SHAPE = [DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_HEIGHT, 
        DEFAULT_IMAGE_WIDTH, 3]

DEFAULT_CHECKPOINT_PATH = './inception_v3.ckpt'

tf.flags.DEFINE_string(
    'checkpoint_path', DEFAULT_CHECKPOINT_PATH, 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

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

#tf.flags.DEFINE_float(
#    'noise_std', DEFAULT_NOISE_STD, 'Additive Gaussian noise std.')

tf.flags.DEFINE_integer(
    'start', 0, 'start index.')

tf.flags.DEFINE_integer(
    'end', None, 'end index.')

FLAGS = tf.flags.FLAGS


def denoise(im, eps):
    """
    http://pyunlocbox.readthedocs.io/en/latest/tutorials/denoising.html
    """
    f1 = functions.norm_tv(maxit=50, dim=3)
    y = np.reshape(im, -1)
    f = functions.proj_b2(y=y, epsilon=eps)
    f2 = functions.func()
    f2._eval = lambda x: 0
    def prox(x, step):
        return np.reshape(f.prox(np.reshape(x, -1), 0), im.shape)
    f2._prox = prox

    solver = solvers.douglas_rachford(step=0.1)
    im0 = np.array(im)
    ret = solvers.solve([f1, f2], im0, solver)
    return ret['sol']


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

    if start == 0 and end is None:
        filepaths = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
    else:
        filepaths = np.sort(tf.gfile.Glob(os.path.join(input_dir, '*.png')))
        filepaths = filepaths[start:end]
    print(len(filepaths))
    for filepath in filepaths:
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = denoise(image * 2.0 - 1.0, eps = 16)
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def prepare_graph(batch_shape = DEFAULT_BATCH_SHAPE, num_classes = 1001,
        checkpoint_path = DEFAULT_CHECKPOINT_PATH):
    graph = tf.Graph()
    with graph.as_default():
        # prepare graph
        images = tf.placeholder(tf.float32, shape=batch_shape, name='images')
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            (logits, end_points) = inception.inception_v3(images, 
                    num_classes=num_classes, is_training=False)
            
            config = tf.ConfigProto()
            # allocate memory based on need
            if not FLAGS.using_docker:
                config.gpu_options.allow_growth = True

            # restore checkpoint
            sess = tf.Session(config=config, graph=graph)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

    prediction = tf.argmax(logits, 1)

    tensors_dict = {'images': images, 'prediction': prediction}
    ops_dict = {}

    return (graph, sess, tensors_dict, ops_dict)


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
    
    (graph, sess, tensors_dict, _) = prepare_graph(batch_shape = batch_shape, 
            num_classes = num_classes,
            checkpoint_path = FLAGS.checkpoint_path)

    images = tensors_dict['images']
    prediction = tensors_dict['prediction']

    c = FLAGS.start
    for filenames, images_v in load_images(FLAGS.input_dir, batch_shape, 
            FLAGS.start, FLAGS.end):
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
