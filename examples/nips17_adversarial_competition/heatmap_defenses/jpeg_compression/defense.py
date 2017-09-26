"""Implementation of sample defense.
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time, tempfile

import numpy as np
from scipy.misc import imread, imresize

from PIL import Image

#from pyunlocbox import functions, solvers # TODO: NEED TO BE ADDED TO DOCKER

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception #, vgg 
import inception_resnet_v2

slim = tf.contrib.slim

DEFAULT_IMAGE_HEIGHT = 299
DEFAULT_IMAGE_WIDTH = 299
DEFAULT_BATCH_SIZE = 100
DEFAULT_BATCH_SHAPE = [DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_HEIGHT, 
        DEFAULT_IMAGE_WIDTH, 3]

DEFAULT_CHECKPOINT_PATH = './inception_v3.ckpt'

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', DEFAULT_CHECKPOINT_PATH, 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file.')

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

tf.flags.DEFINE_string(
    'interp', 'bilinear', 'Interpolation method; choose from ["nearest", "lanczos", "bilinear", "bicubic", or "cubic"]') 

tf.flags.DEFINE_string(
    'net_type', 'googlenet', 'Choose from ["googlenet", "resnet"]')

tf.flags.DEFINE_integer(
    'compression_rate', 50, 'Choose JPEG compression quality')

tf.flags.DEFINE_boolean(
    'downsample', False, 'Downsample compressed image.')

tf.flags.DEFINE_float(
    'downsample_rate', 0.5, 'Choose downsampling factor')

#tf.flags.DEFINE_float(
#    'noise_std', DEFAULT_NOISE_STD, 'Additive Gaussian noise std.')

tf.flags.DEFINE_integer(
    'start', 0, 'start index.')

tf.flags.DEFINE_integer(
    'end', None, 'end index.')

FLAGS = tf.flags.FLAGS


def denoise(im, downsample_rate=0.5, interp='bilinear'):                                                                
    im_size = im.shape[:2]                                                      
    down_sampled = imresize(im, downsample_rate) 
    up_sampled = imresize(down_sampled, im_size, interp=interp)           
    return up_sampled                                                           


def compress_img(img_path, img_quality=50):
    img = Image.open(img_path)
    temp_f = tempfile.TemporaryFile()
    img.save(temp_f, "JPEG", quality=img_quality)
    img_ = imread(temp_f, mode='RGB')
    return img_


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

    if FLAGS.using_docker:
        filepaths = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
    else:
        filepaths = np.sort(tf.gfile.Glob(os.path.join(input_dir, '*.png')))
        if end is not None:
            filepaths = filepaths[start:end]
    for filepath in filepaths:
        with tf.gfile.Open(filepath) as f:
            image = compress_img(f, img_quality=FLAGS.compression_rate).astype(np.float)
        if FLAGS.downsample:
            image = denoise(image, downsample_rate=FLAGS.downsample_rate, 
                    interp=FLAGS.interp)
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image / 255. * 2.0 - 1.0
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

        if FLAGS.net_type == 'googlenet':
            arg_scope = inception.inception_v3_arg_scope()
            net = inception.inception_v3
        elif FLAGS.net_type == 'resnet':
            arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
            net = inception_resnet_v2.inception_resnet_v2
        else:
            assert(False)

        # prepare graph
        images = tf.placeholder(tf.float32, shape=batch_shape, name='images')
        with slim.arg_scope(arg_scope):
            (logits, end_points) = net(images, num_classes=num_classes, 
                    is_training=False)
            
            config = tf.ConfigProto()
            # allocate memory based on need
            if not FLAGS.using_docker:
                config.gpu_options.allow_growth = True

            # restore checkpoint
            sess = tf.Session(config=config, graph=graph)
            saver = tf.train.Saver(slim.get_model_variables())
            saver.restore(sess, checkpoint_path)

            #session_creator = tf.train.ChiefSessionCreator(
            #        scaffold=tf.train.Scaffold(saver=saver),
            #        checkpoint_filename_with_path=checkpoint_path,
            #        master=FLAGS.master)

            #sess = tf.train.MonitoredSession(session_creator=session_creator)

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
