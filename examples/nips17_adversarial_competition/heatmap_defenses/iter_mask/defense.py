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
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.1

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
    'mask_dir', None, 'Output dir to save learned masks.')

tf.flags.DEFINE_string(
    'net_type', DEFAULT_NET_TYPE, 'Choose from ["googlenet", "vgg19"]')

tf.flags.DEFINE_integer(
    'image_width', DEFAULT_IMAGE_WIDTH, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', DEFAULT_IMAGE_WIDTH, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', DEFAULT_BATCH_SIZE, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'gpu', 1, 'What GPU device to use.')

tf.flags.DEFINE_integer(
    'epochs', DEFAULT_EPOCHS, 'How many epochs to run iterative mask optimization.')

tf.flags.DEFINE_float(
    'learning_rate', DEFAULT_LEARNING_RATE, 'Learning rate for Adam optimizer.')

tf.flags.DEFINE_float(
    'tv_beta', 3.0, 'TV beta hyperparameter.')

tf.flags.DEFINE_float(
    'tv_lambda', 0.2, 'TV lambda hyperparameter.')

tf.flags.DEFINE_float(
    'l1_lambda', 0.005, 'L1 lambda hyperparameter.')

tf.flags.DEFINE_integer(
    'start', 0, 'start index.')

tf.flags.DEFINE_integer(
    'end', None, 'end index.')

tf.flags.DEFINE_boolean(
    'add_noise', False, 'add Gaussian noise to input.')

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

def blur(img, radius = 10):
    img = Image.fromarray(np.uint8((img+1)*127.5))
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.array(blurred_img)/127.5-1

def create_blurred_circular_mask(mask_shape, radius, center = None, sigma = 10):
    assert(len(mask_shape) == 2)
    if center is None:
        x_center = int(mask_shape[1]/float(2))
        y_center = int(mask_shape[0]/float(2))
    else:
        (x_center, y_center) = center
    y,x = np.ogrid[-y_center:mask_shape[0]-y_center, -x_center:mask_shape[1]-x_center]
    mask = x*x + y*y <= radius*radius
    grid = np.zeros(mask_shape)
    grid[mask] = 1
    if sigma is not None:
        grid = gaussian_filter(grid, sigma)
    return grid

def create_blurred_circular_mask_pyramid(radii = np.arange(0,25,2), mask_shape = (32,32), sigma = 2):
    assert(len(mask_shape) == 2)
    num_masks = len(radii)
    masks = np.zeros((num_masks, mask_shape[0], mask_shape[1], 1))
    for i in range(num_masks):
        masks[i,:,:,0] = create_blurred_circular_mask(mask_shape, radii[i], sigma = sigma)
    return masks

def prepare_graph(batch_shape = [DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, 3],
        mask_shape = (32,32,1), num_classes = 1001, checkpoint_path = DEFAULT_CHECKPOINT_PATH, 
        learning_rate = DEFAULT_LEARNING_RATE, net_type = DEFAULT_NET_TYPE, add_noise = False):
    graph = tf.Graph()
    with graph.as_default():
        # Prepare graph
        images = tf.placeholder(tf.float32, shape=batch_shape, name='images')
        null_images = tf.placeholder(tf.float32, shape=batch_shape, name='null_images')
        mask = tf.get_variable("mask", shape=mask_shape, initializer=tf.ones_initializer())
        mask_up = tf.image.resize_images(mask,(batch_shape[1], batch_shape[2]))
        if add_noise:
            noise_gate = tf.Variable(1.0, name='noise_gate')
            noise_off_op = tf.assign(noise_gate, 0.0)
            noise_on_op = tf.assign(noise_gate, 1.0)
            noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=0.2, name='noise')
            inputs = images * mask_up + null_images * (1 - mask_up) + noise_gate * noise
        else:
            inputs = images * mask_up + null_images * (1 - mask_up)

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

        # Prepare hyper-parameters for learning iterative mask
        l1_lambda = tf.placeholder(tf.float32, name='l1_lambda')
        tv_lambda = tf.placeholder(tf.float32, name='tv_lambda')
        tv_beta = tf.placeholder(tf.float32, name='tv_beta')

        # Define loss terms
        score_loss = softmax[0][neuron_selector]
        l1_loss = tf.reduce_sum(tf.abs(1-mask))
        tv_norm_loss = (tf.reduce_mean(tf.pow(tf.abs(mask[:-1,:,0]-mask[1:,:,0]),tv_beta))
                        + tf.reduce_mean(tf.pow(tf.abs(mask[:,:-1,0]-mask[:,1:,0]),tv_beta)))
        tot_loss = score_loss + l1_lambda * l1_loss + tv_lambda * tv_norm_loss

        clip_mask_op = tf.assign(mask, tf.clip_by_value(mask, 0, 1))
        no_mask_op = tf.assign(mask, tf.ones_like(mask))

        masks_pyramid = 1 - create_blurred_circular_mask_pyramid(mask_shape = mask_shape[:2])
        init_mask_pyramid_ops = [tf.assign(mask, m) for m in masks_pyramid]

        sess.run(mask.initializer)
        if add_noise:
            sess.run(noise_gate.initializer)

        # Set-up mask optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(tot_loss, var_list=[mask])

        optimizer_slots = [
            optimizer.get_slot(mask, name)
            for name in optimizer.get_slot_names()
        ]
        optimizer_slots.extend([optimizer._beta1_power, optimizer._beta2_power])

        init_optimizer_op = tf.variables_initializer(optimizer_slots)

    tensors_dict = {
        'images': images,
        'null_images': null_images,
        'mask': mask,
        'mask_up': mask_up,
        'neuron_selector': neuron_selector,
        'prediction': prediction,
        'softmax': softmax,
        'l1_lambda': l1_lambda, 
        'tv_lambda': tv_lambda,
        'tv_beta': tv_beta,
        'score_loss': score_loss,
        'l1_loss': l1_loss,
        'tv_norm_loss': tv_norm_loss,
        'tot_loss': tot_loss
    }

    ops_dict = {
        'clip_mask_op': clip_mask_op,
        'no_mask_op': no_mask_op,
        'init_mask_pyramid_ops': init_mask_pyramid_ops,
        'train': train,
        'init_optimizer_op': init_optimizer_op
    }

    if add_noise:
        ops_dict['noise_on_op'] = noise_on_op
        ops_dict['noise_off_op'] = noise_off_op

    return graph, sess, tensors_dict, ops_dict

def get_top_predicted_class(im, null_im, tensors_dict, sess):
    prediction = tensors_dict['prediction']
    images = tensors_dict['images']
    null_images = tensors_dict['null_images']

    return sess.run(prediction, feed_dict = {images: [im],
                                             null_images: [null_im]})[0]

def get_mask_initialization(im, null_im, top_predicted_class,
        softmax, init_mask_pyramid_ops, tensors_dict, sess, init_thres = 1e-2):
    mask_init_scores = np.zeros(len(init_mask_pyramid_ops))

    images = tensors_dict['images']
    null_images = tensors_dict['null_images']

    for i in range(len(init_mask_pyramid_ops)):
        sess.run(init_mask_pyramid_ops[i])

        softmax_scores = sess.run(softmax, feed_dict = {images: [im],
                                            null_images: [null_im]})[0]
        mask_init_scores[i] = softmax_scores[top_predicted_class]

    try:
        first_i = np.where(mask_init_scores < init_thres)[0][0]
    except:
        first_i = -1

    return init_mask_pyramid_ops[first_i]

def get_feed_dict(im, null_im, class_label, tensors_dict, values_dict):
    return {tensors_dict['l1_lambda']: values_dict['l1_lambda'],
            tensors_dict['tv_lambda']: values_dict['tv_lambda'],
            tensors_dict['tv_beta']: values_dict['tv_beta'],
            tensors_dict['neuron_selector']: class_label,
            tensors_dict['images']: [im],
            tensors_dict['null_images']: [null_im]}

def learn_mask(train, prediction, init_ops, clip_mask_op, feed_dict, sess, num_epochs = 300):
    # Initialize mask and optimizer
    sess.run(init_ops)

    # Learn perturbation mask
    for t in range(num_epochs):
        sess.run(train, feed_dict = feed_dict)
        sess.run(clip_mask_op)

    top_class = sess.run(prediction, feed_dict = feed_dict)[0]
    return top_class

def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

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

    if FLAGS.mask_dir is not None and not os.path.exists(FLAGS.mask_dir):
        os.makedirs(FLAGS.mask_dir)
        print('Created %s' % FLAGS.mask_dir)

    add_noise = FLAGS.add_noise
    print(add_noise)
    
    (graph, sess, tensors_dict, ops_dict) = prepare_graph(
        batch_shape = batch_shape, 
        mask_shape = (32,32,1), 
        num_classes = num_classes, 
        checkpoint_path = FLAGS.checkpoint_path,
        learning_rate = FLAGS.learning_rate,
        add_noise = add_noise)
 
    init_mask_pyramid_ops = ops_dict['init_mask_pyramid_ops']
    softmax = tensors_dict['softmax']

    if add_noise:
        noise_on_op = ops_dict['noise_on_op']
        noise_off_op = ops_dict['noise_off_op']

    init_optimizer_op = ops_dict['init_optimizer_op']
    clip_mask_op = ops_dict['clip_mask_op']
    train = ops_dict['train']

    prediction = tensors_dict['prediction']
    mask = tensors_dict['mask']

    values_dict = {
            'l1_lambda': FLAGS.l1_lambda,
            'tv_lambda': FLAGS.tv_lambda,
            'tv_beta': FLAGS.tv_beta
    }

    c = FLAGS.start 
    for filenames, images_v in load_images(FLAGS.input_dir, batch_shape, FLAGS.start, FLAGS.end):
        assert(len(filenames) == 1 and len(images_v) == 1)
        start = time.time()

        im = images_v[0]
        null_im = blur(im, 10)

        if add_noise:
            sess.run(noise_off_op)

        top_predicted_class = get_top_predicted_class(im, null_im, tensors_dict, sess)

        if add_noise:
            sess.run(noise_on_op)

        mask_init_op = get_mask_initialization(im, null_im, top_predicted_class,
                softmax, init_mask_pyramid_ops, tensors_dict, sess, init_thres = 1e-2)

        init_ops = [mask_init_op, init_optimizer_op]

        feed_dict = get_feed_dict(im, null_im, top_predicted_class, tensors_dict, values_dict)
        perturbed_top_class = learn_mask(train, prediction, init_ops, clip_mask_op, 
                feed_dict, sess, num_epochs = FLAGS.epochs)
        
        out_file = open(FLAGS.output_file, 'a')
        out_file.write('{0},{1}\n'.format(filenames[0], perturbed_top_class))
        out_file.close()

        if FLAGS.mask_dir is not None:
            m = np.squeeze(sess.run(mask))
            np.save(os.path.join(FLAGS.mask_dir, '%s.npy' % filenames[0].strip('.png')), m)

        print('%i: %s took %.4f secs' % (c, filenames[0], time.time() - start))
        c += 1

if __name__ == '__main__':
  tf.app.run()
