from defense import denoise
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import os

clean_dir = '/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/images'
adv_dir = '/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/attack_images/iter_target_class/eps_16'

filename = 'b8596bba57a73794.png'
im_clean = imread(os.path.join(clean_dir, filename), mode='RGB')
im = imread(os.path.join(adv_dir, filename), mode='RGB')
im = im / 255.0 * 2.0 - 1.0
im_new = denoise(im, 16 / 255.0)

print np.sum(np.abs(im_new - im))

f, ax = plt.subplots(1, 4)
ax[0].imshow((im + 1.0) / 2.0 * 255.0)
ax[1].imshow((im_new + 1.0) / 2.0 * 255.0)
ax[2].imshow(((im_new-im) + 1.0) / 2.0 * 255.0)
ax[3].imshow(im - im_clean)
plt.show()

