from defense import denoise
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np

im = imread('/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/attack_images/iter_target_class/eps_16/b8596bba57a73794.png', mode='RGB')
im_new = denoise(im, 16)

print np.sum(np.abs(im_new - im))

f, ax = plt.subplots(1, 3)
ax[0].imshow(im)
ax[1].imshow(im_new)
ax[2].imshow((im_new-im)*255)
plt.show()

