#from defense import denoise
from scipy.misc import imread, imresize
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import os
from pyunlocbox import functions, solvers

clean_dir = '/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/images'
adv_dir = '/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/attack_images/iter_target_class/eps_16'

def normalize(im):
    return im
    #return im / 255.0 * 2.0 - 1.0

def denormalize(im):
    return im
    #return (im + 1.0) / 2.0 * 255.0

def denoise(im, _):
    im_size = im.shape[:2]
    down_sampled = imresize(im, (150, 150))
    up_sampled = imresize(down_sampled, im_size)
    #filtered = ndimage.gaussian_filter(up_sampled, 1)
    #alpha = 1.0 
    #im_new = up_sampled + alpha * (up_sampled - filtered)
    return up_sampled 
    

filename = 'b8596bba57a73794.png'
im_clean = imread(os.path.join(clean_dir, filename), mode='RGB')
im_clean = normalize(im_clean)
im = imread(os.path.join(adv_dir, filename), mode='RGB')
im = normalize(im)
im_new = denoise(im, 16.)
print np.max(im_clean), np.min(im_clean)
print np.max(im_new), np.min(im_new)

f_tv = functions.norm_tv()
print 'clean', f_tv.eval(im_clean)
print 'adv', f_tv.eval(im)
print 'adv new', f_tv.eval(im_new)

print np.sum(np.abs(im_new-im))
print np.sum(np.abs(im_new-im_clean))

f, ax = plt.subplots(1, 5)
ax[0].imshow(denormalize(im))
ax[0].set_title('Adversarial')
ax[1].imshow(denormalize(im_new))
ax[1].set_title('Restored')
ax[2].imshow(denormalize(im-im_new))
ax[2].set_title('Adv-Rest')
ax[3].imshow(denormalize(im-im_clean))
ax[3].set_title('Adv-Clean')
ax[4].imshow(denormalize(im_new-im_clean))
ax[4].set_title('Rest-Clean')
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.show()

