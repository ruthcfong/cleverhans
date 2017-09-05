import os, sys

import numpy as np

def main(argv):
    results_f = argv[0]
    dataset_csv = '/users/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/dev_dataset.csv'

    image_ids = np.loadtxt(dataset_csv, delimiter=',', usecols=0, skiprows=1, dtype='S20') 
    gt_labels = np.loadtxt(dataset_csv, delimiter=',', usecols=6, skiprows=1, dtype='int32')
    image_ids_2 = np.array([f.strip('.png') for f in np.loadtxt(results_f, delimiter=',', usecols=0, dtype='S20')])
    pred_labels = np.loadtxt(results_f, delimiter=',', usecols=1, dtype='int32')

    sorted_idx = np.argsort(image_ids)
    sorted_idx_2 = np.argsort(image_ids_2)

    #assert(np.array_equal(image_ids[sorted_idx], image_ids_2[sorted_idx_2]))
    #assert(len(image_ids) == len(filenames))

    N = len(pred_labels)
    assert(np.array_equal(image_ids[sorted_idx[:N]], image_ids_2[sorted_idx_2]))
    
    num_correct = len(np.where(gt_labels[sorted_idx[:N]] == pred_labels[sorted_idx_2])[0])
    print '%d/%d: %f' % (num_correct, N, num_correct/float(N))

if __name__ == '__main__':
    main(sys.argv[1:])
