import os, sys

import numpy as np
from scipy.stats import mode

def main(argv):
    num_results_f = int(argv[0])
    results_fs = []
    for i in range(num_results_f):
        results_fs.append(argv[1+i])
    dataset_csv = '/users/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/dev_dataset.csv'
    if not os.path.exists(dataset_csv):
        dataset_csv = '/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/dev_dataset.csv'

    image_ids = np.loadtxt(dataset_csv, delimiter=',', usecols=0, skiprows=1, dtype='S20') 
    gt_labels = np.loadtxt(dataset_csv, delimiter=',', usecols=6, skiprows=1, dtype='int32')
    pred_labels = np.zeros([num_results_f, len(gt_labels)])
    sorted_idx = np.argsort(image_ids)

    for i in range(num_results_f):
        results_f = results_fs[i]
        image_ids_res = np.array([f.strip('.png') for f in np.loadtxt(
            results_f, delimiter=',', usecols=0, dtype='S20')])
        pred_labels_res = np.loadtxt(results_f, delimiter=',', usecols=1, dtype='int32')
        sorted_idx_res = np.argsort(image_ids_res)
        assert(np.array_equal(image_ids[sorted_idx], image_ids_res[sorted_idx_res]))
        pred_labels[i] = pred_labels_res[sorted_idx_res]

    (modes, m_counts) = mode(pred_labels, axis=0)

    # if there's no consensus, use the first result
    for i in np.where(m_counts == 1)[1]:
        modes[0][i] = pred_labels[0][i]

    num_correct = len(np.where(gt_labels[sorted_idx] == modes)[0])
    #print '%d/%d: %f' % (num_correct, N, num_correct/float(N))
    print(num_correct/float(len(gt_labels)))

if __name__ == '__main__':
    main(sys.argv[1:])
