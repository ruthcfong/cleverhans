import os, sys
import csv
import numpy as np

def main(argv):
    dataset_dir = argv[0]
    out_path = os.path.join(dataset_dir, 'target_class.csv')
    filenames = [f for f in os.listdir(dataset_dir) if 'png' in f]
    target_classes = np.random.randint(1,1001,len(filenames))
    with open(out_path, 'wb') as f:
        writer = csv.writer(f)
        for i in range(len(filenames)):
            writer.writerow([filenames[i], target_classes[i]])

if __name__ == "__main__":
    main(sys.argv[1:])
