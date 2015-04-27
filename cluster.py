import scipy.io as sio
import glob
import numpy as np


'''Loads the objects for training and testing a class in the MSCoco dataset.

Args:
    training_percentage (int): Number from 0 to 100. The percentage of the files
        that will be used in the training set, the rest will be in the testing.
    class_name (string): Name of the class that will be loaded (e.g. "airplane")

Returns:
    np.array: objects for the training set
    np.array: objects for the testing set
'''
def load_class(training_percentage, class_name):
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats/"
    files = glob.glob("{0}/{1}/*.mat".format())
    train = None
    test = None
    training_count = (len(files) * training_percentage) / 100
    for i in range(len(files)):
        f = files[i]
        data = sio.loadmat(f)
        features = data["stored"]
        if i < training_count:
            if train is None:
                train = features
            else:
                train = np.vstack(train, features)
        else:
            if test is None:
                test = features
            else:
                test = np.vstack(test, features)

    return [train, test]
