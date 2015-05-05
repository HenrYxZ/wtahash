import scipy.io as sio
import glob
import numpy as np


'''Loads the objects for a class in the MSCoco dataset.

Args:
    training_percentage (int): Number from 0 to 100. The percentage of the files
        that will be used in the training set, the rest will be in the testing.
    class_name (string): Name of the class that will be loaded (e.g. "airplane")
    set_name (string): It can be training or testing

Returns:
    np.array float: objects for the set
'''
def load_class(training_percentage, path, set_name):
    print ("path = {0}".format(path))
    files = glob.glob("{0}/*.mat".format(path))
    objects = None
    training_count = (len(files) * training_percentage) / 100
    print ("training_count = {0}".format(training_count))
    if set_name == "training":
        my_range = range(training_count)
    else:
        my_range = range(training_count, len(files))
    for i in my_range:
        f = files[i]
        data = sio.loadmat(f)
        features = data["stored"]
        if objects is None:
            objects = features
        else:
            objects = np.vstack((objects, features))
    print ("Done getting the objects from {0}".format(path))
    print ("Objects matrix of shape = {0}".format(objects.shape))
    return objects

'''Loads the objects for all the classes in the MSCoco dataset.

Args:
    training_percentage (int): Number from 0 to 100. The percentage of the files
        that will be used in the training set, the rest will be in the testing.
    set_name (string): It can be training or testing

Returns:
    np.array float: objects for the set
'''
def load_classes(training_percentage, path, set_name):
    folders = glob.glob("{0}/*".format(path))
    objects = None
    # For each folder get the objects of that class
    for i in range(len(folders)):
        full_path = path + "/" + folders[i]
        this_class = load_class(training_percentage, full_path, set_name)
        if objects == None:
            objects = this_class
        else:
            objects = np.vstack((objects, this_class))
    print (
        "Done getting the objects from all the classes, final matrix of shape "\
        "= {0}".format(objects.shape)
    )
    return objects