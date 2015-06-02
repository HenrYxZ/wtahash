import scipy.io as sio
import glob
import numpy as np


def load_class(training_percentage, path, set_name):
    '''Loads the objects for a class in the MSCoco dataset.

    Args:
        training_percentage (int): Number from 0 to 100. The percentage of the files
            that will be used in the training set, the rest will be in the testing.
        path (string): Path to the object files 
            (e.g. "/coco2014/crops/cropsFeats/airplane")
        set_name (string): It can be training or testing

    Returns:
        {
            np.array float: objects for the set,
            string: name of the class
        }
    '''
    files = glob.glob("{0}/*.mat".format(path))
    class_name = path.split("/")[-1]
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
        features = np.array(data["stored"], dtype=np.float32)
        if objects is None:
            objects = features
        else:
            objects = np.vstack((objects, features))
    labels = [class_name] * objects.shape[0]
    print ("Done getting the objects from {0}".format(path))
    print ("Objects matrix of shape = {0}".format(objects.shape))
    return objects, labels

def load_classes(training_percentage, path, set_name):
    '''Loads the objects for all the classes in the MSCoco dataset.

    Args:
        training_percentage (int): Number from 0 to 100. The percentage of the files
            that will be used in the training set, the rest will be in the testing.
        path (string): Path to the folder containing all the classes
            (eg. /coco2014/crops/cropsFeats)
        set_name (string): It can be training or testing

    Returns:
        np.array float: objects for the set
    '''
    folders = glob.glob("{0}/*".format(path))
    folders.sort()
    objects = None
    labels = []
    # For each folder get the objects of that class
    # DEBUGGING ONLY USE 10 CLASSES
    # for i in range(len(folders)):
    for i in range(10):
        full_path = folders[i]
        this_class, this_labels = load_class(
            training_percentage, full_path, set_name
        )
        labels += this_labels
        if objects is None:
            objects = this_class
        else:
            objects = np.vstack((objects, this_class))
    print (
        "Done getting the objects from all the classes, final matrix of shape "\
        "= {0}".format(objects.shape)
    )
    return objects, labels