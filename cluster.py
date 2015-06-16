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
    step = (len(my_range) * 5) / 100
    for i in my_range:
        index = i if set_name == "training" else i - training_count
        if step > 0 and index % step == 0:
            percentage = (index * 100) / len(my_range)
            print("Reading file {0} from {1}({2}%) ...".format(
                    index, len(my_range), percentage
                )
            )
        f = files[i]
        data = sio.loadmat(f)
        features = np.array(data["stored"], dtype=np.float32)
        # Don't use infinity in any of the 4096 feature dimensions
        for j in len(features):
            if features[j] == float("inf"):
                feautres[j] = 0
                print(
                    "Object {0} of {1} from class {2} has dim {3} inf".format(
                        i, len(files), class_name, j
                    )
                )
        if objects is None:
            objects = features
        else:
            objects = np.vstack((objects, features))
    labels = [class_name] * objects.shape[0]
    print ("Done getting the objects from {0}".format(path))
    print ("Objects matrix of shape = {0}".format(objects.shape))
    return objects, labels

def load_classes(training_percentage, path, set_name, n_classes):
    '''Loads the objects for all the classes in the MSCoco dataset.

    Args:
        training_percentage (int): Number from 0 to 100. The percentage of the
            files that will be used in the training set, the rest will be in
            the testing.
        path (string): Path to the folder containing all the classes
            (eg. /coco2014/crops/cropsFeats)
        set_name (string): It can be training or testing
        n_classes (int): Number of the dataset classes that are going to be
            used. Use 0 to load all the 72 classes. 

    Returns:
        np.array float: objects for the set
    '''

    folders = glob.glob("{0}/*".format(path))
    folders.sort()
    if n_classes > len(folders) or n_classes <= 0:
        n_classes = len(folders)
    objects = None
    labels = []
    # For each folder get the objects of that class
    # DEBUGGING ONLY USE 10 CLASSES
    # for i in range(len(folders)):
    step = (n_classes * 5) / 100
    for i in range(20, n_classes):
        percentage = (i * 100) / n_classes
        print("Reading class number {0} from {1} ({2}%) ...".format(
                i, n_classes, percentage
            )
        )
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