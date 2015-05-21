import wtahash as wh
import time
import numpy as np
import cluster
import scipy.io as sio
from sklearn.preprocessing import normalize
from datetime import datetime

def main():
    k = 16
    w = 2
    n = 1200
    ranking_size = 2500
    log = ""
    # Percentage of the data that will be used for training, the rest is testing
    training_percentage = 80

    train_data, wta_hash = train(training_percentage, n, k, w, log)
    test_data, rankings = test(training_percentage, wta_hash, log)

    starting_time = datetime.now()
    log += "Starting time {0}\n".format(starting_time)
    ###                Calculate dot product on the variables                ###
    ###----------------------------------------------------------------------###

    print ("Calculating dot product on the rankings ...")
    start = time.time()
    # products is the matrix for storing the dot product for the testing
    # vectors with every 
    products = np.zeros((len(rankings), ranking_size), dtype=np.float32)
    step = (len(test_data) * 5) / 100
    for i in range(len(test_data)):
        # y is the current testing vector
        y = test_data[i]
        y_norm = normalize(y[:, np.newaxis], axis=0).ravel()
        for j in range(ranking_size):
            # vector is the training object ranked in the current position
            vector_index = rankings[i][j]
            vector = train_data[vector_index]
            vector_norm = normalize(vector[:, np.newaxis], axis=0).ravel()
            products[i][j] = np.dot(y_norm, vector_norm)
        if i % step == 0:
            percentage = (i * 100) / len(test_data)
            print (
                "Vector number {0} of {1} ({2}%) multiplied".format(
                    i, len(test_data), percentage
                )
            )
    end = time.time()
    s = "Elapsed time calculating dot products: {0}".format(end - start)
    log += s + "\n"
    print (s)

    # Write products in a mat file
    sio.savemat("products.mat", {"stored": products})

    end_time = datetime.now()
    log += "Ending time {0}\n".format(end_time)
    # Write times in a text file
    with open("log.txt", "w") as f:
        f.write(log)

def train(training_percentage, n, k, w, log):

    ###                  Load training information matrix                    ###
    ###----------------------------------------------------------------------###
    
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats"
    print ("Reading training instances ...")
    start = time.time()
    train_data = cluster.load_classes(training_percentage, path, "training")
    end = time.time()
    log += "Training matrix of shape {0}".format(train_data.shape) + "\n"
    s = "Elapsed time reading the training files: {0}".format(end - start)
    log += s + "\n"
    print (s)
    
    # ###                        Use WTAHash on it                             ###
    # ###----------------------------------------------------------------------###
    
    print ("Starting to generate hash table ...")
    start = time.time()
    wta_hash = wh.WTAHash(train_data, n, k, w)
    end = time.time()
    table_time = end - start
    s = "Elapsed time on generation of hash table: {0}".format(table_time)
    log += s + "\n"
    print (s)

    return train_data, wta_hash

def test(training_percentage, wta_hash, log):

    ###                    Load testing information matrix                   ###
    ###----------------------------------------------------------------------###

    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats"
    print ("Reading testing instances ...")
    start = time.time()
    test_data = cluster.load_classes(training_percentage, path, "testing")
    end = time.time()
    log += "Testing matrix of shape {0}".format(test_data.shape) + "\n"
    s = "Elapsed time reading the testing files: {0}".format(end - start)
    log += s + "\n"
    print (s)

    ###                   Get the rankings for the test set                  ###
    ###----------------------------------------------------------------------###

    print ("Generating ranking matrix for the test set ...")
    start = time.time()
    rankings = wta_hash.best_classifiers(test_data)
    end = time.time()
    s = "Elapsed time generating ranking matrix: {0}".format(end - start)
    log += s + "\n"
    print (s)

    return test_data, rankings

if __name__ == '__main__':
    main()
