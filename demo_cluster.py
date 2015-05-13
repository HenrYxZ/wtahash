import wtahash as wh
import time
import numpy as np
import cluster
import cPickle as pickle
import scipy.io as sio

def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x/norm

def main():
    k = 16
    w = 2
    n = 1200
    results = ""
    # Percentage of the data that will be used for training, the rest is testing
    training_percentage = 80

    ###                  Load training information matrix                    ###
    ###----------------------------------------------------------------------###
    
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats"
    print ("Reading training instances ...")
    start = time.time()
    train_data = cluster.load_classes(training_percentage, path, "training")
    end = time.time()
    results += "Training matrix of shape {0}".format(train_data.shape) + "\n"
    s = "Elapsed time reading the training files: {0}".format(end - start)
    results += s + "\n"
    print (s)

    
    # ###                        Use WTAHash on it                             ###
    # ###----------------------------------------------------------------------###
    
    print ("Starting to generate hash table ...")
    start = time.time()
    wta_hash = wh.WTAHash(train_data, n, k, w)
    end = time.time()
    table_time = end - start
    s = "Elapsed time on generation of hash table: {0}".format(table_time)
    results += s + "\n"
    print (s)

    # Save the hash in a cPickle file
    # print ("Starting to write the hash in a file ...")
    # start = time.time()
    # pickle.dump(wta_hash, open("/user/hjhenriq/wtahash/hash.obj", "wb"))
    # end = time.time()
    # s = "Elapsed time writing the hash file: {0}".format(end - start)
    # results += s + "\n"
    # print (s)

    ###                    Load testing information matrix                   ###
    ###----------------------------------------------------------------------###

    print ("Reading testing instances ...")
    start = time.time()
    test_data = cluster.load_classes(training_percentage, path, "testing")
    end = time.time()
    results += "Testing matrix of shape {0}".format(test_data.shape) + "\n"
    s = "Elapsed time reading the testing files: {0}".format(end - start)
    results += s + "\n"
    print (s)

    ###                   Load wtahash object with pickle                    ###
    ###----------------------------------------------------------------------###

    # print ("Loading wtahash object from file ...")
    # start = time.time()
    # wta_hash = pickle.load(open("/user/hjhenriq/wtahash/hash.obj", "rb"))
    # end = time.time()
    # s = "Elapsed time loading the wtahash file: {0}".format(end - start)
    # results += s + "\n"
    # print (s)

    ###                   Get the rankings for the test set                  ###
    ###----------------------------------------------------------------------###

    print ("Generating ranking matrix for the test set ...")
    start = time.time()
    rankings = wta_hash.best_classifiers(test_data)
    end = time.time()
    s = "Elapsed time generating ranking matrix: {0}".format(end - start)
    results += s + "\n"
    print (s)

    ###                Calculate dot product on the variables                ###
    ###----------------------------------------------------------------------###

    print ("Calculating dot product on the rankings ...")
    start = time.time()
    # products is the matrix for storing the dot product for the testing
    # vectors with every 
    products = np.zeros((len(rankings), len(rankings[0])), dtype=np.float32)
    for i in range(len(test_data)):
        # y is the current testing vector
        y = normalize(test_data[i])
        for j in range(len(rankings[0])):
            # vector is the training object ranked in the current position
            vector_index = rankings[i][j]
            vector = normalize(train_data[vector_index])
            products[i][j] = np.dot(y, vector)
        percentage = (i * 100) / len(test_data)
        if percentage % 5 == 0:
            print ("Vector number {0} of {1} ({2}%) multiplied")
    end = time.time()
    s = "Elapsed time calculating dot products: {0}".format(end - start)
    results += s + "\n"
    print (s)

    # Write products in a mat file
    sio.savemat("products.mat", {"stored": products})

    # Write times in a text file
    with open("results.txt", "w") as f:
        f.write(results)

if __name__ == '__main__':
    main()
