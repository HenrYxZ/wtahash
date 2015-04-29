import cluster
import numpy as np
from wtahash import WTAHash
import time

def test(objects, train, wta_hash):
    best = wta_hash.best_classifiers(objects)
    products = np.zeros((len(objects), len(train)))
    for i in range(len(objects)):
        for j in range(len(train)):
            # Index of the vector, at this position, in the train matrix
            train_index = best[i][j]
            products[i][j] = np.dot(objects[i], train[train_index])
    np.savetxt("products.csv", products, fmt = "%.6f", delimiter = ",")

if __name__ == "__main__":
    k = 16
    w = 2
    n = 1200
    training_percentage = 80
    path = "/home/jesus/projects/wtahash/airplane"
    train = cluster.load_class(training_percentage, path, "training")
    start = time.time()
    wta_hash = WTAHash(train, n, k, w)
    end = time.time()
    print("Time creating wta hash {0}".format(end - start))

    start = time.time()
    test_objs = cluster.load_class(training_percentage, path, "test")
    end = time.time()
    print ("Time getting testing objects was {0}".format(end - start))

    start = time.time()
    test(test_objs, train, wta_hash)
    end = time.time()
    print ("Time testing was {0}".format(end - start))