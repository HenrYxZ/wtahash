import time
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize
from datetime import datetime
import cPickle as pickle
import sys

# local modules
import wtahash as wh
import cluster
import utils

################################################################################
####                                Main                                    ####
################################################################################
def main():
    # Get the options
    s = "Enter the number of classes from the dataset to be used (0 for all).\n"
    n_classes = input(s)
    s = "Choose an option:\n"\
        "- [0] Load stored values of hash and rankings.\n"\
        "- [1] Calculate hash and rankings.\n"
    opt_load = input(s)
    s = "Choose an option:\n"\
        "- [0] Don't calculate dot products.\n"\
        "- [1] Calculate dot products.\n"
    opt_prod = input(s)
    k = 16
    w = 2
    n = 1200
    ranking_size = 2500
    # Percentage of the data that will be used for training, the rest is testing
    training_percentage = 80
    log = ""
    
    starting_time = datetime.now()
    log += "Starting time {0}\n".format(starting_time)
    log += "Using the first {0} classes of the dataset".format(n_classes)
    
    # Training
    #---------------------------------------------------------------------------
    train_data, train_labels = read_descriptors(
        training_percentage, "training", n_classes, log
    )
    if opt_load == 0:
        wta_hash = pickle.load(open(hash_filename, "rb"))
    else:
        wta_hash = create_hash(train_data, n, k, w, log)
        # store_hash(wta_hash, log)

    # Testing
    #---------------------------------------------------------------------------
    test_data, test_labels = read_descriptors(
        training_percentage, "testing", n_classes, log
    )
    if opt_load == 0:
        data = sio.loadmat(rankings_filename)
        rankings = data["stored"]
    else:
        rankings = get_rankings(test_data, wta_hash, log)
        store_rankings(rankings, n_classes, log)
        store_labels(train_labels, test_labels, n_classes, log)

    # Dot products
    #---------------------------------------------------------------------------
    if opt_prod == 1:
        dot_products(train_data, rankings, ranking_size, log)

    # Precision metrics
    #---------------------------------------------------------------------------
    # Generate relevance rankings
    calculate_metrics(rankings, train_labels, test_labels, log)
    end_time = datetime.now()
    log += "Ending time {0}\n".format(end_time)
    # Write times in a text file
    log_filename = "log_{0}.txt".format(n_classes)
    with open(log_filename, "w") as f:
        f.write(log)

################################################################################

def read_descriptors(training_percentage, set_name, n_classes, log):
    ###                  Load training information matrix                    ###
    ###----------------------------------------------------------------------###
    
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats"
    print ("Reading {0} instances ...".format(set_name))
    start = time.time()
    data, labels = cluster.load_classes(
        training_percentage, path, set_name, n_classes
    )
    end = time.time()
    log += "{0} matrix of shape {1}".format(set_name, data.shape) + "\n"
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time reading the {0} files: {1}".format(set_name, elapsed_time)
    log += s + "\n"
    print (s)

    return data, labels

def create_hash(train_data, n, k, w, log):        
    ###                        Use WTAHash on it                             ###
    ###----------------------------------------------------------------------###
    
    print ("Starting to generate hash table ...")
    start = time.time()
    wta_hash = wh.WTAHash(train_data, n, k, w)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time on generation of hash table: {0}".format(elapsed_time)
    log += s + "\n"
    print (s)

    return wta_hash

def store_hash(wta_hash, n_classes, log):
    ## Store the hash in a binary file
    print("Storing the hash in a file ...")
    start = time.time()
    hash_filename = "wtahash_{0}.obj".format(n_classes)
    pickle.dump(wta_hash, open(hash_filename, "wb"), protocol=2)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time storing the hash {0}".format(elapsed_time)
    log += s + "\n"
    print(s)

def get_rankings(test_data, wta_hash, log):
    ###                   Get the rankings for the test set                  ###
    ###----------------------------------------------------------------------###

    print ("Generating ranking matrix for the test set ...")
    start = time.time()
    rankings = wta_hash.best_classifiers(test_data)
    rankings = np.array(rankings, dtype=np.uint32)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time generating ranking matrix: {0}".format(elapsed_time)
    log += s + "\n"
    print (s)

    return rankings

def store_rankings(rankings, n_classes, log):
    ## Store the rankings in a csv file
    print("Storing rankings in a mat file ...")
    start = time.time()
    rankings_filename = "rankings_{0}.mat".format(n_classes)
    data = {"stored": rankings}
    sio.savemat(rankings_filename, data, do_compression=True)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time storing the rankings {0}".format(elapsed_time)
    log += s + "\n"
    print(s)

def store_labels(train_labels, test_labels, n_classes, log):
    ## Store the labels in a text file
    print("Storing the labels in text files...")
    start = time.time()
    train_labels_fn = "train_labels_{0}.txt".format(n_classes)
    test_labels_fn = "test_labels{0}.txt".format(n_classes)
    utils.write_list(train_labels, train_labels_fn)
    utils.write_list(test_labels, test_labels_fn)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time storing the labels {0}".format(elapsed_time)
    log += s + "\n"
    print(s)

def dot_products(train_data, rankings, ranking_size, log):
    ###                Calculate dot product on the variables                ###
    ###----------------------------------------------------------------------###

    print ("Calculating dot product on the rankings ...")
    start = time.time()
    # products is the matrix that stores the dot product of each testing vector
    # with each training vector
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
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time calculating dot products: {0}".format(elapsed_time)
    log += s + "\n"
    print (s)

    # Write products in a mat file
    sio.savemat("products.mat", {"stored": products}, do_compression=True)

def calculate_metrics(rankings, train_labels, test_labels, log):
    ###           Calculates mAP and 5 random precision queries              ###
    ###----------------------------------------------------------------------###
    
    print("Starting to calculate metrics ...")
    start = time.time()
    rel_ranks = []
    for i in range(len(rankings)):
        rel_ranks.append(
            utils.relevance_ranking(
                rankings[i], train_labels, test_labels[i]
            )
        )
    # Take 5 random queries
    # n_queries = 5
    # sample_indices = np.random.choice(len(rankings), n_queries, replace=False)
    # sample = [rel_ranks[index] for index in sample_indices]
    # # Get precision-recall for each query
    # queries = []
    # for ranking in sample:
    #     precisions = utils.precision_fixed_recall(ranking)
    #     queries.append(utils.interpolate_p(precisions))
    # utils.write_list(queries, "queries.txt")
    # Get average precisions
    avg_precs = [utils.average_precision(rel_rank) for rel_rank in rel_ranks]
    utils.write_list(avg_precs, "avg_precs.txt")
    mean_avg_prec = np.mean(avg_precs)
    s = "mean average precision = {0}".format(mean_avg_prec)
    log += s + "\n"
    print(s)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    s = "Elapsed time calculating metrics: {0}".format(elapsed_time)
    log += s + "\n"
    print (s)

if __name__ == '__main__':
    main()
