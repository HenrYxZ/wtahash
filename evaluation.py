import time
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize
from datetime import datetime
import cPickle as pickle

# local modules
import wtahash as wh
import cluster
import utils

class Evaluation:
    ''' Class for evaluating the speed and storing the rankings of a dataset
        using WTAHash.
    '''
    def __init__(self, dataset_path):
        self.dataset = dataset_path
        self.log = ""

    def run(self):
        # Get the options
        s = "Enter the number of classes from the dataset to be used "\
            "(0 for all).\n"
        self.n_classes = input(s)
        s = "Choose an option:\n"\
            "- [0] Calculate hash and rankings.\n"\
            "- [1] Load stored values of hash and rankings.\n"
        opt_load = input(s)
        s = "Choose an option:\n"\
            "- [0] Calculate dot products.\n"\
            "- [1] Don't calculate dot products.\n"
            
        opt_prod = input(s)
        k = 16
        w = 2
        n = 1200
        ranking_size = 2500
        # Percentage of the data that will be used for training, the rest is 
        # testing
        train_perc = 80
        self.log = ""
        
        starting_time = datetime.now()
        self.log += "Starting time {0}\n".format(starting_time)
        self.log += "Using the first {0} classes of the dataset\n".format(
            self.n_classes
        )
        
        # Training
        #-----------------------------------------------------------------------
        train_data, train_labels = self.read_descriptors(train_perc, "training")
        if opt_load == 1:
            hash_filename = "results/wtahash_{0}.obj".format(self.n_classes)
            wta_hash = pickle.load(open(hash_filename, "rb"))
        else:
            wta_hash = self.create_hash(train_data, n, k, w)
            # store_hash(wta_hash)

        # Testing
        #-----------------------------------------------------------------------
        test_data, test_labels = self.read_descriptors(train_perc, "testing")
        if opt_load == 1:
            rankings_filename = "results/rankings_{0}.mat".format(
                self.n_classes
            )
            data = sio.loadmat(rankings_filename)
            rankings = data["stored"]
        else:
            rankings = self.get_rankings(test_data, wta_hash)
            self.store_rankings(rankings)
            self.store_labels(train_labels, test_labels)

        # Dot products
        #-----------------------------------------------------------------------
        if opt_prod == 0:
            prods = self.dot_products(
                train_data, test_data, rankings, ranking_size
            )
            self.store_products(prods)

        # Precision metrics
        #-----------------------------------------------------------------------
        # Generate relevance rankings
        self.calculate_metrics(rankings, train_labels, test_labels)
        end_time = datetime.now()
        self.log += "Ending time {0}\n".format(end_time)
        # Write times in a text file
        log_filename = "results/log_{0}.txt".format(self.n_classes)
        with open(log_filename, "w") as f:
            f.write(self.log)

    def read_descriptors(self, training_percentage, set_name):
        ###                  Load training information matrix                ###
        ###------------------------------------------------------------------###
        
        print ("Reading {0} instances ...".format(set_name))
        start = time.time()
        data, labels = cluster.load_classes(
            training_percentage, self.dataset, set_name, self.n_classes
        )
        end = time.time()
        self.log += "{0} matrix of shape {1}\n".format(set_name, data.shape)
        elapsed_time = utils.humanize_time(end - start)
        s = "Elapsed time reading the {0} files: {1}".format(
            set_name, elapsed_time
        )
        self.log += s + "\n"
        print (s)

        return data, labels

    def create_hash(self, train_data, n, k, w):        
        ###                        Use WTAHash on it                         ###
        ###------------------------------------------------------------------###
        
        print ("Starting to generate hash table ...")
        start = time.time()
        wta_hash = wh.WTAHash(train_data, n, k, w)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        s = "Elapsed time on generation of hash table: {0}".format(elapsed_time)
        self.log += s + "\n"
        print (s)

        return wta_hash

    def get_rankings(self, test_data, wta_hash):
        ###                   Get the rankings for the test set              ###
        ###------------------------------------------------------------------###

        print ("Generating ranking matrix for the test set ...")
        start = time.time()
        rankings = wta_hash.best_classifiers(test_data)
        rankings = np.array(rankings, dtype=np.uint32)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        s = "Elapsed time generating ranking matrix: {0}".format(elapsed_time)
        self.log += s + "\n"
        print (s)

        return rankings

    def dot_products(self, train_data, test_data, rankings, ranking_size):
        ###                Calculate dot product on the variables            ###
        ###------------------------------------------------------------------###

        print ("Calculating dot product on the rankings ...")
        start = time.time()
        # products is the matrix that stores the dot product of each testing 
        # vector with each training vector
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
        self.log += s + "\n"
        print (s)
        return products

    def calculate_metrics(self, rankings, train_labels, test_labels):
        ###           Calculates mAP and 5 random precision queries          ###
        ###------------------------------------------------------------------###
        
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
        avg_precs = [utils.average_precision(rel_rk) for rel_rk in rel_ranks]
        utils.write_list(avg_precs, "avg_precs.txt")
        mean_avg_prec = np.mean(avg_precs)
        s = "mean average precision = {0}".format(mean_avg_prec)
        self.log += s + "\n"
        print(s)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        s = "Elapsed time calculating metrics: {0}".format(elapsed_time)
        self.log += s + "\n"
        print (s)

    def store_hash(self, wta_hash):
        ## Store the hash in a binary file
        print("Storing the hash in a file ...")
        start = time.time()
        hash_filename = "results/wtahash_{0}.obj".format(self.n_classes)
        pickle.dump(wta_hash, open(hash_filename, "wb"), protocol=2)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        s = "Elapsed time storing the hash {0}".format(elapsed_time)
        self.log += s + "\n"
        print(s)

    def store_rankings(self, rankings):
        ## Store the rankings in a csv file
        print("Storing rankings in a mat file ...")
        start = time.time()
        rankings_filename = "results/rankings_{0}.mat".format(self.n_classes)
        data = {"stored": rankings}
        sio.savemat(rankings_filename, data, do_compression=True)
        end = time.time()
        s = "Elapsed time storing the rankings {0} secs.".format(end - start)
        self.log += s + "\n"
        print(s)

    def store_labels(self, train_labels, test_labels):
        ## Store the labels in a text file
        print("Storing the labels in text files...")
        start = time.time()
        train_labels_fn = "results/train_labels_{0}.txt".format(self.n_classes)
        test_labels_fn = "results/test_labels_{0}.txt".format(self.n_classes)
        utils.write_list(train_labels, train_labels_fn)
        utils.write_list(test_labels, test_labels_fn)
        end = time.time()
        s = "Elapsed time storing the labels {0} secs.".format(end - start)
        self.log += s + "\n"
        print(s)

    def store_products(self, products):
        # Write products in a mat file
        print("Storing products in a mat file ...")
        start = time.time()
        prods_filename = "results/products_{0}.mat".format(self.n_classes)
        sio.savemat(prods_filename, {"stored": products}, do_compression=True)
        end = time.time()
        print("Elapsed time storing the products {0} secs.".format(end - start))
        
        