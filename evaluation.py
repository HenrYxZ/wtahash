import time
import numpy as np
import scipy.io as sio
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
            "- [0] Calculate WTA Hash.\n"\
            "- [1] Load a stored WTA Hash.\n"\
            "- [2] Calculate WTA Hash and store it.\n"
        opt_load = input(s)
        k = 16
        w = 2
        n = 1200

        LOAD_HASH = 1
        STORE_HASH = 2

        s = "Enter the ranking size you want to use.\n"
        ranking_size = input(s)
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
        if opt_load == LOAD_HASH:
            hash_filename = "results/wtahash_{0}.obj".format(self.n_classes)
            wta_hash = pickle.load(open(hash_filename, "rb"))
        else:
            wta_hash = self.create_hash(train_data, n, k, w)
            if opt_load == STORE_HASH:
                store_hash(wta_hash)

        # Testing
        #-----------------------------------------------------------------------
        test_data, test_labels = self.read_descriptors(train_perc, "testing")
        rankings = self.get_rankings(test_data, wta_hash)
        if ranking_size > len(rankings[0]) or ranking_size < 0:
            ranking_size = 100
        self.store_rankings(rankings, ranking_size)
        self.store_labels(train_labels, test_labels)

        # Dot products
        #-----------------------------------------------------------------------
        sorted_prods, prods = self.dot_products(
            train_data, test_data, rankings, ranking_size
        )
        self.store_products(sorted_prods, prods)

        # Precision metrics
        #-----------------------------------------------------------------------
        # Generate relevance rankings
        self.metrics(rankings, train_labels, test_labels, sorted_prods)
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
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        s = "Elapsed time generating ranking matrix: {0}".format(elapsed_time)
        self.log += s + "\n"
        print (s)

        return rankings

    def dot_products(self, train_data, test_data, rankings, ranking_size):
        ''' Calculates the dot product for each element in the test set with
            every element of the train set. Returns a matrix with two columns
            matrix. The first column is the index of the object in the train set
            and the second column is the value of the dot product of that object
            with the test object with index equal to the number of the row. Then
            the number of rows is the number of objects in the test set.

        Args:
            train_data (np matrix of floats): Each row is the vector of an
                object in the train set.
            test_data (np matrix of floats): Each row is the vector of an object
                in the test set.
            rankings (list of lists int): The ranking created for each object
                in the test set.
            ranking_size (int): The number of relevant elements in the ranking.

        Returns:
            list of list of tuples { 
                e.g.:
                         0             ranking_size
                         |                  |
                0 - [[(21, 0.91), (3, 0.87), ...],
                     [(10, 0.83), (0, 0.72), ...],
                             ...
   len(test_data) -  [                      ... ]]

                int: Index of the object in the train set that should be ranked
                    in the i-th position where i is the number of the row,
                float: The value of the dot product between the object in the
                    train set and the object in the test set in the i-th 
                    position where i is the number of the row. 
            },
            numpy array of arrays of floats: Dot products where the [i-th, j-th]
                element is the product between the i-th object of the testing
                set and the j-th object of the training set. 
        '''
        ###                Calculate dot product on the variables            ###
        ###------------------------------------------------------------------###

        print ("Calculating dot products on the rankings ...")
        start = time.time()
        # products is the matrix that stores the dot product of each testing 
        # vector with each training vector
        sorted_prods = []
        products = []
        step = (len(test_data) * 5) / 100
        train_norm = [utils.normalize(train_vec) for train_vec in train_data]
        train_norm = np.array(train_norm)
        for i in range(len(test_data)):
            # y is the current testing vector
            y = test_data[i]
            y_norm = utils.normalize(y)
            current_tuples = []
            products.append([])
            for j in range(len(train_data)):
                # vector is the training object ranked in the current position
                vector_index = rankings[i][j]
                vector_norm = train_norm[j]
                prod = np.dot(y_norm, vector_norm)
                if j < ranking_size:
                    products[i].append(prod)
                current_tuples.append( (j, prod) )
            current_tuples.sort(key=lambda x: x[1], reverse=True)
            sorted_prods.append(current_tuples[:ranking_size])
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
        return sorted_prods, np.array(products)

    def metrics(self, rankings, train_labels, test_labels, sorted_prods):
        ###    Calculates classification and products set and position mAP   ###
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

        # Classification mAP
        #-----------------------------------------------------------------------
        class_ap = [utils.class_ap(rel_rk) for rel_rk in rel_ranks]
        class_ap_filename = "results/class_avg_precs_{0}.txt".format(
            self.n_classes
        )
        utils.write_list(class_ap, class_ap_filename)
        class_map = np.mean(class_ap)
        self.log += "ranking size = {0}".format(ranking_size) + "\n"
        s = "classification mean average precision = {0}".format(class_map)
        self.log += s + "\n"
        print(s)

        # Dot products average precision
        #-----------------------------------------------------------------------
        # Set
        set_prec = []
        for i in range(len(rankings)):
            indices = [prods[0] for prods in sorted_prods[i]]
            precision = utils.prod_set_prec(indices, rankings[i])
            set_prec.append(precision)
        set_ap_filename = "results/set_avg_precs_{0}.txt".format(
            self.n_classes
        )
        utils.write_list(set_prec, set_ap_filename)
        set_map = np.mean(set_prec)
        s = "set mean average precision = {0}".format(set_map)
        self.log += s + "\n"
        print(s)

        # Position
        pos_prec = []
        for i in range(len(rankings)):
            indices = [prods[0] for prods in sorted_prods[i]]
            precision = utils.prod_pos_prec(indices, rankings[i])
            pos_prec.append(precision)
        pos_ap_filename = "results/pos_avg_precs_{0}.txt".format(
            self.n_classes
        )
        utils.write_list(pos_prec, pos_ap_filename)
        pos_map = np.mean(pos_prec)
        s = "position mean average precision = {0}".format(pos_map)
        self.log += s + "\n"
        print(s)


        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        s = "Elapsed time calculating metrics: {0}".format(elapsed_time)
        self.log += s + "\n"
        print (s)
    ############################################################################
    ####                    Functions for storing values                    ####
    ############################################################################
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

    def store_rankings(self, rankings, ranking_size):
        ## Store the rankings in a csv file
        print("Storing rankings in a mat file ...")
        start = time.time()
        rankings_filename = "results/rankings_{0}.mat".format(self.n_classes)
        data = {"stored": rankings}
        sio.savemat(rankings_filename, data)
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

    def store_products(self, sorted_prods, products):
        # Write products in a mat file
        print("Storing products in a mat file ...")
        start = time.time()
        prods_filename = "results/products_{0}.mat".format(self.n_classes)
        sio.savemat(prods_filename, {"stored": products})
        # e.g. elem = [(1, 0.94), (12, 0.83), (4, 0.6), ...]
        #   indices = [1, 12, 4, ...]
        indices = [elem[:, 0] for elem in sorted_prods]
        ids_filename = "results/indices_{0}.mat".format(self.n_classes)
        sio.savemat(ids_filename, {"stored": indices})
        end = time.time()
        print("Elapsed time storing the products {0} secs.".format(end - start))
        
        
