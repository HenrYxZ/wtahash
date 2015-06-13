import time
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize
from datetime import datetime
import cPickle as pickle
from heapq import heappush, heappop

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
        ranking_size = 100
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
            self.store_rankings(rankings, ranking_size)
            self.store_labels(train_labels, test_labels)

        # Dot products
        #-----------------------------------------------------------------------
        if opt_prod == 0:
            prods = self.dot_products(
                train_data, test_data, rankings, ranking_size
            )
            self.store_products(prods)
        else:
            prods_filename = "results/products_{0}.mat"
            prods = sio.loadmat(prods_filename)

        # Precision metrics
        #-----------------------------------------------------------------------
        # Generate relevance rankings
        self.calculate_metrics(rankings, train_labels, test_labels, prods)
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
                int: Index of the object in the train set that should be ranked
                    in the i-th position where i is the number of the row,
                float: The value of the dot product between the object in the
                    train set and the object in the test set in the i-th 
                    position where i is the number of the row. 
            }
        '''
        ###                Calculate dot product on the variables            ###
        ###------------------------------------------------------------------###

        print ("Calculating dot products on the rankings ...")
        start = time.time()
        # products is the matrix that stores the dot product of each testing 
        # vector with each training vector
        step = (len(test_data) * 5) / 100
        train_norm = [utils.norm(train_vec) for train_vec in train_data]
        train_norm = np.array(train_norm)
        for i in range(len(test_data)):
            # y is the current testing vector
            y = test_data[i]
            y_norm = normalize(y[:, np.newaxis], axis=0).ravel()
            current_products = []
            for j in range(len(train_data)):
                # vector is the training object ranked in the current position
                vector_index = rankings[i][j]
                vector_norm = train_norm[j]
                current_products.append((j, np.dot(y_norm, vector_norm)))
            current_products.sort(key=lambda x: x[1], reverse=True)
            products.append(current_products[:ranking_size])
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

    def calculate_metrics(self, rankings, train_labels, test_labels, products):
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

        # Classification mAP
        #-----------------------------------------------------------------------
        class_ap = [utils.class_ap(rel_rk) for rel_rk in rel_ranks]
        class_ap_filename = "results/class_avg_precs_{0}.txt".format(
            self.n_classes
        )
        utils.write_list(class_ap, class_ap_filename)
        class_map = np.mean(class_ap)
        s = "classification mean average precision = {0}".format(class_map)
        self.log += s + "\n"
        print(s)

        # Dot products average precision
        #-----------------------------------------------------------------------
        # Set
        set_prec = []
        for i in range(len(rankings)):
            indices = [prods[0] for prods in products[i]]
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
            indices = [prods[0] for prods in products[i]]
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

    def store_products(self, products):
        # Write products in a mat file
        print("Storing products in a mat file ...")
        start = time.time()
        prods_filename = "results/products_{0}.mat".format(self.n_classes)
        sio.savemat(prods_filename, {"stored": products})
        end = time.time()
        print("Elapsed time storing the products {0} secs.".format(end - start))
        
        
