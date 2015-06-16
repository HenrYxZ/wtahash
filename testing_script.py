import scipy.io as sio
import numpy as np

# Local modules
import utils
import cluster

''' 
Testing precision recall function for ranking retrieval
'''
def test_prec_recall():
	r1 = ["R", "F", "R", "F", "F", "F", "F", "F", "R", "R"]
	r2 = ["F", "R", "F", "F", "R", "R", "R", "F", "F", "F"]

	precision, recall = utils.precision_recall(r1)
	interpolated_p = utils.interpolate_p(precision)
	print("precision = {0}".format(precision))
	print("recall = {0}".format(recall))
	print("interpolated_p = {0}".format(interpolated_p))

	precision, recall = utils.precision_recall(r2)
	interpolated_p = utils.interpolate_p(precision)
	print("precision = {0}".format(precision))
	print("recall = {0}".format(recall))
	print("interpolated_p = {0}".format(interpolated_p))

def test_rankings():
	data = sio.loadmat(open("results/rankings_2.mat", "rb"))
	print("data = {0}".format(data))
	rankings = data["stored"]
	print("rankings shape = {0}".format(rankings.shape))
	print("rankings[:10][:5] = {0}".format(rankings[:10][:5]))

def test_dot_prod_prec():
    # values = np.random.random_sample(10) * 100
    # prods = [(i, values[i]) for i in range(len(values))]
    # print("prods =\n{0}".format(prods))
    # prods.sort(key=lambda x: x[1], reverse=True)
    # print("ordered prods=\n{0}".format(prods))
    ordered_prods = [(1, 73.50),
                     (7, 62.47),
                     (3, 57.34),
                     (2, 42.22), 
                     (6, 40.87),
                     (5, 25.02), 
                     (0, 20.33), 
                     (8, 15.99), 
                     (9, 14.74), 
                     (4, 7.89)]
    ranking = np.array([1,
                        28, 
                        3, 
                        50, 
                        60, 
                        4, 
                        5, 
                        8, 
                        0, 
                        10])
    ids = [prod[0] for prod in ordered_prods]
    print("ids = {0}".format(ids))
    # prod pos prec 0.3
    pos_prec = utils.prod_pos_prec(ids, ranking)
    print("prod pos prec = {0}".format(pos_prec))
    # prod set prec 0.6
    set_prec = utils.prod_set_prec(ids, ranking)
    print("prod pos prec = {0}".format(set_prec))

def test_store_prods():
    ordered_prods = [(1, 73.50),
                     (7, 62.47),
                     (3, 57.34),
                     (2, 42.22), 
                     (6, 40.87),
                     (5, 25.02), 
                     (0, 20.33), 
                     (8, 15.99), 
                     (9, 14.74), 
                     (4, 7.89)]
    ordered_prods_sec = [(4, 91.05),
                         (8, 74.22), 
                         (2, 71.66), 
                         (1, 68.86),
                         (7, 42.86), 
                         (0, 40.09), 
                         (3, 35.71), 
                         (9, 32.09), 
                         (6, 25.62), 
                         (5, 23.52)]
    sorted_prods = [ordered_prods, ordered_prods_sec]
    return sorted_prods
    # print("sorted_prods = \n{0}".format(sorted_prods))
    # indices = []
    # for i in len(sorted_prods):
    #     indices.append([prod[0] for prod in sorted_prods[i]])
    # print("indices = \n{0}".format(indices))

def test_inf_prod():
    train_perc = 80
    n_clases = 0
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats"
    train_data = cluster.load_classes(train_perc, path, "training", n_clases)
    train_norm = [utils.normalize(train_vec) for train_vec in train_data]


if __name__ == '__main__':
    start = time.time()
	test_inf_prod()
    end = time.time()
    print("Elapsed time testing {0} seconds...".format(end - start))