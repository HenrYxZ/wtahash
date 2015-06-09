import utils
import scipy.io as sio

''' 
Testing precision recall function for ranking retrieval
'''

r1 = ["R", "F", "R", "F", "F", "F", "F", "F", "R", "R"]
r2 = ["F", "R", "F", "F", "R", "R", "R", "F", "F", "F"]

# precision, recall = utils.precision_recall(r1)
# interpolated_p = utils.interpolate_p(precision)
# print("precision = {0}".format(precision))
# print("recall = {0}".format(recall))
# print("interpolated_p = {0}".format(interpolated_p))

# precision, recall = utils.precision_recall(r2)
# interpolated_p = utils.interpolate_p(precision)
# print("precision = {0}".format(precision))
# print("recall = {0}".format(recall))
# print("interpolated_p = {0}".format(interpolated_p))

def test_rankings():
	rankings = sio.loadmat(open("results/rankings_2.mat", "rb"))
	print("rankings = {0}".format(rankings))