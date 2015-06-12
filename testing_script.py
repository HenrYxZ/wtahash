import utils
import scipy.io as sio

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

if __name__ == '__main__':
	test_rankings()