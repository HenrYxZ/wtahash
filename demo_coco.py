from random_data import RandomDataGenerator
from wtahash import WTAHash
import time
import numpy as np
#import sys

#def main(argv):
def main():
	############################################################################
	###                        Load information matrix                       ###
	############################################################################

	n = 4096
	classifiers_count = 2786
	k = 16
	w = 2
	filename = "airplane_feats.csv"

	print ("Starting to generate hash table ...")
	start = time.time()
	wta_hash = WTAHash(filename, n, k, w)
	end = time.time()
	print ("Elapsed time on generation of hash table {0}".format(end - start))
	
	data = np.loadtxt(filename, delimiter=",")
	print ("Starting to generate best classifiers matrix")
	start = time.time()
	results = wta_hash.best_classifiers(data)
	end = time.time()
	print ("Total time on best classifiers matrix {0}".format(end - start))
	
	results_filename = "results.csv"
	print ("Writing results in {0}".format(results_filename))
	np.savetxt(results_filename, results, delimiter=",")

	############################################################################
	###                        Use WTAHash on it                             ###
	############################################################################
	

if __name__ == '__main__':
	#main(sys.argv[1:])
	main()