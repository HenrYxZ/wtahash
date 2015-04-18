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
	k = 15
	w = 2
	filename = "airplane_feats.csv"

	print ("Starting to generate hash table ...")
	start = time.time()
	wta_hash = WTAHash(filename, n, k, w)
	end = time.time()
	table_time = end - start 
	print ("Elapsed time on generation of hash table {0}".format(table_time))
	
	data = np.loadtxt(filename, delimiter=",")
	print ("Starting to generate best classifiers matrix")
	start = time.time()
	results = wta_hash.best_classifiers(data)
	end = time.time()
	find_time = end - start
	print ("Total time on best classifiers matrix {0}".format(find_time))
	
	results_filename = "results.csv"
	print ("Writing results in {0} ...".format(results_filename))
	np.savetxt(results_filename, results, delimiter=",")

	times_filename = "times.txt"
	print ("Writing times in {0} ...".format(times_filename))
	f = open(times_filename, "w")
	s = "Time for generating the table {0}.\n"\
			"Total time for finding best {1}.".format(table_time, find_time)
	f.write(s)
	f.close()

	############################################################################
	###                        Use WTAHash on it                             ###
	############################################################################
	

if __name__ == '__main__':
	#main(sys.argv[1:])
	main()