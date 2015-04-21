'''
2015 Hernaldo Jesus Henriquez
'''

from random_data import RandomDataGenerator
from wtahash import WTAHash
import time
import numpy as np
#import sys

#def main(argv):
def main():
	
	k = input("Welcome, introduce the number for k = ")
	w = input("Now introduce the number for w = ")

	############################################################################
	###                        Load information matrix                       ###
	############################################################################

	n = 4096
	classifiers_count = 2786
	filename = "airplane_feats.csv"

	print ("Starting to generate hash table ...")
	start = time.time()
	wta_hash = WTAHash(filename, n, k, w)
	end = time.time()
	table_time = end - start 
	print ("Elapsed time on generation of hash table {0}".format(table_time))

	############################################################################
	###                        Use WTAHash on it                             ###
	############################################################################
	
	data = np.loadtxt(filename, delimiter=",")
	print ("Starting to generate best classifiers matrix")
	start = time.time()
	results = wta_hash.best_classifiers(data)
	end = time.time()
	find_time = end - start
	print ("Total time on best classifiers matrix {0}".format(find_time))
	
	# results_filename = "results.csv"
	# print ("Writing results in {0} ...".format(results_filename))
	# np.savetxt(results_filename, results, delimiter=",")


	####             Generate histogram for the correct ranking
	# ----------------------------------------------------------------------------

	print ("Generating histogram of correct ranking classification")
	# This is the histogram for correspondences between object and the rank given
	# to its hash. The first part is for classification when the object was ranked
	# in the first 1%, then between 1 and 5%, then between 5 and 10%, then 10 and 
	# 25%, then 25 to 50%, 50 to 75% and finally in the last 50% of the ranking.
	positions_hist = [0, 0, 0, 0, 0, 0, 0]
	separators = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]

	for row in range(len(results)):
		for col in range(len(results[row])):
			if results[row, col] == row:
				position = col
				percentage = position / len(results)
				for index in range(len(separators)):
					if percentage < separators[index]:
						positions_hist[index] += 1
					break
				break

	times_filename = "results.txt"
	print ("Writing results in {0} ...".format(times_filename))
	f = open(times_filename, "w")
	s = "Time for generating the table {0}.\n" +
			"Total time for finding best {1}.\n".format(table_time, find_time) +
			"Histogram results: {0}\n".format(positions_hist) +
			"separators percentages used: {0}.".format(separators)

	f.write(s)
	f.close()
	

if __name__ == '__main__':
	#main(sys.argv[1:])
	main()