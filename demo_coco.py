'''
2015 Hernaldo Jesus Henriquez
'''

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

	n = 1200
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

	####             Generate histogram for the correct ranking
	# ----------------------------------------------------------------------------

	print ("Generating histogram of correct ranking classification")
	# This is the histogram for correspondences between object and the rank given
	# to its hash. The first part is for classification when the object was ranked
	# first, then in the first 5%, then between 5 and 10%, then 10 and 25%, then
	# 25 to 50%, 50 to 75% and finally in the last 50% of the ranking.
	positions_hist = [0, 0, 0, 0, 0, 0, 0]
	separators = [0.05, 0.1, 0.25, 0.5, 0.75]

	for row in range(len(results)):
		for col in range(len(results[row])):
			if results[row][col] == row:
				position = col
				if position == 0:
					positions_hist[0] += 1
				else:
					percentage = position / len(results)
					for index in range(len(separators)):
						if percentage < separators[index]:
							positions_hist[index + 1] += 1
						break
				break

	times_filename = "results.txt"
	print ("Writing results in {0} ...".format(times_filename))
	f = open(times_filename, "w")
	s = "Results using k = {0} and w = {1}.\n".format(k, w)
	s += "Time for generating the table {0}.\n".format(table_time)
	s	+= "Total time for finding best {0}.\n".format(find_time)
	s += "Histogram results: {0}\n".format(positions_hist)
	s += "separators percentages used: {0}.".format(separators)
	print (s)
	f.write(s)
	f.close()
	

if __name__ == '__main__':
	#main(sys.argv[1:])
	main()