from random_data import RandomDataGenerator
from wtahash import WTAHash
import time
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
	filename = "../Datos/airplane_feats.csv"
	print ("Starting to generate hash table ...")
	start = time.time()
	wta_hash = WTAHash(filename, n, k, w)
	end = time.time()
	print ("Elapsed time on generation of hash table {0}", end - start)

	############################################################################
	###                        Use WTAHash on it                             ###
	############################################################################
	

if __name__ == '__main__':
	#main(sys.argv[1:])
	main()