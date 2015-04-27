from random_data import RandomDataGenerator
from wtahash import WTAHash
#import sys

#def main(argv):
def main():
	############################################################################
	###                        Create random matrix                          ###
	############################################################################

	n = 1200
	classifiers_count = 200
	min_number = 0
	max_number = 1
	generator = \
		RandomDataGenerator(classifiers_count, n, min_number, max_number)
	#filename = argv[0]
	filename = "random_data.csv"
	print("Exporting data to " + filename + " ...")
	
	generator.export_csv(filename)

	############################################################################
	###                        Use WTAHash on it                             ###
	############################################################################

	k = 15
	# se necesitan 4 bits para formar numeros del 0 al 15
	# se dividen en 2?
	w = 2
	wta_hash = WTAHash(filename, n, k, w)

if __name__ == '__main__':
	#main(sys.argv[1:])
	main()