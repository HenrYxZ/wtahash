import WTALibrary as wta
import cluster
import time
import utils

class WTAHash:
	#Método interno, se usa para cargar los clasificadores
	def load_classifiers(self, path):
		f = open(path, "r")
		classifier = []
		classifiers = []
		line = f.readline()
		while line:
			splitted = line.split(',')
			for i in range(len(splitted)):
				weight = float(splitted[i])
				classifier.append(weight)
			classifiers.append(classifier)
			classifier = []
			line = f.readline()
		return classifiers

	# Crea la estructura WTAHash, recibe como parámetro el path del archivo con
	# los clasificadores y los valores de n, k y w de WTA
	def __init__(self, objects, n, k, w):
		# classifiers = self.load_classifiers(path)
		self.permutations = wta.CrearPermutaciones(objects[0], n)
		print("Permutations ready, converting to WTA")
		ConvWTAClas = wta.ConvertirenWTA(objects, self.permutations, n, k, w)
		print("Objects converted to WTA, passing from binary to int")
		self.classifiersBW = wta.GetinBinaryMayor(ConvWTAClas)
		print("Convertion ready, now creating the hash table")
		self.whash = wta.CrearTablaHash(self.classifiersBW, ConvWTAClas)
		
		self.n = n
		self.k = k
		self.w = w
	
	# Retorna un arreglo por cada vector de imagen con todos los clasificadores
	# ordenados descendientemente según su score de matching
	def best_classifiers(self, images, ranking_size):
		start = time.time()
		ConvWTAImage = wta.ConvertirenWTA(
			images, self.permutations, self.n, self.k, self.w
		)
		end = time.time()
		elapsed_time = utils.humanize_time(end - start)
		print("Elapsed time converting test set to WTA binary {0}.".format(
				elapsed_time
			)
		)
		BW1 = wta.GetinBinaryMayor(ConvWTAImage)		
		values = wta.ObtenerValoresTotalesWTA(
			self.classifiersBW, BW1, self.whash, ranking_size
		)
		return values