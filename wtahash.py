import WTALibrary as wta

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
	def __init__(self, path, n, k, w):
		classifiers = self.load_classifiers(path)
		print ("Se termino de leer el archivo, creando tabla de hash")
		self.permutations = wta.CrearPermutaciones(classifiers[0], n)
		print ("Permutaciones creadas, convirtiendo en WTA")
		ConvWTAClas = wta.ConvertirenWTA(classifiers, self.permutations, n, k, w)
		self.classifiersBW = wta.GetinBinaryMayor(ConvWTAClas)
		self.whash = wta.CrearTablaHash(self.classifiersBW, ConvWTAClas)
		
		self.n = n
		self.k = k
		self.w = w
	
	# Retorna un arreglo por cada vector de imagen con todos los clasificadores
	# ordenados descendientemente según su score de matching
	def best_classifiers(self, images):
		ConvWTAImage = wta.ConvertirenWTA(
			images, self.permutations, self.n, self.k, self.w
		)
		BW1 = wta.GetinBinaryMayor(ConvWTAImage)		
		return wta.ObtenerValoresTotalesWTA(self.classifiersBW, BW1, self.whash)