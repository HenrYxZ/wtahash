"""
by: Cristobal Guell
"""
import random
import math
import time
import numpy
# import wtahash
import cPickle as pickle

def CrearPermutaciones(vector, n):
	numero = 0
	base = []
	for numero in range(n):
		# Crea los arreglos para guardar los arreglos permutacion dependiendo
		# del N
		# Crea las permutaciones
		b = random.sample(range(len(vector)), len(vector))
		#print b
		#print "permutacion"
		base.append(b)
	return base

# convierte el vector en WTA
def ConvertirenWTA(listaVectores, Permutaciones, n, k, w):
	arrFinal = []
	Waux = n / w
	cant_vectores = len(listaVectores)
	for i in range(cant_vectores):
		# convwta=wtahash.wtahash(listaVectores[i],15,600)
		# entregar K, N y W. Ojo que el W esta un poco distinto al paper pero
		# hace lo mismo si W=3 lo divide en 3 , no en partes de 3 bits solo hay
		# que calcular cuantas partes quedan y poner eso como W
		convwta = wtahash(listaVectores[i], k, n, Waux, Permutaciones) 
		arrFinal.append(convwta)
		step = (cant_vectores * 5) /100
		if i % step == 0:
			print (
				"Vector numero {0} de {1} convertido a WTA".format(i, cant_vectores)
			)
	return arrFinal

def wtahash(vector, k, n, w, permutaciones):
	base = permutaciones
	answers = []
	maximales = []
	binario = []
	binariofinal = []

	for i in range(n):
		answers.append([])
		maximales.append([0])
	indice_del_mayor = 0

	for i in range(n):
		# Hace la transformacion
		for conteo in range(k):
		    answers[i].append(vector[base[i][conteo]])
		    if answers[i][conteo] > maximales[i][0]:
		    	# maximales solo seria de la forma [... ,[0], ...] ->
		    	#  0   i   n
		    	# [..., [12], ...] -> [..., [20], ...]
		    	maximales[i][0] = answers[i][conteo]
		    	indice_del_mayor = conteo

		L = map(int, bin(indice_del_mayor)[2:])
		L = deterLargo(k, L)
		#Se convierte en binario y luego se da vuelta por el left-bitmost
		binario.append(L[::-1])
		# Concatenacion de arreglos
		binariofinal = binariofinal + binario[i]
		indice_del_mayor = 0
	#          0                           n
	# w = 4, [00110010, 10010001, ..., 01101011]
	#          0                                            w x n
	# -> [array[0011], array[0010], array[1001], array[0001], ...]

	# OJO para w = 2 y n = 4096 el largo de auxsplit es 2048
	auxsplit = numpy.array_split(binariofinal, w)
	list1 = []

	for auxlist in range(len(auxsplit)):
		list1.append(auxsplit[auxlist].tolist())
	# -> list1 = [[0, 0, 1, 1], [0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, 1],  ...]
	return list1

# hace que los numeros de indice tengan el mismo largo, es decir si existe hasta
# el indice 2 osea 10 en bianrio si sale 0 en vez de entregar 0 se entrega 00

def deterLargo(long1, vecbinario): 
	L = map(int, bin(long1 - 1)[2:])
	length = len(L)
	largo_vecbinario = len(vecbinario)
	if(length == largo_vecbinario):
		return vecbinario
	else:
		diferencia_largos = length - largo_vecbinario
		for i in range(diferencia_largos):
			vecbinario = [0] + vecbinario
	return vecbinario
	baseAxu = 0

def GetinBinaryMayor(ArregloListo):
	bw = 0
	auxbit = 0
	auxbase = 0
	auxbase2 = 0
	newVector = []
	newVectorTotal = []

	# numero de clasificadores
	for auxbit in range(len(ArregloListo)):
		# n dimensiones
		for auxbase in range(len(ArregloListo[0])):
			# w bits
			for auxbase2 in range(len(ArregloListo[0][0])):
				if ArregloListo[auxbit][auxbase][auxbase2] == 0:
					bw = bw << 1
				elif ArregloListo[auxbit][auxbase][auxbase2] == 1:
					bw = bw << 1 | 1
			#   0    auxbase    n
			# [[3], [2], [9], ...]
			newVector.append([bw])
			bw = 0
		newVectorTotal.append(newVector)
		newVector = []
		bw = 0
	#  0	                        numero de clasificadores
	#    0               n
	# [[[3], [2], [9], ...], [[], ...], ...]
	return newVectorTotal

def CrearTablaHash(ClasificadoresBit, ClasificadoresWTA):
	numeroClasificadores = len(ClasificadoresBit)
	numeroBandas = len(ClasificadoresBit[0])
	# Creo que esto seria como hacer len(0) o len(1)
	numeroElementos = len(ClasificadoresWTA[0][0])
	# Esto da 8 para n = 1200, con 200 clasificadores, k = 16 y w = 2
	indicemayor = int(math.pow(2, numeroElementos))
	Indices = []

	############################################################################
	#

	TablaHashTotal = []
	for i in range(numeroBandas):
		for j in range(indicemayor):
			Indices.append([])
		TablaHashTotal.append(Indices)
		Indices = []

	for NumeroCalsifAux in range(numeroClasificadores):
		for NumeroBandasAux in range(numeroBandas):
			numeroIndice = \
				ClasificadoresBit[NumeroCalsifAux][NumeroBandasAux][0] 
			TablaHashTotal[NumeroBandasAux][numeroIndice].append(
				(NumeroCalsifAux + 1)
			)
	return TablaHashTotal

	# ?
	############################################################################

# compara entre los canales de la imagen y cada clasificador sumandole uno al 
# clasificador que tenga un MATCH, finalmente se entrega en indice del mejor
# clasificador
	
def FindBestClassifiers(TodosVEctoresWTAClasif, WTA1Imagen, TabladeHash):
	ArregloVAcioClasif = [0] * len(TodosVEctoresWTAClasif)

	for aux4 in range(len(WTA1Imagen)):
		Numero = WTA1Imagen[aux4][0]
        #REVISAR
		ArregloClasificadoresPresentes = TabladeHash[aux4][Numero]

		for auxNumVecClasif in range(len(ArregloClasificadoresPresentes)):
			clasificador = ArregloClasificadoresPresentes[auxNumVecClasif]
			ArregloVAcioClasif[clasificador - 1] = \
				ArregloVAcioClasif[clasificador - 1] + 1

	AuxSort = ArregloVAcioClasif[:]
	AuxSort.sort(reverse = True)
	auxSort2 = 0
	Ayuda = 99999
	AyudaIndex = 999999
	ClasificadoresTop = []
	#print AuxSort
	for auxSort2 in range(len(TodosVEctoresWTAClasif)):
		#while(329 > auxSort2):
		MayorMomentaneo = AuxSort[auxSort2]
		if Ayuda == MayorMomentaneo:
			Index24 = ArregloVAcioClasif.index(MayorMomentaneo,AyudaIndex+1)
			Ayuda = MayorMomentaneo
			AyudaIndex = Index24
			ClasificadoresTop.append(Index24)
		else:
			Index24 = ArregloVAcioClasif.index(MayorMomentaneo)
			ClasificadoresTop.append(Index24)			
			Ayuda = MayorMomentaneo
			AyudaIndex = Index24
	return ClasificadoresTop	

def ObtenerValoresTotalesWTA(listWTAClasif, listWTAImagenes, tablahash):
	listaValoresWTAClasif = []
	total_time = 0
	for auxWTAimagenes in range(len(listWTAImagenes)):
		start = time.time()
		auxWTAIndice = FindBestClassifiers(
			listWTAClasif,listWTAImagenes[auxWTAimagenes],tablahash
		)
		end = time.time()
		elapsed_time = end - start
		total_time = total_time + elapsed_time
		s = "Elapsed time in finding best classifiers for row {0} is {1}".format(
			auxWTAimagenes, elapsed_time
		)
		print (s)
		listaValoresWTAClasif.append(auxWTAIndice)
	avg_time = total_time / float(len(listWTAImagenes))
	print ("Average time in finding best classifiers is {0}".format(avg_time))
	return listaValoresWTAClasif
