La librería consta de 2 archivos, wtahash.py y WTALibrary.

Para utilizar la librería se debe importar wtahash. (wtahash usa a WTALibrary por lo que ese archivo también debe estar presente en la carpeta)

Se debe crear una instancia de la clase WTAHash que recibe como parámetro el path del archivo donde se encuentran los clasificadores y los valores n,k y w de WTAHash.
El archivo de clasificadores debe tener un vector de pesos por fila, donde cada elemento está separado por una coma.
La clase WTAHash tiene un método llamado best_classifiers, que recibe una lista de vectores y retorna una arreglo de listas donde para cada lista del input hay una lista
con todos los clasificadores ordenados descendientemente según su puntaje de match.

Ejemplo:

import cPickle as pickle
import wtahash as wta

x = wta.WtaHash('ArchivoDeClasificadores.txt',1200,16,2)
# img = lista de vectores
bestClassifiers = x.best_classifiers(img)

#en bestClassifiers[0][0] está el mejor clasificador para el vector 0, bestClassifiers[0][1] el segundo mejor clasificador para el vector 0,
#bestClassifiers[1][0] el mejor clasificador para el vector 1, etc.



PD: Dado que construir el hash toma bastante tiempo, si se va a utilizar el mismo hash en ocaciones reiteradas se recomienda crear el objeto una sola vez y luego
serializar y deserializar (se puede utilizar cPickle por ejemplo)

Carpeta en cluster de MS - COCO

/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats
270.000 archivos .mat de dimension (D) = 4.096, 72 clases
1.105.920.000 floats de ¿32 bit? = 4 bytes en la matriz de datos de 270000x4096.
1.080.000 kbytes
1.055 Mbytes
1.03 Gbytes de datos.