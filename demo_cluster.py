import wtahash as wh
import time
import numpy as np
import cluster
import cPickle as pickle

def main():
    k = 16
    w = 2
    n = 1200


    ###                  Load training information matrix                    ###
    ###----------------------------------------------------------------------###
    
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats/airplane"
    print ("Reading training instances ...")
    start = time.time()
    cluster.load_class(80, path, "training")
    end = time.time()
    print ("Elapsed time reading the instances files {0}".format(end - start))

    
    ###                        Use WTAHash on it                             ###
    ###----------------------------------------------------------------------###
    
    print ("Starting to generate hash table ...")
    start = time.time()
    wta_hash = wh.WTAHash(objects, n, k, w)
    end = time.time()
    table_time = end - start 
    print ("Elapsed time on generation of hash table {0}".format(table_time))

    # Save the hash in a cPickle file
    print ("Starting to write the hash in a file ...")
    start = time.time()
    pickle.dump(wta_hash, open("/user/hjhenriq/hash.obj", "wb"))
    end = time.time()
    print ("Elapsed time writing the hash file {0}".format(end - start))

if __name__ == '__main__':
    main()
