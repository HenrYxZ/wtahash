import wtahash as wh
import time
import numpy as np
import cluster
import cPickle as pickle

def main():
    k = 16
    w = 2
    n = 1200
    results = ""

    ###                  Load training information matrix                    ###
    ###----------------------------------------------------------------------###
    
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats"
    print ("Reading training instances ...")
    start = time.time()
    objects = cluster.load_classes(80, path, "training")
    end = time.time()
    s = "Elapsed time reading the instances files {0}".format(end - start)
    results += s + "\n"
    print (s)

    
    ###                        Use WTAHash on it                             ###
    ###----------------------------------------------------------------------###
    
    print ("Starting to generate hash table ...")
    start = time.time()
    wta_hash = wh.WTAHash(objects, n, k, w)
    end = time.time()
    table_time = end - start
    s = "Elapsed time on generation of hash table {0}".format(table_time)
    results += s + "\n"
    print (s)

    # Save the hash in a cPickle file
    print ("Starting to write the hash in a file ...")
    start = time.time()
    with open("/user/hjhenriq/wtahash/hash.obj", "wb") as f:
        pickle.dump(wta_hash, f)
    end = time.time()
    s = "Elapsed time writing the hash file {0}".format(end - start)
    results += s + "\n"
    print (s)

    # Write times in a text file
    with open("results.txt", "w") as f:
        f.write(s)

if __name__ == '__main__':
    main()
