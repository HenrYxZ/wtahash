import wtahash
import time
import numpy as np

def main():
    k = input("Welcome, introduce the number for k = ")
    w = input("Now introduce the number for w = ")

    ############################################################################
    ###                        Load information matrix                       ###
    ############################################################################

    n = 1200
    path = ""

    print ("Starting to generate hash table ...")
    start = time.time()
    wta_hash = WTAHash(filename, n, k, w)
    end = time.time()
    table_time = end - start 
    print ("Elapsed time on generation of hash table {0}".format(table_time))