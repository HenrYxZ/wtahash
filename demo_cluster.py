from evaluation import Evaluation

################################################################################
####                                Main                                    ####
################################################################################
def main():
    path = "/mnt/nas/GrimaRepo/datasets/mscoco/coco2014/crops/cropsFeats"
    #DEBUGGING
    #path = "./features"
    ev = Evaluation(path)
    ev.run()    

################################################################################



if __name__ == '__main__':
    main()
