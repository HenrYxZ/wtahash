import utils

if __name__ == "__main__":
    a = ['R', 'F', 'R', 'F', 'F', 'R', 'R', 'R', 'R', 'R', 'F', 'F', 'R', 'F', 'R', 'F', 'R', 'R', 'R', 'F']
    precs = utils.precision_fixed_recall(a)
    print(precs)