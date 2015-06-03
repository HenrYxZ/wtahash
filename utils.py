import numpy as np

def humanize_time(secs):
    # Extracted from http://testingreflections.com/node/6534
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)

def precision_recall(ranking):
    precisions = []
    recalls = []
    retrieved_inter_relevant = 0.0
    relevants_count = 0
    retrieved = 0
    for elem in ranking:
        if elem == "R":
            relevants_count += 1
    for i in range(len(ranking)):
        retrieved += 1
        # if this element is relevant
        if ranking[i] == "R":
            retrieved_inter_relevant += 1
            recalls.append(retrieved_inter_relevant / relevants_count)
            precisions.append(retrieved_inter_relevant / retrieved)
    return precisions, recalls

def average_precision(ranking):
    precisions, recalls = precision_recall(ranking)
    return np.average(precisions)

def relevants_retrieveds(ranking):
    relevant = 0
    retrieved = 0
    for i in range(len(ranking)):
        retrieved += 1
        if ranking[i] == "R":
            relevant += 1
    return relevant, retrieved

def precision_fixed_recall(ranking):
    precisions = [1.0]
    relevants_count = 0
    retrieved_inter_relevant = 0.0
    retrieved = 0
    step = len(ranking) / 10
    separators = []
    for i in range(len(ranking)):
        if ranking[i] == "R":
            if relevants_count % step == 0:
                separators.append(i)
            relevants_count += 1
    print("Separators = {0}".format(separators))
    # Iterate each 10% of the ranking
    for i in range(9):
        a = separators[i]
        b = separators[i + 1] - 1
        this_ranking = ranking[a:b]
        this_relevants, this_retrieveds = relevants_retrieveds(this_ranking)
        retrieved_inter_relevant += this_relevants
        retrieved += this_retrieveds
        precisions.append(retrieved_inter_relevant / retrieved)
    this_ranking = ranking[separators[-1]:]
    retrieved_inter_relevant += this_relevants
    retrieved += this_retrieveds
    precisions.append(retrieved_inter_relevant / retrieved)
    return precisions

def interpolate_p(precisions):
    # This is not an efficient algorithm but it shouldn't be slow
    return [max(precisions[i:]) for i in range(len(precisions))]

def relevance_ranking(ranking, labels, class_name):
    return ["R" if labels[index] == class_name else "F" for index in ranking]

def write_list(l, path):
    with open(path, "w") as f:
        for elem in l:
            f.write("{0}\n".format(elem))

def read_list(path):
    l = []
    with open(path, "r") as f:
        for line in f:
            l.append(line)
    print("first line = {0}".format(l[0]))
    return l