def jaccard_sim(list_test, list_train):
    test = set(list_test)
    train = set(list_train)
    c = test.intersection(train)
    # return float(len(c)) / (len(test) + len(train) - len(c)) 
    return float(len(c)) / (len(test))