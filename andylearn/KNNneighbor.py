from sklearn import neighbors

def one_verification(train_data, train_label, test_data, test_label, k):
    knn = neighbors.KNeighborsClassifier(n_neighbors = k)  
    knn.fit(train_data, train_label)
    pre = knn.predict(test_data)
    print(pre[0], test_label)
    return pre[0]==test_label

def accuracy(train_data, train_label, test_data, test_label, k):
    knn = neighbors.KNeighborsClassifier(n_neighbors = k)  
    knn.fit(train_data, train_label)
    s = knn.score(test_data, test_label)
    return s