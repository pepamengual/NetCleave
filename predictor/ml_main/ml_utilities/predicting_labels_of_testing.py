from sklearn.naive_bayes import MultinomialNB

def predicting_labels_of_testing(classifier, data_test):
    print("Predicting the labels of the test set...")
    class_labels_test_predicted = classifier.predict(data_test)
    return class_labels_test_predicted
