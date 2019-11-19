from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_metrics(class_labels_test, class_labels_test_predicted):
    print("Computing accuracy, precision, recall and f1 score...")
    accuracy = accuracy_score(class_labels_test, class_labels_test_predicted)
    precision = precision_score(class_labels_test, class_labels_test_predicted, average='weighted')
    recall = recall_score(class_labels_test, class_labels_test_predicted, average='weighted')
    f1 = f1_score(class_labels_test, class_labels_test_predicted, average='weighted')
    return accuracy, precision, recall, f1
