from sklearn.model_selection import train_test_split

def splitting_data(data, class_labels):
    data_train, data_test, class_labels_train, class_labels_test = train_test_split(data, class_labels, test_size = 0.20, random_state=42)
    print("Shape of training data looks like the following...")
    print(data_train.shape)
    print("Shape of testing data looks like the following...")
    print(data_test.shape)
    return data_train, data_test, class_labels_train, class_labels_test
