import matplotlib.pyplot as plt
import os
import pandas as pd
from predictor.ml_main.ml_utilities import read_table
from predictor.ml_main.ml_utilities import splitting_data
from predictor.ml_main.ml_utilities import predicting_labels_of_testing
from predictor.ml_main.ml_utilities import get_metrics
from predictor.ml_main.ml_utilities import print_and_save_metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from joblib import dump, load


def process_data(file_path):
    print("Training for {} data".format(file_path))
    training_table = read_table.read_training_table(file_path)
    class_labels = training_table['class']
    training_table_texts = training_table.drop(['class'], axis=1)
    
    data_train, data_test, class_labels_train, class_labels_test = splitting_data.splitting_data(training_table_texts, class_labels)
    classifier = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='sgd', verbose=10, random_state=21, tol=0.000000001)
    classifier.fit(data_train, class_labels_train)
    class_labels_test_predicted = classifier.predict(data_test)

    accuracy, precision, recall, f1 = get_metrics.get_metrics(class_labels_test, class_labels_test_predicted)
    cm = confusion_matrix(class_labels_test, class_labels_test_predicted)
    cm_string = str(cm)
    output_path = "{}_NN_report.txt".format(file_path.split(".txt")[0])

    with open(output_path, "w") as f:
        f.write("accuracy: {}".format(accuracy))
        f.write("precision: {}".format(precision))
        f.write("recall: {}".format(recall))
        f.write("f1: {}".format(f1))
        f.write("confusion matrix: {}".format(cm_string))

    sns.heatmap(cm, center=True)
    plt.savefig("{}_NN_cm.png".format(file_path.split(".txt")[0]))
    dump(classifier, "{}_NN_model.joblib".format(file_path.split(".txt")[0]))
    #clf = load('filename.joblib') 
