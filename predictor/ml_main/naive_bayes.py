import os
import pandas as pd
from predictor.ml_main.ml_utilities import read_table
from predictor.ml_main.ml_utilities import convert_to_uniform_length
from predictor.ml_main.ml_utilities import splitting_data
from predictor.ml_main.ml_utilities import bayes_classifier
from predictor.ml_main.ml_utilities import predicting_labels_of_testing
from predictor.ml_main.ml_utilities import get_metrics
from predictor.ml_main.ml_utilities import print_and_save_metrics
from predictor.ml_main.ml_utilities import gaussian_naive_bayes

def process_data(file_path):
    print("Training for {} data".format(file_path))
    training_table = read_table.read_training_table(file_path)
    training_table_texts = training_table['cleavage_region']
    class_labels = training_table.iloc[:, 0].values
    
    data = convert_to_uniform_length.convert_to_uniform_length(training_table_texts)

    data_train, data_test, class_labels_train, class_labels_test = splitting_data.splitting_data(data, class_labels)
    classifier = gaussian_naive_bayes.gaussian_classifier(data_train, class_labels_train)
    class_labels_test_predicted = predicting_labels_of_testing.predicting_labels_of_testing(classifier, data_test)
    accuracy, precision, recall, f1 = get_metrics.get_metrics(class_labels_test, class_labels_test_predicted)
    print_and_save_metrics.show_save_results(accuracy, precision, recall, f1, class_labels_test, class_labels_test_predicted, file_path)

