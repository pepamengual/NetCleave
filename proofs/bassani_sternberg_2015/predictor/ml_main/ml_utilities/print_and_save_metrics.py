import pandas as pd

def show_save_results(accuracy, precision, recall, f1, class_labels_test, class_labels_test_predicted, training_table_path):
    name = "{}_results.txt".format(training_table_path.split(".txt")[0])
    print("{}\n".format(name))

    print("Confusion matrix\n")
    print(pd.crosstab(pd.Series(class_labels_test, name='Actual'), pd.Series(class_labels_test_predicted, name='Predicted')))
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    
    with open(name, "w") as f:
        confusion_matrix = pd.crosstab(pd.Series(class_labels_test, name='Actual'), pd.Series(class_labels_test_predicted, name='Predicted'))
        metrics = "accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1)
        f.write("Confusion matrix\n")
        f.write(confusion_matrix)
        f.write(metrics)
