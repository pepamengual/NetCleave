import matplotlib.pyplot as plt

def parse_data(data_file):
    data = {}
    with open(data_file, "r") as f:
        for line in f:
            line = line.split(" ")
            feature_id = int(line[1][:-1])
            feature_score = float(line[-1][:-1])
            data.setdefault(feature_id, feature_score)
    return data

def plot_data(data, descriptors, positions, colors):
    all_descriptors = descriptors*positions
    all_colors = colors*positions
    plt.bar(data.keys(), data.values(), color = all_colors)
    plt.show()


def main():
    data_file = "VHSE_feature_importance_xgboost.txt"
    data = parse_data(data_file)
    descriptors = ["VHSE1", "VHSE2", "VHSE3", "VHSE4", "VHSE5", "VHSE6", "VHSE7", "VHSE8"]
    positions = 7
    colors = ["yellow", "orange", "pink", "red", "green", "cyan", "blue", "violet"]    
    plot_data(data, descriptors, positions, colors)   


main()


