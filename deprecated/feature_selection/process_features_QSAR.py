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

def plot_data(data, descriptors, positions, colors, descriptors_):
    all_descriptors_ = descriptors_.split(",")
    print(all_descriptors_)
    all_descriptors = descriptors*positions
    all_colors = colors*positions
    descriptor_sum = {}
    x, y = [], []
    for i in range(48):
        l = []
        for j in range(positions):
            l.append(i + (48*j))
        v = [data[k] for k in l]
        print(all_descriptors_[i], i, round(sum(v), 4))
        x.append(all_descriptors_[i])
        y.append(round(sum(v), 4))

    plt.bar(x, y)
    plt.xticks(fontsize=8, rotation=90)
    plt.show()


def main():
    data_file = "QSAR_feature_importance_xgboost.txt"
    data = parse_data(data_file)
    descriptors_ = "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V14,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,V29,V30,V31,V32,V33,V34,V35,V36,V37,V38,V39,V40,V41,V42,V43,V44,V45,V46,V47,V48,V49,V50"
    descriptors = [str(i) for i in range(48)]
    positions = 7
    colors = ["yellow"]*16 + ["red"]*17 + ["green"]*15
    plot_data(data, descriptors, positions, colors, descriptors_)   


main()


