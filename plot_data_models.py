import matplotlib.pyplot as plt
import pandas as pd


def read_csv(input_path):
    df = pd.read_csv(input_path)
    df.rename(columns={df.columns[0]: "amino_acid"}, inplace = True)
    df.rename(columns={df.columns[1]: "group"}, inplace = True)
    return df

def pivot_dataframe(df, value):
    pivot_df = df.pivot(index='amino_acid', columns='group', values=value)
    return pivot_df  

def plot_pivot_df(pivot_df, value, ylabel, input_path):
    colors = ["#006D2C", "#31A354","#74C476"]
    pivot_df.loc[:,['Train','Test', 'Val']].plot.bar(stacked=False, color=colors, figsize=(12,5))
    plt.xticks(rotation="horizontal")
    plt.xlabel('')
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=False)
    plt.savefig("images/{}_{}.png".format(input_path.split(".csv")[0], value), dpi=300, bbox_inches="tight")

def main():
    input_path = "models_class_I_MS.csv"
    df = read_csv(input_path)
    value, ylabel = "loss", "Loss"
    value, ylabel = "auc", "Area under the curve (AUC)"
    pivot_df = pivot_dataframe(df, value)
    plot_pivot_df(pivot_df, value, ylabel, input_path)

main()
