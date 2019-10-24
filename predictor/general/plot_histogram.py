def histogram_plotter(plot_name, scored_dict):
    import matplotlib.pyplot as plt
    for position, scored_list in scored_dict.items():
        plt.hist(scored_list, bins=200, label="{}".format(position), alpha=0.5)
        plt.legend()
    plt.xlabel("PROcleavage score")
    plt.ylabel("Number of peptides")
    plt.savefig(plot_name)
    plt.cla()
    plt.clf()
