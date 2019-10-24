def histogram_plotter(plot_name, peptide_dictionary_ms_random, max_mcc, index):
    import matplotlib.pyplot as plt
    
    all_data = peptide_dictionary_ms_random["MS"] + peptide_dictionary_ms_random["RANDOM"]
    min_data = min(all_data)
    max_data = max(all_data)
    diff = abs(min_data) + abs(max_data)
    bins_number = int(diff * 5)

    for data_kind, scored_list in peptide_dictionary_ms_random.items():
        plt.hist(scored_list, bins=bins_number, label="{}".format(data_kind), alpha=0.5)
    index_rounded = round(index, 4)
    max_mcc_rounded = round(max_mcc, 2)
    plt.axvline(x=index_rounded, label="MCC = {}".format(max_mcc_rounded), c="black")
    plt.xlabel("PROcleavage score")
    plt.ylabel("Number of peptides")
    plt.legend()
    plt.savefig("{}.png".format(plot_name))
    plt.cla()
    plt.clf()
