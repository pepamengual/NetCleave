def file_saver(probability_dict, cleavage_type, counts_dict):
    printer_dict = {}
    for side, cleavage_region_dict in sorted(probability_dict.items()):
        for cleavage_region, probability in cleavage_region_dict.items():
            printer_dict.setdefault(cleavage_region, (probability, counts_dict[side][cleavage_region]))

    with open("{}_scores.txt".format(cleavage_type), "w") as f:
        for cleavage_region, tupple_information in sorted(printer_dict.items(), key=lambda kv: kv[1]):
            probability = round(tupple_information[0], 3)
            counts = tupple_information[1]
            line = "{} {} {}".format(probability, counts, cleavage_region)
            f.write(line + "\n")
