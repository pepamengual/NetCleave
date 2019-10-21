def file_saver(input_dictionary):
    for side, cleavage_region_dict in sorted(input_dictionary.items()):
        with open("{}.txt".format(side), "w") as f:
            for cleavage_region, probability in sorted(cleavage_region_dict.items(), key=lambda kv: kv[1]):
                line = "{} {} {}".format(side, cleavage_region, probability)
                f.write(line + "\n")
