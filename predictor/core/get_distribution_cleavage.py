def distribution_cleavage(data, frequency_random_model, pre_post_cleavage):
    import numpy as np
    count = 0
    data_dictionary = {}
    for peptide in data:
        cleavage_region_left = ""
        cleavage_region_right = ""
        for side, side_values in pre_post_cleavage.items():
            cleavage_site = "".join([peptide[r] for r in side_values])
            if side == "left":
                cleavage_region_left += cleavage_site
            if side == "right": # right
                cleavage_region_right += cleavage_site

        data_dictionary.setdefault("left", {}).setdefault(cleavage_region_left, 0)
        data_dictionary.setdefault("right", {}).setdefault(cleavage_region_right, 0)
        data_dictionary["left"][cleavage_region_left] += 1
        data_dictionary["right"][cleavage_region_right] += 1
        count += 1
        
    frequency_dictionary = get_frequency_dictionary(data_dictionary, count, frequency_random_model, pre_post_cleavage)

    return frequency_dictionary


def get_frequency_dictionary(data_dictionary, count, frequency_random_model, pre_post_cleavage):
    import numpy as np
    frequency_dictionary = {}
    for side, cleavage_region_dict in data_dictionary.items():
        len_side = len(pre_post_cleavage[side])
        for cleavage_region, counts in cleavage_region_dict.items():
            frequency = (counts / count)
            probability_cleavage_region_random = np.prod([frequency_random_model[cleavage_region[n]] for n in range(len_side)])
            probability = np.log2(frequency / probability_cleavage_region_random) * -1
            frequency_dictionary.setdefault(side, {}).setdefault(cleavage_region, probability)
    return frequency_dictionary
