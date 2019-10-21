from predictor.iedb_functions import ms_extractor
from predictor.uniprot_functions import uniprot_extractor
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import analyze_distribution
from predictor.core import get_distribution_cleavage
from predictor.core import score_cleavage_sites
from predictor.general import save_pickle
from predictor.general import save_file


def main():
    ### IEDB ###
    print("Extracting MS data from IEDB...")
    iedb_data_file_raw_path = "../data/raw/iedb/mhc_ligand_full.csv"
    iedb_data = ms_extractor.extract_ms_data(iedb_data_file_raw_path)
    
    ### UNIPROT ###
    print("Extracting UniProt data...")
    uniprot_data_file_raw_path = "../data/raw/uniprot/uniprot_sprot.fasta"
    uniprot_data = uniprot_extractor.id_sequence_extractor(uniprot_data_file_raw_path)
    
    ### Saving pickles ###
    print("Saving MS and UniProt data into pickles...")
    iedb_data_file_parsed_path = "../data/parsed/iedb/ms_allele_peptides.pickle"
    save_pickle.pickle_saver(iedb_data_file_parsed_path, iedb_data)
    uniprot_data_file_parsed_path = "../data/parsed/uniprot/uniprot_sequences.pickle"
    save_pickle.pickle_saver(uniprot_data_file_parsed_path, uniprot_data)

    ### Seeking MS into UniProt ###
    print("Seeking for MS peptides into UniProt data...")
    adjacent_lenght = 5
    large_uniprot_peptide = seek_ms_uniprot.seeking_ms(iedb_data, uniprot_data, adjacent_lenght)
    #frequency_dictionary_preadjacent = analyze_distribution.distribution_analyzer(large_uniprot_peptide, adjacent_lenght)
    #analyze_distribution.distribution_plotter(frequency_dictionary_preadjacent, adjacent_lenght)
    #frequency_random_model = random_model.random_model_all_peptides(large_uniprot_peptide)
    
    ### Random model from UniProt ###
    print("Computing random probabilities from UniProt...")
    pre_post_cleavage = {"left": [adjacent_lenght -2, adjacent_lenght - 1, adjacent_lenght], "right": [(adjacent_lenght + 1) * -1, adjacent_lenght * -1]}
    frequency_random_model = random_model.random_model_uniprot_collections(uniprot_data)
    
    ### Computing cleavage probabilities ###
    print("Computing cleavage probabilities...")
    probability_dictionary_cleavage_region = get_distribution_cleavage.distribution_cleavage(large_uniprot_peptide, frequency_random_model, pre_post_cleavage)
    
    ### Saving cleavage probabilities ###
    print("Saving cleavage probabilities...")
    save_file.file_saver(probability_dictionary_cleavage_region)
 
    ### Scoring MS peptides ###
    print("Scoring MS peptides...")
    scored_dict = score_cleavage_sites.scoring_peptides(large_uniprot_peptide, probability_dictionary_cleavage_region, pre_post_cleavage)
    import matplotlib.pyplot as plt
    for position, scored_list in scored_dict.items():
        plt.hist(scored_list, bins=50, label="{}".format(position), alpha=0.5)
        plt.legend()    
    plt.show()

if __name__ == "__main__":
    main()
