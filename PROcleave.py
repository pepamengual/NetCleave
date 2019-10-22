from predictor.iedb_functions import ms_extractor
from predictor.uniprot_functions import uniprot_extractor
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import analyze_distribution
from predictor.core import get_distribution_cleavage
from predictor.core import score_cleavage_sites_new
from predictor.general import save_pickle
from predictor.general import save_file
from predictor.general import plot_histogram
from predictor.general import compute_mcc
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

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
    
    ### Random model from UniProt ###
    print("Computing random probabilities from UniProt...")
    pre_post_cleavage_large = {"left": [adjacent_lenght - 2, adjacent_lenght - 1, adjacent_lenght, adjacent_lenght + 1], "right": [(adjacent_lenght + 2) * -1, (adjacent_lenght + 1) * -1, adjacent_lenght * -1, (adjacent_lenght - 1) * -1]}
    pre_post_cleavage_short = {"left": [adjacent_lenght - 1, adjacent_lenght, adjacent_lenght + 1], "right": [(adjacent_lenght + 1) * -1, adjacent_lenght * -1, (adjacent_lenght - 1) * -1]}
    
    pre_post_cleavage_list = [pre_post_cleavage_large, pre_post_cleavage_short]
    pre_post_cleavage_names = ["large", "short"]
    frequency_random_model = random_model.random_model_uniprot_collections(uniprot_data)
    
    ### Computing cleavage probabilities ###
    print("Computing cleavage probabilities...")
    probability_dictionary_cleavage_region_large = get_distribution_cleavage.distribution_cleavage(large_uniprot_peptide, frequency_random_model, pre_post_cleavage_large)
    probability_dictionary_cleavage_region_short = get_distribution_cleavage.distribution_cleavage(large_uniprot_peptide, frequency_random_model, pre_post_cleavage_short)
    
    probability_dictionary_cleavage_region_list = [probability_dictionary_cleavage_region_large, probability_dictionary_cleavage_region_short]

    ### Saving cleavage probabilities ###
    print("Saving cleavage probabilities...")
    #save_file.file_saver(probability_dictionary_cleavage_region)
 
    ### Scoring MS peptides ###
    print("Scoring MS peptides...")
    scored_dict = score_cleavage_sites_new.scoring_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list, frequency_random_model)
    
    ### Computing KS and p-value ###
    print(ks_2samp(scored_dict["Cleavage sites"], scored_dict["Random sites"]))

    ### Plotting histogram ###
    plot_name = "fig_PROcleave_4-3_4-3.png"
    plot_histogram.histogram_plotter(plot_name, scored_dict)

    ### Computing MCC ###
    compute_mcc.mcc_computer(scored_dict)

if __name__ == "__main__":
    main()
