from predictor.iedb_functions import ms_extractor
from predictor.uniprot_functions import uniprot_extractor
from predictor.general import save_pickle
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import cleavage_probabilities
from predictor.core import score_peptides
from predictor.general import compute_statistics
from predictor.general import plot_histogram
from predictor.general import save_file
from scipy.stats import ks_2samp

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
    n = 5
    large_uniprot_peptide = seek_ms_uniprot.seeking_ms(iedb_data, uniprot_data, n)
    
    ### Random model from UniProt ###
    print("Computing random probabilities from UniProt...")
    frequency_random_model = random_model.random_model_uniprot_collections(uniprot_data)

    ### Computing cleavage probabilities ###
    print("Computing cleavage probabilities...")
    erad_cleavage = {"large": [n -3, n - 2, n - 1, n, n + 1], "short": [n - 2, n - 1, n, n + 1], "extra-short": [n - 1, n, n + 1]}
    proteasome_cleavage = {"large": [(n + 2) * -1, (n + 1) * -1, (n) * -1, (n - 1) * -1, (n - 2) * -1], "short": [(n + 2) * -1, (n + 1) * -1, (n) * -1, (n - 1) * -1], "extra-short": [(n + 2) * -1, (n + 1) * -1, (n) * -1]}
    print("Computing erad cleavage probabilities...")
    erad_cleavage_probabilities = cleavage_probabilities.distribution_cleavage(large_uniprot_peptide, frequency_random_model, erad_cleavage)
    print("Computing proteasome cleavage probabilities...")
    proteasome_cleavage_probabilities = cleavage_probabilities.distribution_cleavage(large_uniprot_peptide, frequency_random_model, proteasome_cleavage)

    ### Saving cleavage probabilities ###
    print("Saving cleavage probabilities...")
    save_file.file_saver(erad_cleavage_probabilities, "erad")
    save_file.file_saver(proteasome_cleavage_probabilities, "proteasome")
 
    ### Scoring MS peptides ###
    print("Scoring MS peptides...")
    scored_dict = score_peptides.main_scorer(large_uniprot_peptide, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model)
    
    ### Computing KS, p-value, mcc and ROC curves ###
    for kind_prediction, peptide_dictionary_ms_random in scored_dict.items():
        plot_name = "{}".format(kind_prediction)
        ms_peptide_scores = peptide_dictionary_ms_random["MS"]
        random_peptide_scores = peptide_dictionary_ms_random["RANDOM"]

        print("Results of {} only prediction:".format(kind_prediction))
        print("--> KS-test:")
        print(ks_2samp(ms_peptide_scores, random_peptide_scores))

        print("Computing MCC and plotting ROC curve...")
        max_mcc, index = compute_statistics.statistics(peptide_dictionary_ms_random, plot_name)

        print("--> Plotting {} histogram".format(kind_prediction))
        plot_histogram.histogram_plotter(plot_name, peptide_dictionary_ms_random, max_mcc, index)

if __name__ == "__main__":
    main()
