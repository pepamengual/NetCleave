from predictor.iedb_functions import ms_extractor
from predictor.general import save_pickle
from predictor.uniprot_functions import uniprot_extractor
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import analyze_distribution

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
    print("Computing random probabilities from UniProt")
    frequency_random_model = random_model.random_model_uniprot(uniprot_data)
    probability_dictionary_cleavage_region = analyze_distribution.distribution_cleavage(large_uniprot_peptide, adjacent_lenght, frequency_random_model)

    for cleavage_region, probability in sorted(probability_dictionary_cleavage_region.items(), key=lambda kv: kv[1]):
        print(cleavage_region, probability)


if __name__ == "__main__":
    main()
