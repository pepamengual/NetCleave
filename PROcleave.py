from predictor.iedb_functions import ms_extractor
from predictor.uniprot_functions import uniprot_extractor
from predictor.general import save_pickle
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import random_peptide_generator

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
    iedb_data_file_parsed_path = "../data/parsed/iedb/ms_allele_peptides"
    save_pickle.pickle_saver(iedb_data, iedb_data_file_parsed_path)
    uniprot_data_file_parsed_path = "../data/parsed/uniprot/uniprot_sequences"
    save_pickle.pickle_saver(uniprot_data, uniprot_data_file_parsed_path)

    ### Seeking MS into UniProt ###
    print("Seeking for MS peptides into UniProt data...")
    n = 5
    large_uniprot_peptide = seek_ms_uniprot.seeking_ms(iedb_data, uniprot_data, n)
    
    ### Random model from UniProt ###
    print("Computing random probabilities from UniProt...")
    frequency_random_model = random_model.random_model_uniprot_collections(uniprot_data)
    print(frequency_random_model)
    
    ### Generating random peptides ###
    random_peptides = random_peptide_generator.generate_random_peptides(large_uniprot_peptide, frequency_random_model)



if __name__ == "__main__":
    main()
