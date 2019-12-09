from predictor.database_functions import ms_extractor
from predictor.database_functions import uniprot_extractor
from predictor.general import save_pickle
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import random_peptide_generator
from predictor.core import get_cleavage_region
from predictor.ml_main import NN

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
    #frequency_random_model = random_model.random_model_uniprot_collections(uniprot_data)
    frequency_random_model = {'A': 0.08258971312579017, 'C': 0.013826094946210853, 'D': 0.054625650802595425, 'E': 0.0673214708897148, 'F': 0.03866188645338429, 'G': 0.07077863527330625, 'H': 0.022761656446475265, 'I': 0.05923828965491043, 'K': 0.05815460235107699, 'L': 0.09655733034859719, 'M': 0.024154886555486327, 'N': 0.04061129236837406, 'P': 0.047331721936265635, 'Q': 0.03932403048405303, 'R': 0.05534153979141534, 'S': 0.06631318414876945, 'T': 0.05355909368186356, 'V': 0.06865326331945962, 'W': 0.010987143802538912, 'Y': 0.029208513619712422}
    print(frequency_random_model)
    
    ### Generating random peptides ###
    random_peptides = random_peptide_generator.generate_random_peptides(large_uniprot_peptide, frequency_random_model)

    ### Saving file for ML ###
    get_cleavage_region.compute_cleavage_regions(large_uniprot_peptide, random_peptides, n)
    
    ### Running ML ###
    proteasome_path = "data/ml_dataframes/proteasome_dataframe_for_ml.txt"
    NN.process_data(proteasome_path)

    erap_path = "data/ml_dataframes/erap_dataframe_for_ml.txt"
    NN.process_data(erap_path)

if __name__ == "__main__":
    main()
