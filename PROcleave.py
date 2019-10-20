from predictor.iedb_functions import ms_extractor
from predictor.general import save_pickle
from predictor.uniprot_functions import uniprot_extractor



def main():
    ### IEDB ###
    iedb_data_file_raw_path = "data/raw/iedb/mhc_ligand_full.csv"
    iedb_data = ms_extractor.extract_ms_data(iedb_data_file_raw_path)
    
    iedb_data_file_parsed_path = "data/parsed/iedb/ms_allele_peptides.pickle"
    save_pickle.pickle_saver(iedb_data_file_parsed_path, iedb_data)

    ### UNIPROT ###
    uniprot_data_file_raw_path = "data/raw/uniprot/uniprot_sprot.fasta"
    uniprot_data = uniprot_extractor.id_sequence_extractor(uniprot_data_file_raw_path)

    uniprot_data_file_parsed_path = "data/parsed/uniprot/uniprot_sequences.pickle"
    save_pickle.pickle_saver(uniprot_data_file_parsed_path, uniprot_data)

if __name__ == "__main__":
    main()
