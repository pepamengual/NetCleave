import argparse
from predictor.database_functions import peptide_extractor, uniprot_extractor
from predictor.core import peptide_uniprot_locator, training_data_generator, scoring_data_generator
from predictor.ml_main import training_engine
from predictor.new_predictions import predictor_fasta, predictor_set

HELP = " \
Command:\n \
----------\n \
Run: python3 PROcleave.py --generate_data --NN\
"


def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate', help='Generate training data for the neural network', action='store_true')
    parser.add_argument('--train', help='Train the neural network', action='store_true')
    parser.add_argument('--predict_fasta', help='Predict proteasome cleavage from FASTA sequence', action='store_true')
    parser.add_argument('--score_set', help='Predict a set of peptides from IEDB file', action='store_true')
    args = parser.parse_args()
    return args.generate, args.train, args.predict_fasta, args.score_set

def generating_training_data(iedb_path, uniprot_path, training_data_path):
    print("Generating training data...")
    mhc_class = "I"
    iedb_data = peptide_extractor.extract_peptide_data(iedb_path, mhc_class) #Reading IEDB data, consider add argument about MHC class
    uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path) #Extracting uniprot data
    proteasome_dictionary = peptide_uniprot_locator.locate_peptides(iedb_data, uniprot_data) #Finding neighbours
    training_data_generator.prepare_cleavage_data(proteasome_dictionary, training_data_path)

def main(generate=False, train=False, predict_fasta=False, score_set=False):
    iedb_path = "../../data/raw/iedb/mhc_ligand_full.csv"
    uniprot_path = "../../data/raw/uniprot/uniprot_sprot.fasta"
    
    iedb_path = "../PROcleave_data/mhc_ligand_full.csv" # download and unzip from http://www.iedb.org/database_export_v3.php
    uniprot_path = "../PROcleave_data/uniprot_sprot.fasta" # download and decompress from https://www.uniprot.org/downloads REVIEWED fasta
    training_data_path = "data/training_data/proteasome"
    models_export_path = "data/models/proteasome"

    if not any([generate, train, predict_fasta, score_set]):
        print("Please, provide an argument. See python3 PROcleave.py -h for more information")
    if generate:
        generating_training_data(iedb_path, uniprot_path, training_data_path)
    if train:
        training_engine.create_models(training_data_path, models_export_path)
    if predict_fasta:
        sequence = "LVVSFVVGGLAPKLEDIDLE"
        peptide_lenght_list = [8, 9, 10]
        peptide_list = predictor_fasta.proteasome_prediction(sequence, models_export_path, peptide_lenght_list)
    if score_set:
        mhc_class = "I" #conditions of peptides, more can be added. Changing parameters in peptide_extractor
        export_path = "data/score_set/class_{}/class_{}_data.csv".format(mhc_class, mhc_class)
        #iedb_data = peptide_extractor.extract_peptide_data(iedb_path, mhc_class)
        conditions_dictionary = {"Description": None, "Parent Protein IRI": None, "Method/Technique": ("contains", "mass spectrometry"), "MHC allele class": ("match", "I")}
        iedb_data = peptide_extractor.extract_peptide_data_pandas(iedb_path, conditions_dictionary)
        uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path)
        proteasome_dictionary = peptide_uniprot_locator.locate_peptides(iedb_data, uniprot_data)
        scoring_data_generator.generating_scoring_data(proteasome_dictionary, export_path)
        predictor_set.proteasome_prediction(export_path, models_export_path)

if __name__ == "__main__":
    generate, train, predict_fasta, score_set = parse_args()
    main(generate, train, predict_fasta, score_set)
