import argparse
from predictor.database_functions import peptide_extractor, uniprot_extractor
from predictor.core import peptide_uniprot_locator, data_generator
from predictor.ml_main import training_engine
from predictor.new_predictions import predictor

HELP = " \
Command:\n \
----------\n \
Run: python3 PROcleave.py --generate_data --NN\
"


def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate', help='Generate training data for the neural network', action='store_true')
    parser.add_argument('--train', help='Train the neural network', action='store_true')
    parser.add_argument('--predict', help='Predict proteasome cleavage from FASTA sequence', action='store_true')
    args = parser.parse_args()
    return args.generate, args.train, args.predict

def generating_training_data(iedb_path, uniprot_path, training_data_path):
    print("Generating training data...")
    iedb_data = peptide_extractor.extract_peptide_data(iedb_path) #Reading IEDB data, consider add argument about MHC class
    uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path) #Extracting uniprot data
    proteasome_dictionary = peptide_uniprot_locator.locate_peptides(iedb_data, uniprot_data) #Finding neighbours
    data_generator.prepare_cleavage_data(proteasome_dictionary, training_data_path)

def main(generate=False, train=False, predict=False):
    iedb_path = "../../data/raw/iedb/mhc_ligand_full.csv"
    uniprot_path = "../../data/raw/uniprot/uniprot_sprot.fasta"
    training_data_path = "data/training_data/proteasome"
    models_export_path = "data/models/proteasome"

    if not any([generate, train, predict]):
        print("Please, provide an argument. See python3 PROcleave.py -h for more information")
    if generate:
        generating_training_data(iedb_path, uniprot_path, training_data_path)
    if train:
        training_engine.create_models(training_data_path, models_export_path)
    if predict:
        sequence = "LVVSFVVGGLA"
        peptide_lenght_list = [9, 10]
        peptide_list = predictor.proteasome_prediction(sequence, models_export_path, peptide_lenght_list)


if __name__ == "__main__":
    generate, train, predict = parse_args()
    main(generate, train, predict)
