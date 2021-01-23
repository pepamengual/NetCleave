import argparse
from predictor.database_functions import peptide_extractor, uniprot_extractor, uniparc_extractor
from predictor.core import all_peptide_uniprot_locator, all_training_data_generator
from predictor.ml_main import run_NN
from predictor.predictions import predict_csv

HELP = " \
Command:\n \
----------\n \
Run: python3 NetCleave.py --ARG\
"
def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate', help='Generate training data for the neural network', action='store_true')
    parser.add_argument('--train', help='Train the neural network', action='store_true')
    parser.add_argument('--score_csv', help='Predict a set of cleavage sites from csv', action='store_true')
    args = parser.parse_args()
    return args.generate, args.train, args.score_csv

def generating_data(iedb_path, uniprot_path, uniparc_path_headers, uniparc_path_sequence, conditions):
    iedb_data = peptide_extractor.extract_peptide_data(iedb_path, conditions)
    uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path)
    uniparc_data = uniparc_extractor.extract_uniparc_data(uniparc_path_headers, uniparc_path_sequence)
    sequence_data = all_peptide_uniprot_locator.join_data(uniprot_data, uniparc_data)
    selected_dictionary = all_peptide_uniprot_locator.locate_peptides(iedb_data, sequence_data)
    return selected_dictionary

def main(generate=False, train=False, score_csv=False):
    mhc_class, technique, mhc_family = "I", "mass spectrometry", "HLA-A"
    training_data_path = "data/training_data/{}_{}_{}".format(mhc_class, technique.replace(" ", "-"), mhc_family)
    models_export_path = "data/models/{}_{}_{}".format(mhc_class, technique.replace(" ", "-"), mhc_family)

    if not any([generate, train, score_csv]):
        print("Please, provide an argument. See python3 NetCleave.py -h for more information")
    
    if generate:
        iedb_path = "../NetCleave_data/mhc_ligand_full.csv" # download and unzip from http://www.iedb.org/database_export_v3.php
        uniprot_path = "../NetCleave_data/uniprot/uniprot_sprot.fasta" # download and decompress from https://www.uniprot.org/downloads REVIEWED fasta
        uniparc_path_headers = "../NetCleave_data/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.tab"
        uniparc_path_sequence = "../NetCleave_data/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.fasta"

        conditions = {"Description": None, "Parent Protein IRI": None, 
                      "Method/Technique": ("contains", technique),
                      "MHC allele class": ("match", mhc_class),
                      "Allele Name": ("contains", mhc_family),
                      #"Name": ("contains", "Homo sapiens"),
                      #"Parent Species": ("contains", "Homo sapiens")
                     }
        selected_dictionary = generating_data(iedb_path, uniprot_path, uniparc_path_headers, uniparc_path_sequence, conditions)
        all_training_data_generator.prepare_cleavage_data(selected_dictionary, training_data_path)

    if train:
        run_NN.create_models(training_data_path, models_export_path)

    if score_csv:
        csv_path = "example_file_NetCleave_score.csv"
        predict_csv.score_set(csv_path, models_export_path, "ABC")

if __name__ == "__main__":
    generate, train, score_csv = parse_args()
    main(generate, train, score_csv)
