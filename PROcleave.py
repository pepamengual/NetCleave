import argparse
from predictor.database_functions import peptide_extractor, uniprot_extractor, uniparc_extractor
from predictor.core import all_peptide_uniprot_locator, all_training_data_generator
from predictor.ml_main import all_training_engine
from predictor.new_predictions import all_predictor_fasta, predictor_set, predictor_linker_all

HELP = " \
Command:\n \
----------\n \
Run: python3 PROcleave.py --ARG\
"
def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate', help='Generate training data for the neural network', action='store_true')
    parser.add_argument('--train', help='Train the neural network', action='store_true')
    parser.add_argument('--predict_fasta', help='Predict selected cleavage from FASTA sequence', action='store_true')
    parser.add_argument('--score_set', help='Predict a set of peptides from IEDB file', action='store_true')
    args = parser.parse_args()
    return args.generate, args.train, args.predict_fasta, args.score_set

def generating_data(iedb_path, uniprot_path, uniparc_path_headers, uniparc_path_sequence, conditions):
    iedb_data = peptide_extractor.extract_peptide_data(iedb_path, conditions)
    uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path)
    uniparc_data = uniparc_extractor.extract_uniparc_data(uniparc_path_headers, uniparc_path_sequence)
    sequence_data = all_peptide_uniprot_locator.join_data(uniprot_data, uniparc_data)
    selected_dictionary = all_peptide_uniprot_locator.locate_peptides(iedb_data, sequence_data)
    return selected_dictionary

def main(generate=False, train=False, predict_fasta=False, score_set=False):
    iedb_path = "../PROcleave_data/mhc_ligand_full.csv" # download and unzip from http://www.iedb.org/database_export_v3.php
    uniprot_path = "../PROcleave_data/uniprot_sprot.fasta" # download and decompress from https://www.uniprot.org/downloads REVIEWED fasta
    uniparc_path_headers = "../PROcleave_data/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.tab"
    uniparc_path_sequence = "../PROcleave_data/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.fasta"
    mhc_class, technique = "I", "mass spectrometry"

    training_data_path = "data/training_data/{}_{}".format(mhc_class, technique.replace(" ", "-"))
    models_export_path = "data/models/{}_{}".format(mhc_class, technique.replace(" ", "-"))

    if not any([generate, train, predict_fasta, score_set]):
        print("Please, provide an argument. See python3 PROcleave.py -h for more information")
    
    if generate:
        conditions = {"Description": None, "Parent Protein IRI": None, 
                      "Method/Technique": ("contains", technique),
                      "Description": ("not_contains", "SIINFEKL"),
                      "MHC allele class": ("match", mhc_class)}
                      #"Name": ("contains", "Homo sapiens")}
                      #"Cell Type": ("match", "B cell")}
                      #"Allele Name": ()},
                      #"Parent Species": ("is_in", virus_and_bacteria)}
                      #"Name": ("contains", "Homo sapiens"), # host
                      #"Parent Species": ("not_contains", "Homo sapiens")} # other epitope different than human
        selected_dictionary = generating_data(iedb_path, uniprot_path, uniparc_path_headers, uniparc_path_sequence, conditions)
        all_training_data_generator.prepare_cleavage_data(selected_dictionary, training_data_path)

    if train:
        all_training_engine.create_models(training_data_path, models_export_path)
    
    if predict_fasta:
        results = {}
        peptide = "SIINFEKL"
        linker_list = ["AAA", "AAL", "AAY", "ADL", "A", "GGGS", "SSS"]
        linker_data = {}
        for linker in linker_list:
            sequence = "{0}{1}{0}".format(peptide, linker)
            print("Peptide: {}\nLinker: {}\nSequence: {}\n".format(peptide, linker, sequence))
            peptide_lenght_list = [len(peptide)]
            peptide_list, possible_cleavages = predictor_linker_all.processing_prediction(sequence, models_export_path, peptide_lenght_list, peptide, linker)
            for data in possible_cleavages:
                position = data[0]
                probability = data[1]
                linker_data.setdefault(linker, {}).setdefault(position, probability)
        print(linker_data)

        for linker in linker_data.keys():
            neighbours = linker_data[linker][5] + linker_data[linker][6] + linker_data[linker][7] + linker_data[linker][8] + linker_data[linker][9] 
            results.setdefault(linker, round(linker_data[linker][8]/neighbours, 3))
        for k, v in results.items():
            print(k, v)

    if score_set:
        conditions = {"Description": None, "Parent Protein IRI": None, 
                      "Method/Technique": ("contains", "mass spectrometry"),
                      #"Method/Technique": ("contains", "radio"),
                      "MHC allele class": ("match", "I")}
                      #"Name": ("contains", "Homo sapiens")}
                      #"Cell Type": ("match", "B cell")}
        export_set_path = "data/score_set/class_{0}/class_{0}_{1}_data.csv".format(conditions["MHC allele class"][1], "Individual")
        selected_dictionary = generating_data(iedb_path, uniprot_path, uniparc_path_headers, uniparc_path_sequence, conditions)
        scoring_data_generator.generating_scoring_data(selected_dictionary, export_set_path)
        predictor_set.selected_prediction(export_set_path, models_export_path, "Individual", "set")
    
if __name__ == "__main__":
    generate, train, predict_fasta, score_set = parse_args()
    main(generate, train, predict_fasta, score_set)
