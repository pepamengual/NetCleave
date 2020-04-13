import argparse
from predictor.database_functions import peptide_extractor, uniprot_extractor
from predictor.core import peptide_uniprot_locator, training_data_generator, scoring_data_generator
from predictor.ml_main import training_engine, training_engine_all
from predictor.new_predictions import predictor_fasta, predictor_set, predictor_linker, predictor_set_all

HELP = " \
Command:\n \
----------\n \
Run: python3 PROcleave.py --predict_fasta\
"


def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate', help='Generate training data for the neural network', action='store_true')
    parser.add_argument('--train', help='Train the neural network', action='store_true')
    parser.add_argument('--train_all', help='Train the neural network', action='store_true')
    parser.add_argument('--predict_fasta', help='Predict proteasome cleavage from FASTA sequence', action='store_true')
    parser.add_argument('--score_set', help='Predict a set of peptides from IEDB file', action='store_true')
    parser.add_argument('--score_set_all', help='Predict a set of peptides from IEDB file', action='store_true')
    args = parser.parse_args()
    return args.generate, args.train, args.train_all, args.predict_fasta, args.score_set, args.score_set_all

def generating_data(iedb_path, uniprot_path, conditions):
    iedb_data = peptide_extractor.extract_peptide_data(iedb_path, conditions)
    uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path)
    proteasome_dictionary, mutation_dictionary = peptide_uniprot_locator.locate_peptides(iedb_data, uniprot_data)
    
    return proteasome_dictionary

def main(generate=False, train=False, train_all=False, predict_fasta=False, score_set=False, score_set_all=False):
    iedb_path = "../PROcleave_data/mhc_ligand_full.csv" # download and unzip from http://www.iedb.org/database_export_v3.php
    uniprot_path = "../PROcleave_data/uniprot_sprot.fasta" # download and decompress from https://www.uniprot.org/downloads REVIEWED fasta
    training_data_path = "data/training_data/proteasome"
    models_export_path = "data/models/proteasome"

    if not any([generate, train, train_all, predict_fasta, score_set, score_set_all]):
        print("Please, provide an argument. See python3 PROcleave.py -h for more information")
    
    if generate:
        virus_and_bacteria = []
        with open("../PROcleave_data/virus_and_bacteria.txt", "r") as f:
            for line in f:
                line = line.rstrip().split(",")
                virus_and_bacteria.append(line[0])

        conditions = {"Description": None, "Parent Protein IRI": None, 
                      "Method/Technique": ("contains", "mass spectrometry"), 
                      "MHC allele class": ("match", "I"),
                      "Cell Type": ("match", "B cell")}
                      #"Allele Name": ()},
                      #"Parent Species": ("is_in", virus_and_bacteria)}
                      #"Name": ("contains", "Homo sapiens"), # host
                      #"Parent Species": ("not_contains", "Homo sapiens")} # other epitope different than human
        proteasome_dictionary = generating_data(iedb_path, uniprot_path, conditions)
        training_data_generator.prepare_cleavage_data(proteasome_dictionary, training_data_path)

    if train:
        training_engine.create_models(training_data_path, models_export_path)
    if train_all:
        training_engine_all.create_models(training_data_path, models_export_path)

    if predict_fasta:
        peptide = "SIINFEKL"
        linker_list = ["AAA", "AAL", "AAY", "ADL", "A", "GGGS", "SSS"]
        linker_data = {}
        for linker in linker_list:
            sequence = "{0}{1}{0}".format(peptide, linker)
            print("Peptide: {}\nLinker: {}\nSequence: {}\n".format(peptide, linker, sequence))
            peptide_lenght_list = [len(peptide)]
            peptide_list, possible_cleavages = predictor_linker.proteasome_prediction(sequence, models_export_path, peptide_lenght_list, peptide, linker)
            for data in possible_cleavages:
                position = data[0]
                probability = data[1]
                linker_data.setdefault(linker, {}).setdefault(position, probability)
        print(linker_data)
        
        for linker, linker_dict in linker_data.items():
            sum_probabilities = sum(linker_dict.values())
            num_probabilities = len(linker_dict)
            cleavage = linker_dict[7]
            prob = (cleavage/(sum_probabilities))/num_probabilities
            print(linker, cleavage, round(prob, 5))
        d = []
        for linker, linker_dict in linker_data.items():
            cleavage = linker_dict[7]
            #n1 = sum(list(linker_dict.values()))
            n1 = linker_dict[6] + linker_dict[8] + cleavage + linker_dict[5] + linker_dict[9]
            prob = (cleavage/n1)*100
            d.append(round(prob, 2))
            print(linker, prob)
        print(d)

    if score_set:
        conditions = {"Description": None, "Parent Protein IRI": None, 
                      "Method/Technique": ("contains", "mass spectrometry"), 
                      "MHC allele class": ("match", "I"),
                      "Cell Type": ("match", "B cell")}
        export_set_path = "data/score_set/class_{0}/class_{0}_{1}_data.csv".format(conditions["MHC allele class"][1], "Individual")
        proteasome_dictionary = generating_data(iedb_path, uniprot_path, conditions)
        scoring_data_generator.generating_scoring_data(proteasome_dictionary, export_set_path)
        predictor_set.proteasome_prediction(export_set_path, models_export_path, "Individual", "set")
    
    if score_set_all:
        conditions = {"Description": None, "Parent Protein IRI": None,
                          "Method/Technique": ("contains", "mass spectrometry"),
                          "MHC allele class": ("match", "I"),
                          "Cell Type": ("match", "B cell")}
        export_set_path = "data/score_set/class_{0}/class_{0}_{1}_data.csv".format(conditions["MHC allele class"][1], "All")
        proteasome_dictionary = generating_data(iedb_path, uniprot_path, conditions)
        scoring_data_generator.generating_scoring_data(proteasome_dictionary, export_set_path)
        predictor_set_all.proteasome_prediction(export_set_path, models_export_path, "All", "set_all")

if __name__ == "__main__":
    generate, train, train_all, predict_fasta, score_set, score_set_all = parse_args()
    main(generate, train, train_all, predict_fasta, score_set, score_set_all)
