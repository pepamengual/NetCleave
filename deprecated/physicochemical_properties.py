from predictor.iedb_functions import ms_extractor
from predictor.uniprot_functions import uniprot_extractor
from predictor.general import save_pickle
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import cleavage_probabilities
from predictor.general import save_file
import numpy as np
from predictor.physicochemical_properties import distribution_cleavage
from predictor.physicochemical_properties import frequency_cleavage
from predictor.physicochemical_properties import random_peptide_generator
from predictor.physicochemical_properties import score_peptides
import matplotlib.pyplot as plt
from predictor.physicochemical_properties import plot_histogram
from predictor.physicochemical_properties import score_peptides_same_time

def main():
    print("Extracting MS data from IEDB...")
    iedb_data_file_raw_path = "../data/raw/iedb/mhc_ligand_full.csv"
    iedb_data = ms_extractor.extract_ms_data(iedb_data_file_raw_path)
    
    print("Extracting UniProt data...")
    uniprot_data_file_raw_path = "../data/raw/uniprot/uniprot_sprot.fasta"
    uniprot_data = uniprot_extractor.id_sequence_extractor(uniprot_data_file_raw_path)
    
    print("Seeking for MS peptides into UniProt data...")
    n = 5
    large_uniprot_peptide = seek_ms_uniprot.seeking_ms(iedb_data, uniprot_data, n)
    
    erap_cleavage = [n - 3, n - 2, n - 1, n, n + 1, n + 2]
    proteasome_cleavage = [(n + 3) * -1, (n + 2) * -1, (n + 1) * -1, (n) * -1, (n - 1) * -1, (n - 2) * -1]
    
    erap_cleavage = [n - 2, n - 1, n, n + 1, n + 2]
    proteasome_cleavage = [(n + 3) * -1, (n + 2) * -1, (n + 1) * -1, (n) * -1, (n - 1) * -1]

    erap_cleavage = [n -3, n - 2, n - 1, n, n + 1]
    proteasome_cleavage = [(n + 2) * -1, (n + 1) * -1, (n) * -1, (n - 1) * -1, (n - 2) * -1]

    uniprot_probabilities = {'A': 0.08258971312579017, 'C': 0.013826094946210853, 'D': 0.054625650802595425, 'E': 0.0673214708897148, 'F': 0.03866188645338429, 'G': 0.07077863527330625, 'H': 0.022761656446475265, 'I': 0.05923828965491043, 'K': 0.05815460235107699, 'L': 0.09655733034859719, 'M': 0.024154886555486327, 'N': 0.04061129236837406, 'P': 0.047331721936265635, 'Q': 0.03932403048405303, 'R': 0.05534153979141534, 'S': 0.06631318414876945, 'T': 0.05355909368186356, 'V': 0.06865326331945962, 'W': 0.010987143802538912, 'Y': 0.029208513619712422}

    properties = {"A": "UAS", "C": "UAM", "D": "-PM", "E": "-PM", "F": "UHL", "G": "UAS", "H": "YYY", "I": "UAM", "K": "+PL", "L": "UAM", "M": "UHL", "N": "UPM", "P": "XXX", "Q": "UPM", "R": "+PL", "S": "UPS", "T": "UPS", "V": "UAM", "W": "UHL", "Y": "UHL"}
    
    

    random_peptides = random_peptide_generator.generate_random_peptides(large_uniprot_peptide, uniprot_probabilities)

    print("ERAP")
    erap_counts = distribution_cleavage.distribution_cleavage(large_uniprot_peptide, erap_cleavage, properties)
    probability_data_erap = frequency_cleavage.probability_cleavage(erap_counts, properties, uniprot_probabilities)
    
    ms_erap_scores = score_peptides.score_peptides(large_uniprot_peptide, erap_cleavage, properties, probability_data_erap)
    random_erap_scores = score_peptides.score_peptides(random_peptides, erap_cleavage, properties, probability_data_erap)
    plot_histogram.histogram_plotter(ms_erap_scores, random_erap_scores, "ERAP")

    print("PROTEASOME")
    proteasome_counts = distribution_cleavage.distribution_cleavage(large_uniprot_peptide, proteasome_cleavage, properties)
    probability_data_proteasome = frequency_cleavage.probability_cleavage(proteasome_counts, properties, uniprot_probabilities)
    
    ms_proteasome_scores = score_peptides.score_peptides(large_uniprot_peptide, proteasome_cleavage, properties, probability_data_proteasome)
    random_proteasome_scores = score_peptides.score_peptides(random_peptides, proteasome_cleavage, properties, probability_data_proteasome)
    plot_histogram.histogram_plotter(ms_proteasome_scores, random_proteasome_scores, "PROTEASOME")

    print("ERAP+PROTEASOME")
    ms_erap_proteasome_scores = score_peptides_same_time.score_peptides(large_uniprot_peptide, erap_cleavage, proteasome_cleavage, properties, probability_data_erap, probability_data_proteasome)
    random_erap_proteasome_scores = score_peptides_same_time.score_peptides(random_peptides, erap_cleavage, proteasome_cleavage, properties, probability_data_erap, probability_data_proteasome)
    plot_histogram.histogram_plotter(ms_erap_proteasome_scores, random_erap_proteasome_scores, "ERAP_PROTEASOME")


if __name__ == "__main__":
    main()
