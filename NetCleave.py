import argparse
import gzip
import shutil
import os
from predictor.database_functions import peptide_extractor, uniprot_extractor, uniparc_extractor
from predictor.core import all_peptide_uniprot_locator, all_training_data_generator
from predictor.ml_main import run_NN
from predictor.predictions import predict_csv_or_fasta


def parse_args():
    package_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Trains and runs NetCleave")

    parser.add_argument("--mhc_class",
                        dest="mhc_class",
                        type=str,
                        help="MHC class I or II",
                        choices=["I", "II"],
                        default="I"
                        )

    parser.add_argument("--technique",
                        dest="technique",
                        type=str,
                        help="Technique to focus the predictions",
                        choices=["mass_spectrometry", "radioactivity", "fluorescence"],
                        default="mass_spectrometry"
                        )

    parser.add_argument("--mhc_family",
                        dest="mhc_family",
                        type=str,
                        help="MHC family or allele name. Partial names are allowed (e.g., HLA-A or HLA-A*02)",
                        default="HLA-A*02:01"
                        )

    parser.add_argument("--iedb_path",
                        dest="iedb_path",
                        type=str,
                        help="Path to IEDB file;"
                             "download and unzip from http://www.iedb.org/database_export_v3.php",
                        default=f"{package_dir}/data/databases/iedb/mhc_ligand_full.csv"
                        )

    parser.add_argument("--uniprot_path",
                        dest="uniprot_path",
                        type=str,
                        help="Path to UniProt file;"
                             "download and decompress from https://www.uniprot.org/downloads -> REVIEWED -> fasta",
                        default=f"{package_dir}/data/databases/uniprot/uniprot_sprot.fasta"
                        )

    parser.add_argument("--uniparc_path_headers",
                        dest="uniparc_path_headers",
                        type=str,
                        help="Path to UniParc headers file",
                        default=f"{package_dir}/data/databases/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.tab"
                        )

    parser.add_argument("--uniparc_path_sequence",
                        dest="uniparc_path_sequence",
                        type=str,
                        help="Path to UniParc sequence file",
                        default=f"{package_dir}/data/databases/uniparc/uniparc-yourlist_M20200416A94466D2655679D1FD8953E075198DA854EB3ES.fasta"
                        )

    parser.add_argument("--qsar_table",
                        dest="qsar_table",
                        type=str,
                        help="Path to the QSAR table file",
                        default=f"{package_dir}/predictor/ml_main/QSAR_table.csv"
                        )

    parser.add_argument('--generate',
                        help='Generate training data for the neural network',
                        action='store_true'
                        )

    parser.add_argument('--train',
                        help='Train the neural network',
                        action='store_true'
                        )

    parser.add_argument('--score_csv',
                        dest="score_csv",
                        type=str,
                        help='Predict a set of cleavage sites from csv (7 residues)',
                        default=None
                        )

    parser.add_argument('--score_fasta',
                        dest="score_fasta",
                        type=str,
                        help='Predict all cleavage sites in a fasta',
                        default=None
                        )

    args = parser.parse_args()
    return args


def decompress_databases(database_list):
    for database_file in database_list:
        if not os.path.exists(database_file):
            database_file_compressed = f"{database_file}.gz"
            with gzip.open(database_file_compressed, "r") as f_in, open(database_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def generating_data(iedb_path, uniprot_path, uniparc_path_headers, uniparc_path_sequence, conditions):
    iedb_data = peptide_extractor.extract_peptide_data(iedb_path, conditions)
    uniprot_data = uniprot_extractor.extract_uniprot_data(uniprot_path)
    uniparc_data = uniparc_extractor.extract_uniparc_data(uniparc_path_headers, uniparc_path_sequence)
    sequence_data = all_peptide_uniprot_locator.join_data(uniprot_data, uniparc_data)
    data = all_peptide_uniprot_locator.locate_peptides(iedb_data, sequence_data)
    return data


def main():
    args = parse_args()
    args.technique = args.technique.replace("_", " ")
    technique_name = args.technique.replace(" ", "-")
    mhc_family_name = args.mhc_family.replace("*", "").replace(":", "")

    database_list = [args.iedb_path, args.uniprot_path, args.uniparc_path_headers, args.uniparc_path_sequence]
    decompress_databases(database_list)

    training_data_path = f"data/training_data/{args.mhc_class}_{technique_name}_{mhc_family_name}"
    models_export_path = f"data/models/{args.mhc_class}_{technique_name}_{mhc_family_name}"

    if args.generate:
        conditions = {"Method/Technique": ("contains", args.technique),
                      "MHC allele class": ("match", args.mhc_class),
                      "Allele Name": ("contains", args.mhc_family)}

        data = generating_data(args.iedb_path, args.uniprot_path, args.uniparc_path_headers,
                               args.uniparc_path_sequence, conditions)
        all_training_data_generator.prepare_cleavage_data(data, training_data_path)

    if args.train:
        run_NN.create_models(training_data_path, models_export_path, args.qsar_table)

    if args.score_csv:
        predict_csv_or_fasta.score_set(args.score_csv, models_export_path, args.qsar_table)

    if args.score_fasta:
        predict_csv_or_fasta.score_set(args.score_fasta, models_export_path, args.qsar_table)


if __name__ == "__main__":
    main()
