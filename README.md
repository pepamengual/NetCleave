# NetCleave

<p align="justify">
NetCleave is a retrainable method for predicting C-terminal peptide processing of MHC-I and MHC-II pathways.
</p>

<p align="center">
<img src="images/draw_scheme_method.png" width="600">
</p>

<p align="justify">
In brief, NetCleave maps reported IEDB peptides to protein sequences in UniProt/UniParc. After the identification of the C-terminal cleavage site, amino acid sequences are coded using QSAR descriptors, including steric, electrostatic and hydrophobic properties. Finally, a neural network architecture is used to generate the predictive model.
</p>

If you use NetCleave, please cite us:

> NetCleave: an open-source algorithm for predicting C-terminal antigen processing for MHC-I and MHC-II (manuscript in submission)

NetCleave has the following dependencies:

- [argparse](https://docs.python.org/3/library/argparse.html)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [pathlib](https://docs.python.org/3/library/pathlib.html)
- [sklearn](https://scikit-learn.org/stable/)
- [keras](https://keras.io/)
- [tensorflow](https://www.tensorflow.org/)

## How to use NetCleave

<p align="justify">
NetCleave is very easy to use. It has three main functions:

- **generate** - gets C-terminal data from IEDB and UniProt/UniParc.
- **train** - runs the neural network and saves weights.
- **predict_csv** - scores C-terminal sites from a csv file.

Users can choose between using NetCleave **pre-trained models** or **easily retraining them**.
</p>

### Using pre-trained models

<p align="justify">
In order to use NetCleave, user needs to define a few parameters:

- HLA class of interest
> Either I or II - **mhc_class variable on NetCleave main function**.
- HLA family of interest
> Several pre-trained models are available, which should cover most of the needs of the scientific community. This includes models for *HLA-A*, *HLA-B*, *HLA-C*, *HLA-DP*, *HLA-DQ*, *HLA-DP*, *H2-Kb*, *H2-Db*, *HLA-A02:01*, *HLA-B07:02*, and others. Check data/models folder - **mhc_family variable on NetCleave main function**.

NetCleave can predict the cleavage probability of a C-terminal site, which we defined as:

> **Four (4) last amino acids of a peptide + three (3) following amino acids in sequence**

User needs to define **a sequence of seven (7) residues lenght** following the previous scheme, and add them into a csv file (column name: "sequence"). An example can be found on the "*example_file_NetCleave_score.csv*" file. The command to score cvs files is the following:

> python3 NetCleave.py --score_csv

After running this command, a csv file with the results will be generated.
</p>

### Retraining the method and constructing your own models

NetCleave was specificaly build to be easily retrained: IEDB is continously being updated and algorithms should be periodically updated. In order to retrain NetCleave method, user needs first to download the last versions of IEDB and UniParc/UniProt.

