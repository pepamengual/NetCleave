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

> <p align="justify"> Amengual-Rigo, P., Guallar, V. NetCleave: an open-source algorithm for predicting C-terminal antigen processing for MHC-I and MHC-II. Sci Rep 11, 13126 (2021). https://doi.org/10.1038/s41598-021-92632-y
</p>

Install NetCleave dependencies by:

```
pip install -r requirements.txt
```

## How to use NetCleave

<p align="justify">
NetCleave is easy to use. It can be run using a python command-line (which can be stored in a bash script).

```bash
netcleave_path="NetCleave.py"  # path to NetCleave.py
mhc_class="I"                  # MHC class to be modelled
technique="mass_spectrometry"  # Technique to be modelled
mhc_family="HLA"               # MHC family to be modelled
fasta="example/example.fasta"  # Input fasta file

python $netcleave_path \
        --generate \
        --train \
        --mhc_class $mhc_class \
        --technique $technique \
        --mhc_family $mhc_family \
        --score_fasta $fasta
```

The arguments --generate and --train are needed for creating the model for the first time.

</p>

### Retraining the method and constructing your own models

<p align="justify">
NetCleave was specificaly designed to be easily retrained. In order to do so, user needs:

- Decompress or obtain databases
> IEDB - data/databases/iedb folder. New versions may be found at: http://www.iedb.org/database_export_v3.php

> UniProt/UniParc - data/databases/uniprot and data/databases/uniparc folders. New versions may be found at: https://www.uniprot.org/downloads

**We recomend to use the last version of IEDB while keeping the same UniProt/UniParc version that we provided in this repository**

- Define IEDB, UniProt and UniParc file paths on NetCleave main function

