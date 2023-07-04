

netcleave_path="NetCleave.py"  # path to NetCleave.py
mhc_class="I"                  # MHC class to be modelled
technique="mass_spectrometry"  # Technique to be modelled
mhc_family="HLA" # DRB1        # MHC family to be modelled
fasta="example/example.fasta"  # Input fasta file


# Add --generate and --generate flags to train the model

# Predict only
python $netcleave_path \
        --mhc_class $mhc_class \
        --technique $technique \
        --mhc_family $mhc_family \
        --score_fasta $fasta
