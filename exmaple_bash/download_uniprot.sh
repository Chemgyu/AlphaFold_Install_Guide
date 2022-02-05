#!/bin/bash
# Description: Downloads and unzips all required data for AlphaFold2 (AF2).
# Author: Sanjay Kumar Srikakulam

# Since some parts of the script may resemble AlphaFold's download scripts copyright and License notice is added.
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

set -e

# Input processing
usage() {
        echo ""
        echo "Please make sure all required parameters are given"
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-d <download_dir>     Absolute path to the AF2 download directory (example: /home/johndoe/alphafold_data)"
        echo "Optional Parameters:"
        echo "-m <download_mode>    full_dbs or reduced_dbs mode [default: full_dbs]"
        echo ""
        exit 1
}

while getopts ":d:m:" i; do
        case "${i}" in
        d)
                download_dir=$OPTARG
        ;;
        m)
                download_mode=$OPTARG
        ;;
        esac
done

if [[  $download_dir == "" ]]; then
    usage
fi

if [[  $download_mode == "" ]]; then
    download_mode="full_dbs"
fi

if [[ $download_mode != "full_dbs" && $download_mode != "reduced_dbs" ]]; then
    echo "Download mode '$download_mode' is not recognized"
    usage
fi

# Check if rsync, wget, gunzip and tar command line utilities are available
check_cmd_line_utility(){
    cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        echo "Command line utility '$cmd' could not be found. Please install."
        exit 1
    fi    
}

check_cmd_line_utility "wget"
check_cmd_line_utility "rsync"
check_cmd_line_utility "gunzip"
check_cmd_line_utility "tar"

# Make AF2 data directory structure
params="$download_dir/params"
mgnify="$download_dir/mgnify"
pdb70="$download_dir/pdb70"
pdb_mmcif="$download_dir/pdb_mmcif"
mmcif_download_dir="$pdb_mmcif/data_dir"
mmcif_files="$pdb_mmcif/mmcif_files"
uniclust30="$download_dir/uniclust30"
uniref90="$download_dir/uniref90"
uniprot="$download_dir/uniprot"
pdb_seqres="$download_dir/pdb_seqres"

download_dir=$(realpath "$download_dir")
mkdir --parents "$download_dir"

# Download Uniclust30 database
echo "Downloading Uniclust30 database"
uniclust30_filename="uniclust30_2018_08_hhsuite.tar.gz"
wget -c -P "$uniclust30" "https://storage.googleapis.com/alphafold-databases/casp14_versions/${uniclust30_filename}"
tar --extract --verbose --file="$uniclust30/$uniclust30_filename" --directory="$uniclust30"
rm "$uniclust30/$uniclust30_filename"

# Download Uniref90 database
echo "Downloading Unifef90 database"
uniref90_filename="uniref90.fasta.gz"
wget -c -P "$uniref90" "ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/${uniref90_filename}"
(cd "$uniref90" && gunzip "$uniref90/$uniref90_filename")

# Download Uniprot database
echo "Downloading Uniprot (TrEMBL and Swiss-Prot) database"
trembl_filename="uniprot_trembl.fasta.gz"
trembl_unzipped_filename="uniprot_trembl.fasta"
wget -c -P "$uniprot" "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/${trembl_filename}"
(cd "$uniprot" && gunzip "$uniprot/$trembl_filename")

sprot_filename="uniprot_sprot.fasta.gz"
sprot_unzipped_filename="uniprot_sprot.fasta"
wget -c -P "$uniprot" "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/${sprot_filename}"
(cd "$uniprot" && gunzip "$uniprot/$sprot_filename")

# Concatenate TrEMBL and Swiss-Prot, rename to uniprot and clean up.
cat "$uniprot/$sprot_unzipped_filename" >> "$uniprot/$trembl_unzipped_filename"
mv "$uniprot/$trembl_unzipped_filename" "$uniprot/uniprot.fasta"
rm "$uniprot/$sprot_unzipped_filename"

# Download PDB seqres database
wget -c -P "$pdb_seqres" "ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"

echo "All Uniprot data is downloaded"
