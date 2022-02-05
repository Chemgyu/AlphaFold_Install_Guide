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

# Download PDB obsolete data
wget -P "$pdb_mmcif" "ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat"

# Download PDB mmCIF database
echo "Downloading PDB mmCIF database"
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ "$mmcif_download_dir"
find "$mmcif_download_dir/" -type f -iname "*.gz" -exec gunzip {} +
find "$mmcif_download_dir" -type d -empty -delete

for sub_dir in "$mmcif_download_dir"/*; do
  mv "$sub_dir/"*.cif "$mmcif_files"
done

find "$mmcif_download_dir" -type d -empty -delete

echo "mmcif data is downloaded"
