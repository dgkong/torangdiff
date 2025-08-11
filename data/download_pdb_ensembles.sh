#!/bin/bash

# Script to read a text file containing comma-separated PDB IDs, where each line represents an ensemble.
# Downloads corresponding files from RCSB

INPUT_FILE="data/pdb_ensembles.txt"
OUTPUT_DIR="data/pdb_ensemble_downloads"
PARALLEL_JOBS=8

mkdir -p "$OUTPUT_DIR"

echo "Generating a list of all files to download"
DOWNLOAD_LIST_FILE=$(mktemp)
ENSEMBLE_NUM=0
while read -r line; do
    ((ENSEMBLE_NUM++))
    ENSEMBLE_DIR="$OUTPUT_DIR/ensemble_$ENSEMBLE_NUM"
    mkdir -p "$ENSEMBLE_DIR"

    IFS=',' read -ra full_ids_array <<< "$line"

    for full_id in "${full_ids_array[@]}"; do
        pdb_id=${full_id%_*}
        URL="https://files.rcsb.org/download/${pdb_id}.cif"
        OUTPUT_PATH="$ENSEMBLE_DIR/${full_id}.cif"
        if [ ! -s "$OUTPUT_PATH" ]; then
            echo "$OUTPUT_PATH $URL" >> "$DOWNLOAD_LIST_FILE"
        fi
    done
done < "$INPUT_FILE"

echo "Starting parallel download of $(wc -l < $DOWNLOAD_LIST_FILE) files"
cat "$DOWNLOAD_LIST_FILE" | xargs -n 2 -P $PARALLEL_JOBS bash -c 'wget -q -O "$1" "$2"' _
rm "$DOWNLOAD_LIST_FILE"

echo "Download complete"
