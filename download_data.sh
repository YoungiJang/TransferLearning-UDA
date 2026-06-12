#!/bin/bash
# Download & extract the datasets for DL_HW3 (UDA).
# Recreates the layout expected by DL_HW3.py:
#   ./CUB_200_2011/images/   (source/target: real bird photos, 200 classes)
#   ./CUB-200-Painting/      (source/target: bird paintings, 200 classes)
#
# Usage:  bash download_data.sh
# Requires: wget, tar, unzip, and gdown (pip install gdown) for the Google Drive zip.

set -euo pipefail
cd "$(dirname "$0")"

CUB_URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
PAINT_GDRIVE_ID="1G327KsD93eyGTjMmByuVy9sk4tlEOyK3"

# 1. CUB-200-2011 (real photos) -> CUB_200_2011/images
if [ ! -d "CUB_200_2011/images" ]; then
    if [ ! -f "CUB_200_2011.tgz" ]; then
        echo ">>> Downloading CUB-200-2011 (~1.1GB)..."
        wget -O CUB_200_2011.tgz "$CUB_URL"
    fi
    echo ">>> Extracting CUB-200-2011..."
    tar -xzf CUB_200_2011.tgz
else
    echo ">>> CUB_200_2011/images already exists. Skipping."
fi

# 2. CUB-200-Paintings -> CUB-200-Painting
if [ ! -d "CUB-200-Painting" ]; then
    if [ ! -f "CUB_200_Paintings.zip" ]; then
        echo ">>> Downloading CUB-200-Paintings (Google Drive)..."
        gdown "https://drive.google.com/uc?id=${PAINT_GDRIVE_ID}" -O CUB_200_Paintings.zip
    fi
    echo ">>> Unzipping CUB-200-Paintings..."
    unzip -q -o CUB_200_Paintings.zip
else
    echo ">>> CUB-200-Painting already exists. Skipping."
fi

echo ">>> Done. Datasets ready:"
echo "    CUB_200_2011/images : $(ls CUB_200_2011/images 2>/dev/null | wc -l) classes"
echo "    CUB-200-Painting    : $(find CUB-200-Painting -maxdepth 1 -type d | tail -n +2 | wc -l) classes"
