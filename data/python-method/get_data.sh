#!/usr/bin/env bash

echo "Downloading python-method dataset"
FILE=python-method.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    fileid="1qM7PGoJ48HfPUi4dcrqrJA2g7QgxzGFk"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi

echo "Aggregating statistics of the dataset"
python get_stat.py
