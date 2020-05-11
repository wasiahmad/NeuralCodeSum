#!/usr/bin/env bash

echo "Downloading python-method dataset"
FILE=python.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    # https://drive.google.com/open?id=1XPE1txk9VI0aOT_TdqbAeI58Q8puKVl2
    fileid="1XPE1txk9VI0aOT_TdqbAeI58Q8puKVl2"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi

echo "Aggregating statistics of the dataset"
python get_stat.py
