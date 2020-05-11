#!/usr/bin/env bash

echo "Downloading TL-CodeSum dataset"
FILE=java.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    # https://drive.google.com/open?id=13o4MiELiQoomlly2TCTpbtGee_HdQZxl
    fileid="13o4MiELiQoomlly2TCTpbtGee_HdQZxl"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi

echo "Aggregating statistics of the dataset"
python get_stat.py
