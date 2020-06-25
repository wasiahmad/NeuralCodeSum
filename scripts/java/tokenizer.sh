#!/usr/bin/env bash

SRC_DIR=../..

function tokenize_data () {
FILE_TYPE=$1
FILE_PATH=$2
PYTHONPATH=$SRC_DIR python -W ignore ${SRC_DIR}/tokenizer/java/java_tokenizer.py \
-f $FILE_TYPE \
-p ${SRC_DIR}/$FILE_PATH \

}

tokenize_data $1 $2