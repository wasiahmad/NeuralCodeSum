#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp

make_dir $MODEL_DIR

DATASET=java

function generate () {

echo "============Generating (Beam)============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/test.py \
--only_generate True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src $3 \
--uncase True \
--max_examples -1 \
--max_src_len 150 \
--max_tgt_len 50 \
--test_batch_size 64 \
--beam_size 4 \
--n_best 1 \
--block_ngram_repeat 3 \
--stepwise_penalty False \
--coverage_penalty none \
--length_penalty none \
--beta 0 \
--gamma 0 \
--replace_unk

}

# run: bash generate.sh 0 MODEL_NAME source_code_filename
generate $1 $2 $3
