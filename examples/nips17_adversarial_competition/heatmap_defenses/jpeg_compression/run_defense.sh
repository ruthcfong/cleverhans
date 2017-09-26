#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#

INPUT_DIR=$1
OUTPUT_FILE=$2
COMPRESSION_RATE=$3
DOWNSAMPLE_RATE=$4
CHECKPOINT_PATH=$5
NET_TYPE=$6
#DOWNSAMPLE_RATE=$4
#CHECKPOINT_PATH="inception_v3.ckpt"
#NET_TYPE="googlenet"

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --compression_rate="${COMPRESSION_RATE}" \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --net_type="${NET_TYPE}" \
  --gpu=0 \
  --downsample_rate="${DOWNSAMPLE_RATE}" \
  --downsample
  #--using_docker

