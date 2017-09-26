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
#CHECKPOINT_PATH="ens_adv_inception_resnet_v2.ckpt"
#NET_TYPE="resnet"
DOWNSAMPLE=$3
INTERP=$4
CHECKPOINT_PATH=$5
NET_TYPE=$6

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --net_type="${NET_TYPE}" \
  --downsample="${DOWNSAMPLE}" \
  --interp="${INTERP}" \
  --gpu=0
  #--using_docker

