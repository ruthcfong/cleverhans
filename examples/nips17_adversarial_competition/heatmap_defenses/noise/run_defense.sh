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
NOISE_STD=0.25

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_path=inception_v3.ckpt \
  --using_docker \
  --noise_std="${NOISE_STD}"

