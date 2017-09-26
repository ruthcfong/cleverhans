#!/bin/bash

interp=$1
checkpoint_path=$2
net_type=$3
results_name=$4
images_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/attack_images"
attacks=( "fgsm" "noop" "random_noise" "iter_target_class" "step_target_class" )
results_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/heatmap_defenses/denoise/results/""${results_name}""/""${interp}"
epsilons=( 1 2 4 8 12 16 )
scales=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 )

mkdir "${results_dir}"
for attack in "${attacks[@]}"
do
    mkdir "${results_dir}""/""${attack}" 
    for eps in "${epsilons[@]}"
    do
        in_dir="${images_dir}""/""${attack}""/eps_""${eps}"
        out_dir="${results_dir}""/""${attack}""/eps_""${eps}"
        mkdir "${out_dir}"
        for scale in "${scales[@]}"
        do
            out_file="${out_dir}""/downsample_""${scale}"".csv"
            echo "${out_file}"
            sh run_defense.sh "${in_dir}" "${out_file}" "${scale}" "${interp}" "${checkpoint_path}" "${net_type}"
        done
    done
done
