#!/bin/bash

images_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/attack_images"
attacks=( "fgsm" "noop" "random_noise" "iter_target_class" "step_target_class" )
results_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/sample_defenses/adv_inception_v3/results/"
epsilons=( 1 2 4 8 12 16 )

mkdir "${results_dir}"
for attack in "${attacks[@]}"
do
    mkdir "${results_dir}""/""${attack}" 
    for eps in "${epsilons[@]}"
    do
        in_dir="${images_dir}""/""${attack}""/eps_""${eps}"
        out_file="${results_dir}""/""${attack}""/eps_""${eps}"".csv"
        echo "${out_file}"
        sh run_defense.sh "${in_dir}" "${out_file}" 
    done
done
