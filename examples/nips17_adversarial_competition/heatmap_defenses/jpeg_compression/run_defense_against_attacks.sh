#!/bin/bash

checkpoint_path=$1
net_type=$2
results_name=$3
images_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/attack_images"
attacks=( "fgsm" "noop" "random_noise" "iter_target_class" "step_target_class" )
results_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/heatmap_defenses/jpeg_compression/results/""${results_name}""/""${interp}"
#epsilons=( 1 2 4 8 12 16 )
epsilons=( 16 )
drs=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 )
#crs=( 5 10 15 20 30 40 50 60 70 80 90 )
crs=( 5 10 15 20 )

mkdir "${results_dir}"
for attack in "${attacks[@]}"
do
    mkdir "${results_dir}""/""${attack}" 
    for eps in "${epsilons[@]}"
    do
        in_dir="${images_dir}""/""${attack}""/eps_""${eps}"
        out_dir="${results_dir}""/""${attack}""/eps_""${eps}"
        mkdir "${out_dir}"
        for cr in "${crs[@]}"
        do
            for dr in "${drs[@]}"
            do
                out_file="${out_dir}""/cr_""${cr}""_dr_""${dr}"".csv"
                #out_file="${out_dir}""/cr_""${cr}"".csv"
                echo "${out_file}"
                sh run_defense.sh "${in_dir}" "${out_file}" "${cr}" "${dr}" "${checkpoint_path}" "${net_type}"
                #sh run_defense.sh "${in_dir}" "${out_file}" "${cr}" "${checkpoint_path}" "${net_type}"
            done
        done
    done
done
