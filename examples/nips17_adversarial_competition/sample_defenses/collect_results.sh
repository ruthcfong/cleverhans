#!/bin/bash

attacks=( "fgsm" "noop" "random_noise" "iter_target_class" "step_target_class" )
results_dir=$1
#results_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/heatmap_defenses/denoise/results/nearest"
results_file=$2
#results_file="results_nearest.csv"
epsilons=( 1 2 4 8 12 16 )                                                     


for attack in "${attacks[@]}"                                                   
do                                                                              
    for eps in "${epsilons[@]}"                                                 
    do                                                                          
        class_file="${results_dir}""/""${attack}""/eps_""${eps}"".csv" 
        result="$(python ../heatmap_defenses/calculate_classification_accuracy.py $class_file)" 
        echo $attack","$eps","$result >> $results_file 
    done                                                                        
done                                                                            

