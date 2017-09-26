#!/bin/bash

#attacks=( "fgsm" "noop" "random_noise" "iter_target_class" "step_target_class" )
results_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/heatmap_defenses/denoise/results/adv_inception_v3/bilinear"
results_dir="grid_search"
results_file="grid_search_fgsm_inception.csv"
crs=( 10 20 30 40 50 60 70 80 90 )
drs=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 )                                       

for dr in "${drs[@]}"
do                                                                      
    for cr in "${crs[@]}"
    do
        class_file="${results_dir}""/cr_""${cr}""_dr_""${dr}"".csv"                 
        result="$(python ../calculate_classification_accuracy.py 1 $class_file)" 
        echo $dr","$cr","$result >> $results_file 
    done
done

