#!/bin/bash

attacks=( "fgsm" "noop" "random_noise" "iter_target_class" "step_target_class" )
results_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/heatmap_defenses/denoise/results/adv_inception_v3/bilinear"
results_file="results_adv_inception_v3_bilinear.csv"
epsilons=( 1 2 4 8 12 16 )                                                     
scales=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 )                                       
#scales=( 0.1 0.2 0.3 0.4 )


for attack in "${attacks[@]}"                                                   
do                                                                              
    for eps in "${epsilons[@]}"                                                 
    do                                                                          
        out_dir="${results_dir}""/""${attack}""/eps_""${eps}"                   
        for scale in "${scales[@]}"                                             
        do                                                                      
            class_file="${out_dir}""/downsample_""${scale}"".csv"                 
            result="$(python ../calculate_classification_accuracy.py $class_file)" 
            echo $attack","$eps","$scale","$result >> $results_file 
        done                                                                    
    done                                                                        
done                                                                            

