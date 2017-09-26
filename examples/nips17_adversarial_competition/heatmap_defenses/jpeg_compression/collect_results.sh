#!/bin/bash

attacks=( "fgsm" "noop" "random_noise" "iter_target_class" "step_target_class" )
results_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/heatmap_defenses/jpeg_compression/results/base_inception_model"
results_file="results_base_inception_model_grid_search.csv"
#epsilons=( 1 2 4 8 12 16 )                                                     
epsilons=( 16 )
drs=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 )                                       
#crs=( 10 20 30 40 50 60 70 80 90 )
crs=( 5 10 15 20 )
#scales=( 0.1 0.2 0.3 0.4 )


for attack in "${attacks[@]}"                                                   
do                                                                              
    for eps in "${epsilons[@]}"                                                 
    do                                                                          
        out_dir="${results_dir}""/""${attack}""/eps_""${eps}"                   
        for cr in "${crs[@]}"                                             
        do                                                                      
            for dr in "${drs[@]}"
            do
                #class_file="${out_dir}""/cr_""${cr}"".csv"                 
                class_file="${out_dir}""/cr_""${cr}""_dr_""${dr}"".csv"
                result="$(python ../calculate_classification_accuracy.py 1 $class_file)" 
                echo $attack","$eps","$cr",""$dr",$result >> $results_file 
            done
        done                                                                    
    done                                                                        
done                                                                            

