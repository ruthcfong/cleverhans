#/bin/bash

#attacks=( "fgsm" "noop" "random_noise" )
attacks=( "step_target_class" "iter_target_class" )
image_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/images"
#attack_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/sample_attacks/"
attack_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/sample_targeted_attacks/"
out_base_dir="/home/ruthfong/tensorflow/cleverhans/examples/nips17_adversarial_competition/dataset/attack_images/"
epsilons=( 1 2 4 8 12 16 )

for attack in "${attacks[@]}"
do
    mkdir "${out_base_dir}""${attack}"
    cd "${attack_dir}""${attack}"
    for eps in "${epsilons[@]}" 
    do
       out_dir="${out_base_dir}""${attack}""/eps_""${eps}" 
       mkdir "${out_dir}"
       pwd
       sh run_attack.sh "${image_dir}" "${out_dir}" "${eps}" 
    done
done
