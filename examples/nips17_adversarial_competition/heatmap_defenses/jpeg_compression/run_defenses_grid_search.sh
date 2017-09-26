img_dir="../../dataset/attack_images/fgsm/eps_16"
out_dir="grid_search"
downsample_rates=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 )
#compression_rates=( 10 20 30 40 50 60 80 90 )
compression_rates=( 70 )

for dr in "${downsample_rates[@]}"
do
    for cr in "${compression_rates[@]}"
    do
        out_f="${out_dir}""/cr_""${cr}""_dr_""${dr}"".csv"
        sh run_defense.sh $img_dir $out_f $cr $dr
    done
done
