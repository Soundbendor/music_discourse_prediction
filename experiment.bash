for dataset in AMG1608 DEAM PmEmo; do
    for source in reddit twitter youtube; do
        wordbag_features -i /mnt/f/last_ditch_effort/$dataset/$source -w All --source $source --dataset $dataset
    done
done