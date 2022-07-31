for dataset in AMG1608 DEAM PmEmo; do
    for source in reddit twitter youtube; do
        wordbag_features -i /mnt/f/fix_data/$dataset/$source -w All --source $source --dataset $dataset
    done
done

for dataset in Deezer; do
    for source in reddit twitter; do
        wordbag_features -i /mnt/f/fix_data/$dataset/$source -w All --source $source --dataset $dataset
    done
done

for dataset in AMG1608 DEAM PmEmo Deezer; do
    wordbag_features -i /mnt/f/fix_data/$dataset/$source -w All --source All --dataset $dataset
done