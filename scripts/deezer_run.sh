#!/bin/bash
for dataset in amg1608 deam_new pmemo; do
    for source in Reddit Youtube Twitter; do 
        bert_features --source $source --dataset $dataset --batch_size 128 --epochs 2 --make_csv True
    done
done
