#!/bin/bash

annotation_data_path=$1

out_path=$2

trainval="$out_path/trainval.txt"
val="$out_path/val.txt"
train="$out_path/train.txt"

annotations=($(ls $annotation_data_path))

for annotation_path in ${annotations[@]}
do
  echo $annotation_path | cut -d'.' -f1 >> $trainval
done

ann_len=${#annotations[@]}

echo "$ann_len"

((num_val=ann_len*20/100))
((num_train=ann_len-num_val))

echo "train/val/tot = ${num_train}/${num_val}/${ann_len}"

head -n $num_train $trainval > $train

tail -n $num_val $trainval  > $val