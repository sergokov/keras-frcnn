#!/bin/bash

annotation_data_path=$1

out_path=$2

trainval="$out_path/trainval.txt"

annotations=($(ls $annotation_data_path))

for annotation_path in ${annotations[@]}
do
  echo $annotation_path | cut -d'.' -f1 >> $trainval
done

ann_len=${#annotations[@]}

echo "$ann_len"

((num_val=ann_len*20/100))
((num_train=ann_len-num_val))

#echo "train/val/tot = ${num_train}/${num_val}/${ann_len}"

#echo "${strHeader}" > $foutTrain
###:> $foutTrain
#cat $finpShuf | head -n $numTrain >> $foutTrain
#
#echo "${strHeader}" > $foutVal
###:> $foutVal
#cat $finpShuf | tail -n $numVal   >> $foutVal


