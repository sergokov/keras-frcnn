#!/usr/bin/env bash

annotation_data_path=$1
input_img_path=$2
out_img_path=$3

annotations=($(cat $annotation_data_path))

for annotation in ${annotations[@]}
do
  cp $input_img_path/$annotation.jpg $out_img_path/$annotation.jpg
done