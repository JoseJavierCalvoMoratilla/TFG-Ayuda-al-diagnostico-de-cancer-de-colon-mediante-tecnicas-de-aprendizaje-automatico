#!/bin/bash

#n=0
#while [ ${n} -lt 500 ]
#do
#    n=$(wc -l evolution_iou_model_1a_sigmoid_in_training.txt | awk '{print $1}')
#    echo "n = ${n}, waiting till 500"
#    sleep 60
#done

model_id='1a'
gpus="1"
#activation_output='sigmoid'
activation_output='softmax'
#subset='training'
subset='validation'
batch_size='24'
#batch_size='80'


for model_id in '1a' # '1b' '2'
do
    for subset in 'test' # 'training' 'validation'
    do
        base_data_dir="data/paip2020/${subset}"
        models_dir="models.${activation_output}"
        output_file="evolution_iou_model_${model_id}_${activation_output}_in_${subset}.txt"

        if [ ! -f ${output_file} ]
        then
            for epoch in 154 # {0..500}
            do
                model_filename="${models_dir}/unet${model_id}-${epoch}.onnx"

                if [ -f ${model_filename} ]
                then
                    python python/unet2.py \
                        --batch-size ${batch_size} \
                        --gpu ${gpus} \
                        --model-id ${model_id} \
                        --${activation_output} \
                        --task evaluate \
                        --data-dir ${base_data_dir} \
                        --model-filename ${model_filename} \
                        --output-dir ${base_data_dir}/mask_img_l3_predicted.128x128/

                    python python/join_mask_2.py \
                        --base-dir ${base_data_dir} \
                        --output-file ${output_file}
                fi
            done # for epoch
        fi
    done # for subset
done # for model_id
