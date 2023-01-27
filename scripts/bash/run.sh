#! /bin/bash

cd ..
python re-7_bass_fullgrad.py \
--attr_map=full_grad \
--seg_map=bass \
--output_class=9 \
--img_dir=/workspace/adv_robustness/region_explainability/labelme/MNIST_94/test_images
