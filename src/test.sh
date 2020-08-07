#!/usr/bin/env bash
python test.py \
  --saved_fn 'yolo3d_yolov4_im_re' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/yolo3d_yolov4.cfg \
  --batch_size 1 \
  --num_workers 1 \
  --gpu_idx 0 \
  --pretrained_path ../checkpoints/yolo3d_yolov4_im_re/Model_yolo3d_yolov4_im_re_epoch_300.pth \
  --img_size 608 \
  --conf_thresh 0.6 \
  --nms_thresh 0.1 \
  --show_image \
  --save_test_output \
  --output_format 'image'
