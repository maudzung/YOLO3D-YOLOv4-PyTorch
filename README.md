# YOLO3D-YOLOv4-PyTorch


[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

The PyTorch Implementation based on YOLOv4 of the paper:
 [YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud **(ECCV 2018)**](https://arxiv.org/pdf/1808.02350v1.pdf)

---

## Demo

![demo](./docs/demo.gif)

- **Inputs**: Bird-eye-view _(BEV)_ maps that are encoded by _**height, intensity and density**_ of 3D LiDAR point clouds.
- **The input size**: _608 x 608 x 3_
- **Outputs**: **7 degrees of freedom** _(7-DOF)_ of objects: `(cx, cy, cz, l, w, h, θ)`
   - `cx, cy, cz`: The center coordinates.
   - `l, w, h`: length, width, height of the bounding box.
   - `θ`: The heading angle in radians of the bounding box.
- **Objects**: Cars, Pedestrians, Cyclists.

## Features
- [x] Realtime 3D object detection based on YOLOv4
- [x] Support [distributed data parallel training](https://github.com/pytorch/examples/tree/master/distributed/ddp)
- [x] Tensorboard
- [x] Mosaic/Cutout augmentation for training

## 2. Getting Started
### 2.1. Requirement

```shell script
pip install -U -r requirements.txt
```

For [`mayavi`](https://docs.enthought.com/mayavi/mayavi/installation.html) and [`shapely`](https://shapely.readthedocs.io/en/latest/project.html#installing-shapely) 
libraries, please refer to the installation instructions from their official websites.


### 2.2. Data Preparation
Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

The downloaded data includes:

- Velodyne point clouds _**(29 GB)**_: input data to the YOLO3D model
- Training labels of object data set _**(5 MB)**_: input label to the YOLO3D model
- Camera calibration matrices of object data set _**(16 MB)**_: for visualization of predictions
- Left color images of object data set _**(12 GB)**_: for visualization of predictions

Please make sure that you construct the source code & dataset directories structure as below.

### 2.3. YOLOv4 architecture

![architecture](./docs/yolov4_architecture.png)

This work has been based on the paper [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934).

#### List of usage for Bag of Freebies (BoF) & Bag of Specials (BoS) in this implementation


|   |Backbone   | Detector   |
|---|---|---|
|**BoF**   |[x] Dropblock <br> [x] Random rescale, rotation (global) <br> [x] Mosaic/Cutout augmentation|[x] Cross mini-Batch Normalization <br>[x] Dropblock <br> [x] Random training shapes <br>   |
|**BoS**   |[x] Mish activation <br> [x] Cross-stage partial connections (CSP) <br> [x] Multi-input weighted residual connections (MiWRC)   |[x] Mish activation <br> [x] SPP-block <br> [x] SAM-block <br> [x] PAN path-aggregation block <br>|


### 2.4. How to run

#### 2.4.1. Visualize the dataset (both BEV images from LiDAR and camera images)

```shell script
cd src/data_process
```

- To visualize BEV maps and camera images (with 3D boxes), let's execute _**(the `output-width` param can be changed to 
show the images in a bigger/smaller window)**_:

```shell script
python kitti_dataloader.py --output-width 608
```

- To visualize the `cutout augmentation`, let's execute:

```shell script
python kitti_dataloader.py --show-train-data --cutout_prob 1. --cutout_nholes 1 --cutout_fill_value 1. --cutout_ratio 0.3 --output-width 608
```

#### 2.4.2. Inference

Download the trained model from [**_here_**](https://drive.google.com/drive/folders/1y0635Kq_Zh45km95ykct-CcVEsJrxLBK?usp=sharing), 
then put it to `${ROOT}/checkpoints/` and execute:

```shell script
python test.py --gpu_idx 0 --pretrained_path ../checkpoints/yolo3d_yolov4.pth --cfgfile ./config/cfg/yolo3d_yolov4.cfg 
```

#### 2.4.3. Evaluation

```shell script
python evaluate.py --gpu_idx 0 --pretrained_path <PATH> --cfgfile <CFG> --img_size <SIZE> --conf-thresh <THRESH> --nms-thresh <THRESH> --iou-thresh <THRESH>
```
(The `conf-thresh`, `nms-thresh`, and `iou-thresh` params can be adjusted. By default, these params have been set to _**0.5**_)

#### 2.4.4. Training

##### 2.4.4.1. Single machine, single gpu

```shell script
python train.py --gpu_idx 0 --batch_size <N> --num_workers <N>...
```

##### 2.4.4.2. Multi-processing Distributed Data Parallel Training
We should always use the `nccl` backend for multi-processing distributed training since it currently provides the best 
distributed training performance.

- **Single machine (node), multiple GPUs**

```shell script
python train.py --dist-url 'tcp://127.0.0.1:29500' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

- **Two machines (two nodes), multiple GPUs**

_**First machine**_

```shell script
python train.py --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0
```
_**Second machine**_

```shell script
python train.py --dist-url 'tcp://IP_OF_NODE2:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1
```

To reproduce the results, you can run the bash shell script

```bash
./train.sh
```

#### Tensorboard

- To track the training progress, go to the `logs/` folder and 

```shell script
cd logs/<saved_fn>/tensorboard/
tensorboard --logdir=./
```

- Then go to [http://localhost:6006/](http://localhost:6006/):


## Contact

If you think this work is useful, please give me a star! <br>
If you find any errors or have any suggestions, please contact me (**Email:** `nguyenmaudung93.kstn@gmail.com`). <br>
Thank you!


## Citation

```bash
@article{YOLOv4,
  author = {Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  title = {YOLOv4: Optimal Speed and Accuracy of Object Detection},
  year = {2020},
  journal = {arXiv},
}
@article{YOLO3D,
  author = {Waleed Ali, Sherif Abdelkarim, Mohamed Zahran,  Mahmoud Zidan, Ahmad El Sallab},
  title = {YOLO3D: End-to-end real-time 3d oriented object bounding box detection from lidar point cloud},
  year = {2018},
  conference = {ECCV 2018},
}
@misc{YOLO3D-YOLOv4-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{YOLO3D-YOLOv4-PyTorch: PyTorch Implementation of based on YOLOv4 of YOLO3D paper}},
  howpublished = {\url{https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch}},
  year =         {2020}
}
```


## Folder structure


```
${ROOT}
└── checkpoints/    
│   ├── yolo3d_yolov4.pth
└── dataset/    
│   └── kitti/
│   │   ├──ImageSets/
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   └── val.txt
│   │   ├── training/
│   │   │   ├── image_2/ <-- for visualization
│   │   │   ├── calib/
│   │   │   ├── label_2/
│   │   │   └── velodyne/
│   │   └── testing/  
│   │   │   ├── image_2/ <-- for visualization
│   │   │   ├── calib/
│   │   │   └── velodyne/ 
│   │   └── classes_names.txt
└── src/
│   ├── config/
│   │   ├── cfg/
│   │   │    ├── yolo3d_yolov4.cfg
│   │   │    ├── yolo3d_yolov4_tiny.cfg
│   │   ├── train_config.py
│   │   └── kitti_config.py
│   ├── data_process/
│   │   ├── kitti_bev_utils.py
│   │   ├── kitti_dataloader.py
│   │   ├── kitti_dataset.py
│   │   ├── kitti_data_utils.py
│   │   └── transformation.py
│   ├── models/
│   │   ├── darknet2pytorch.py
│   │   ├── darknet_utils.py
│   │   ├── model_utils.py
│   │   ├── yolo_layer.py
│   └── utils/
│   │   ├── evaluation_utils.py
│   │   ├── iou_utils.py
│   │   ├── logger.py
│   │   ├── misc.py
│   │   ├── torch_utils.py
│   │   ├── train_utils.py
│   │   └── visualization_utils.py
│   ├── evaluate.py
│   ├── test.py
│   ├── test.sh
│   ├── train.py
│   └── train.sh
├── README.md 
└── requirements.txt
```

## Usage

```
python train.py --help

usage: train.py [-h] [--seed SEED] [--saved_fn FN] [--root-dir PATH]
                [-a ARCH] [--cfgfile PATH] [--pretrained_path PATH]
                [--use_giou_loss] [--img_size IMG_SIZE]
                [--hflip_prob HFLIP_PROB] [--cutout_prob CUTOUT_PROB]
                [--cutout_nholes CUTOUT_NHOLES] [--cutout_ratio CUTOUT_RATIO]
                [--cutout_fill_value CUTOUT_FILL_VALUE]
                [--multiscale_training] [--mosaic] [--random-padding]
                [--no-val] [--num_samples NUM_SAMPLES]
                [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE]
                [--print_freq N] [--tensorboard_freq N] [--checkpoint_freq N]
                [--start_epoch N] [--num_epochs N] [--lr_type LR_TYPE]
                [--lr LR] [--minimum_lr MIN_LR] [--momentum M] [-wd WD]
                [--optimizer_type OPTIMIZER] [--burn_in N]
                [--steps [STEPS [STEPS ...]]] [--world-size N] [--rank N]
                [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                [--gpu_idx GPU_IDX] [--no_cuda]
                [--multiprocessing-distributed] [--evaluate]
                [--resume_path PATH] [--conf-thresh CONF_THRESH]
                [--nms-thresh NMS_THRESH] [--iou-thresh IOU_THRESH]

The Implementation of YOLO3D-YOLOv4 using PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           re-produce the results with seed random
  --saved_fn FN         The name using for saving logs, models,...
  --root-dir PATH    The ROOT working directory
  -a ARCH, --arch ARCH  The name of the model architecture
  --cfgfile PATH        The path for cfgfile (only for darknet)
  --pretrained_path PATH
                        the path of the pretrained checkpoint
  --use_giou_loss       If true, use GIoU loss during training. If false, use
                        MSE loss for training
  --img_size IMG_SIZE   the size of input image
  --hflip_prob HFLIP_PROB
                        The probability of horizontal flip
  --cutout_prob CUTOUT_PROB
                        The probability of cutout augmentation
  --cutout_nholes CUTOUT_NHOLES
                        The number of cutout area
  --cutout_ratio CUTOUT_RATIO
                        The max ratio of the cutout area
  --cutout_fill_value CUTOUT_FILL_VALUE
                        The fill value in the cut out area, default 0. (black)
  --multiscale_training
                        If true, use scaling data for training
  --mosaic              If true, compose training samples as mosaics
  --random-padding      If true, random padding if using mosaic augmentation
  --no-val              If true, dont evaluate the model on the val set
  --num_samples NUM_SAMPLES
                        Take a subset of the dataset to run and debug
  --num_workers NUM_WORKERS
                        Number of threads for loading data
  --batch_size BATCH_SIZE
                        mini-batch size (default: 4), this is the totalbatch
                        size of all GPUs on the current node when usingData
                        Parallel or Distributed Data Parallel
  --print_freq N        print frequency (default: 50)
  --tensorboard_freq N  frequency of saving tensorboard (default: 20)
  --checkpoint_freq N   frequency of saving checkpoints (default: 2)
  --start_epoch N       the starting epoch
  --num_epochs N        number of total epochs to run
  --lr_type LR_TYPE     the type of learning rate scheduler (cosin or
                        multi_step)
  --lr LR               initial learning rate
  --minimum_lr MIN_LR   minimum learning rate during training
  --momentum M          momentum
  -wd WD, --weight_decay WD
                        weight decay (default: 1e-6)
  --optimizer_type OPTIMIZER
                        the type of optimizer, it can be sgd or adam
  --burn_in N           number of burn in step
  --steps [STEPS [STEPS ...]]
                        number of burn in step
  --world-size N        number of nodes for distributed training
  --rank N              node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --gpu_idx GPU_IDX     GPU index to use.
  --no_cuda             If true, cuda is not used.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --evaluate            only evaluate the model, not training
  --resume_path PATH    the path of the resumed checkpoint
  --conf-thresh CONF_THRESH
                        for evaluation - the threshold for class conf
  --nms-thresh NMS_THRESH
                        for evaluation - the threshold for nms
  --iou-thresh IOU_THRESH
                        for evaluation - the threshold for IoU
```

[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
