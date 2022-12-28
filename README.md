# AugTheFace

## AugTheFace?

AugTheFace is computer vision-based script for image augmentation from images. It uses the true bbox to crop the face images from images. Then, dlib based detector is used for identifying the face features. Those features are applied for preventing inappropriate face images. Using an image reenactment model created from ["Latent Image Animator: Learning to Animate Images via Latent Space Navigation"](https://openreview.net/pdf?id=7r6kDq0mK_), the available face images can imitate the driving images. Finally, AugTheFace uses [MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) to define the new bounding boxes of face images. The augmented face images are put back to original images following coordinate of new boxes. AugTheFace identifies all face images which are available, however, it saves image with single augmented face.

## How to use?
* Pre-trained weights

    Pre-trained checkpoints can be found in [LIA](https://github.com/wyhsirius/LIA). AugTheFace only uses vox.pt so put the model in `./checkpoints`.

* Ubuntu Environment
    ```shell script
    pip install -r requirements.txt
    ```
* Datasets

    AugTheFace uses the ground truth bbox followed by format of [widerface](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip) for cropping the face images. Input image and driving image can be any types of image, however, face in driving image should occupy at least half of image.


## Args

## Demo

## Acknowledgement
Code and pretrain weights heavily borrows from [LIA](https://github.com/wyhsirius/LIA) and [MTCNN](https://github.com/ipazc/mtcnn). Thanks for the excellent work!

Latent Image Animator: Learning to Animate Images via Latent Space Navigation:
```bibtex
@inproceedings{
wang2022latent,
title={Latent Image Animator: Learning to Animate Images via Latent Space Navigation},
author={Yaohui Wang and Di Yang and Francois Bremond and Antitza Dantcheva},
booktitle={International Conference on Learning Representations},
year={2022}
}
```
    
MTCNN:
```bibtex
@article{7553523,
    author={K. Zhang and Z. Zhang and Z. Li and Y. Qiao}, 
    journal={IEEE Signal Processing Letters}, 
    title={Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks}, 
    year={2016}, 
    volume={23}, 
    number={10}, 
    pages={1499-1503}, 
    keywords={Benchmark testing;Computer architecture;Convolution;Detectors;Face;Face detection;Training;Cascaded convolutional neural network (CNN);face alignment;face detection}, 
    doi={10.1109/LSP.2016.2603342}, 
    ISSN={1070-9908}, 
    month={Oct}
}
```   

### [Project Page](https://wyhsirius.github.io/LIA-project/) | [Paper](https://openreview.net/pdf?id=7r6kDq0mK_)
This is the official PyTorch implementation of the ICLR 2022 paper "Latent Image Animator: Learning to Animate Images via Latent Space Navigation"

[![Replicate](https://replicate.com/wyhsirius/lia/badge)](https://replicate.com/wyhsirius/lia)

<img src="LIA.gif" width="500">

## Requirements
- Python 3.7
- PyTorch 1.5+
- tensorboard
- moviepy
- av
- tqdm
- lpips

## 1. Animation demo

Download pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1N4QcnqUQwKUZivFV-YeBuPyH4pGJHooc?usp=sharing) and put models under `./checkpoints`. We have provided several demo source images and driving videos in `./data`. 
To obtain demos, you could run following commands, generated results will be saved under `./res`.
```shell script
python run_demo.py --model vox --source_path ./data/vox/macron.png --driving_path ./data/vox/driving1.mp4 # using vox model
python run_demo.py --model taichi --source_path ./data/taichi/subject1.png --driving_path ./data/taichi/driving1.mp4 # using taichi model
python run_demo.py --model ted --source_path ./data/ted/subject1.png --driving_path ./data/ted/driving1.mp4 # using ted model
```
If you would like to use your own image and video, indicate `<SOURCE_PATH>` (source image), `<DRIVING_PATH>` (driving video), `<DATASET>` and run   
```shell script
python run_demo.py --model <DATASET> --source_path <SOURCE_PATH> --driving_path <DRIVING_PATH>
```
## 2. Datasets

Please follow the instructions in [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [MRAA](https://github.com/snap-research/articulated-animation) to download and preprocess VoxCeleb, Taichi and Ted datasets. Put datasets under `./datasets` and organize them as follows:

#### Vox (Taichi, Ted)
```
Video Dataset (vox, taichi, ted)
|-- train
    |-- video1
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- video2
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- ...
|-- test
    |-- video1
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- video2
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- ...
```
## 3. Training
By default, we use `DistributedDataParallel` on 8 V100 for all datasets. To train the netowrk, run
```shell script
python train.py --dataset <DATSET> --exp_path <EXP_PATH> --exp_name <EXP_NAME>
```
The dataset list is as follows, `<DATASET>`: {`vox`,`taichi`,`ted`}. Tensorboard log and checkpoints will be saved in `<EXP_PATH>/<EXP_NAME>/log` and `<EXP_PATH>/<EXP_NAME>/chekcpoints` respectively.

To train from a checkpoint, run
```shell script
python train.py --dataset <DATASET> --exp_path <EXP_PATH> --exp_name <EXP_NAME> --resume_ckpt <CHECKPOINT_PATH>
```
## 4. Evaluation
To obtain reconstruction and LPIPS results, put checkpoints under `./checkpoints` and run
```shell script
python evaluation.py --dataset <DATASET> --save_path <SAVE_PATH>
```
Generated videos will be save under `<SAVE_PATH>`. For other evaluation metrics, we use the code from [here](https://github.com/AliaksandrSiarohin/pose-evaluation).
## 5. Linear manipulation
To obtain linear manipulation results of a single image, run
```shell script
python linear_manipulation.py --model <DATAET> --img_path <IMAGE_PATH> --save_folder <RESULTS_PATH>
```
By default, results will be saved under `./res_manipulation`.
## Ack
If you find this code useful for your research, please consider citing our paper:
```bibtex
@inproceedings{
wang2022latent,
title={Latent Image Animator: Learning to Animate Images via Latent Space Navigation},
author={Yaohui Wang and Di Yang and Francois Bremond and Antitza Dantcheva},
booktitle={International Conference on Learning Representations},
year={2022}
}
```
## Acknowledgement
Part of the code is adapted from [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [MRAA](https://github.com/snap-research/articulated-animation). We thank authors for their contribution to the community.
