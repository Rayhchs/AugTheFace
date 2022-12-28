# AugTheFace

## AugTheFace?

AugTheFace is computer vision-based script for image augmentation from images. It uses the true bbox to crop the face images from images. Then, dlib based detector is used for identifying the face features. Those features are applied for preventing inappropriate face images. Using an image reenactment model created from ["Latent Image Animator: Learning to Animate Images via Latent Space Navigation"](https://openreview.net/pdf?id=7r6kDq0mK_), the available face images can imitate the driving images. Finally, AugTheFace uses [MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) to define the new bounding boxes of face images. The augmented face images are put back to original images following coordinate of new boxes. AugTheFace identifies all face images which are available, however, it saves image with single augmented face.

## How to use?
* Clone this repo

    ```shell script
    git clone https://github.com/Rayhchs/AugTheFace.git
    ```

* Download pre-trained weights

    Pre-trained checkpoints can be found in [LIA](https://github.com/wyhsirius/LIA). AugTheFace only uses vox.pt so put the model in `./checkpoints`. AugTheFace also uses dlib detector which can be downloaded [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

* Setup Environment
    ```shell script
    pip install -r requirements.txt
    ```
    
* Datasets

    AugTheFace uses the ground truth bbox followed by format of [widerface](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip) for cropping the face images. Input image and driving image can be any types of image, however, face in driving image should occupy at least half of image.
    
* Results

    Augmented images would be saved in `./res` defaultly. Please notice that not all of images can be used for augmentation.


### Path Arguments
|    Argument    |                                                                                                       Explanation                                                                                                       |
|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      source_path      | Path of image where you want to augment |
|    driving_path   | Path of image where you want your source image to imitate |
|    save_folder   | Where to save the augmented results |
|    bbox_dir   | Truth data of bounding box of source images |

## Demo
This repository uses widerface images for demo.

<img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo1.gif"> <img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo2.gif">
<img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo3.gif"> <img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo4.gif">
<img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo5.gif"> <img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo6.gif">
<img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo7.gif"> <img src="https://github.com/Rayhchs/AugTheFace/blob/main/images/demo8.gif">

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

Widerface:
```bibtex
@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}
```
