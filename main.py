from utils.crop_img import crop_img, face_angle
from utils.put_back_mtcnn import put_back
from utils.run_demo import Demo
from verification_net.mtcnn import MTCNN
import multiprocessing as mp
from tqdm import tqdm
import argparse
import os, cv2
import dlib

# training params
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--channel_multiplier", type=int, default=1)
parser.add_argument("--model", type=str,
                    choices=['vox'], default='vox')
parser.add_argument("--latent_dim_style", type=int, default=512)
parser.add_argument("--latent_dim_motion", type=int, default=20)
parser.add_argument("--source_path", type=str,
                    default='./datasets/images/')
parser.add_argument("--driving_path", type=str, default='./datasets/driving/')
parser.add_argument("--save_folder", type=str, default='./res/widerface_aug/')
args = parser.parse_args()

def main():
    # dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    bbox_dir = './datasets/annotation/wider_face_train_bbx_gt.txt'

    with open(bbox_dir, 'r') as f:
        bbox = f.read().splitlines()
        
    init_dir = args.source_path
    output_dir = args.save_folder

    imgs, annos, names = crop_img(bbox, init_dir, output_dir)
    imgs, annos, names = face_angle(detector, predictor, imgs, annos, names)

    gan_imgs = []
    gan_names = []
    gan_annos = []
    demo = Demo(args, annos, names)
    for n, i in tqdm(enumerate(imgs[:])):
        gan_imgs, gan_annos, gan_names = demo.run(n, i, gan_imgs, gan_annos, gan_names)
    
    detector = MTCNN()
    for n, i in tqdm(enumerate(gan_imgs[:])):
        put_back(detector, output_dir, i, n, gan_annos, gan_names)

if __name__ == '__main__':
    main()