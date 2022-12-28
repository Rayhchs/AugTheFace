import torch
import torch.nn as nn
from networks.generator import Generator
import numpy as np
import torchvision
from glob2 import glob
import cv2


def load_image(filename, size):
    try:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img = filename
    img = cv2.resize(img, (size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


class Demo(nn.Module):
    def __init__(self, args, annos, names):
        super(Demo, self).__init__()

        self.args = args
        self.annos = annos
        self.names = names
        self.driving_dir = glob(args.driving_path + '**.jpg')
        print('==> loading model')

        if args.model == 'vox':
            model_path = 'checkpoints/vox.pt'
        else:
            raise NotImplementedError

        self.gen = Generator(args.size, args.latent_dim_style,
                             args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(
            model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        self.vid_target = []
        for i in self.driving_dir:
            self.vid_target.append(img_preprocessing(i, args.size))

        self.vid_target = torch.unsqueeze(torch.cat(self.vid_target), 0).cuda()

    def run(self, n, img_dir, gan_imgs, gan_annos, gan_names):

        self.img_source = img_preprocessing(img_dir, self.args.size).cuda()
        ori_size = img_dir.shape
        with torch.no_grad():

            if self.args.model == 'ted':
                h_start = None
            else:
                h_start = self.gen.enc.enc_motion(
                    self.vid_target[:, 0, :, :, :])


            for i in range(self.vid_target.size(1)):
                img_target = self.vid_target[:, i, :, :, :]
                img_recon = self.gen(self.img_source, img_target, h_start)[0]
                #img_recon = ((img_recon - img_recon.min()) / (img_recon.max() -
                #                   img_recon.min()) * 255).type('torch.ByteTensor')
                img_recon = img_recon.cpu()       
                img_recon = (img_recon+1)*127.5
                img_recon = np.transpose(np.array(img_recon), [1, 2, 0])
                img_recon = cv2.resize(img_recon, (ori_size[0], ori_size[1]))
                # img_recon = cv2.cvtColor(img_recon, cv2.COLOR_RGB2BGR)
                gan_imgs.append(img_recon)
                gan_names.append(self.names[n])
                gan_annos.append(self.annos[n])

        return gan_imgs, gan_annos, gan_names