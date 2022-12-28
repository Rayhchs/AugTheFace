import numpy as np
import cv2
import os
from tqdm import tqdm
from glob2 import glob

def verification(img, detector):

    # augmented image name
    result = detector.detect_faces(img)

    if len(result) == 1:
        return result[0]
    else:
        return False
    
def put_img(dominate, newx, newy, new_box, init_img, gan_img):

    if True:
        endx = newx+new_box[0]+new_box[2]
        endy = newy+new_box[1]+new_box[3]
        end_point = (new_box[0]+new_box[2], new_box[1]+new_box[3])
        
        size = gan_img.shape[0]
        new_img = init_img.copy()
        
        for i in range(new_box[1]):
            y_r1, y_r2 = (1 - i/new_box[1]), i/new_box[1]
            new_img[newy+i:newy+(i+1), newx:newx+dominate, :] = init_img[newy+i:newy+(i+1), newx:newx+dominate, :]*y_r1 + gan_img[0+i:(i+1), :, :]*y_r2
        
        for i in range(new_box[0]):
            x_r1, x_r2 = (1 - i/new_box[0]), i/new_box[0]
            new_img[newy:newy+dominate, newx+i:newx+(i+1), :] = init_img[newy:newy+dominate, newx+i:newx+(i+1), :]*x_r1 + gan_img[:, 0+i:(i+1), :]*x_r2
            
        for i in range(size-end_point[1]):
            y_r1, y_r2 =  i/(size-end_point[1]), (1 - i/(size-end_point[1]))
            new_img[endy+i:endy+(i+1), newx+new_box[0]:newx+dominate, :] = init_img[endy+i:endy+(i+1), newx+new_box[0]:newx+dominate, :]*y_r1 + gan_img[endy-newy+i:endy-newy+(i+1), new_box[0]:, :]*y_r2
        
        for i in range(size-end_point[0]):
            x_r1, x_r2 = i/(size-end_point[0]), (1 - i/(size-end_point[0]))
            new_img[newy+new_box[1]:newy+dominate, endx+i:endx+(i+1), :] = init_img[newy+new_box[1]:newy+dominate, endx+i:endx+(i+1), :]*x_r1 + gan_img[new_box[1]:, endx-newx+i:endx-newx+(i+1), :]*x_r2
            
        new_img[newy+new_box[1]-1:newy+new_box[1]+new_box[3], newx+new_box[0]-1:newx+new_box[0]+new_box[2], :] = gan_img[new_box[1]-1:new_box[1]+new_box[3], new_box[0]-1:new_box[0]+new_box[2], :]
    return new_img

def put_back(detector, output_dir, i, n, annos, names):

    check = verification(i, detector)
    if check != False and check['confidence'] >= 0.999:

        # Setup
        anno = annos[n]
        x1, y1, w, h = int(anno[0]), int(anno[1]), int(anno[2]), int(anno[3])
        dominate = w*2 if w > h else h*2
        newx, newy = int(x1-(dominate-w)/2), int(y1-(dominate-h)/2)
                
        # New filename
        os.mkdir(output_dir+names[n].split('/')
                [-2]) if not os.path.exists(output_dir+names[n].split('/')[-2]) else None
        newname = output_dir + names[n].split('/')[-2] + '/' + names[n].split('/')[-1][:-4] + f'_p{n}.jpg'

        new_box = check['box']

        # Create new image
        init_img = cv2.imread(names[n])
        try:
            new_img = put_img(dominate, newx, newy, new_box, init_img, i)
            cv2.imwrite(newname, new_img)
        except:
            print(f"Got wrong in {newname}")