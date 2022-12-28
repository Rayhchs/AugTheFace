import cv2, os, math
from tqdm import tqdm
from glob2 import glob
import numpy as np
import dlib


"""
Capture face image
"""
# Crop image from wider face dataset
def crop_img(bbox, init_dir, out_dir):

    os.mkdir(out_dir) if not os.path.exists(out_dir) else None
    i=0
    crop_imgs = []
    annos = []
    ori_names = []
    for n, file in tqdm(enumerate(bbox[:])):
        if '.jpg' in file:
            img = cv2.imread(init_dir + file)
            begin, end = n+2, n+2+int(bbox[n+1])
            pre_dir, filename = os.path.split(file)
            os.mkdir(out_dir+pre_dir+'/') if not os.path.exists(out_dir+pre_dir+'/') else None
            
            for i in range(begin, end):
                pre_filename, _ = os.path.splitext(filename)
                anno = bbox[i].split()
                x1, y1, w, h = int(anno[0]), int(anno[1]), int(anno[2]), int(anno[3])
                i+=1
                dominate = w*2 if w > h else h*2
                if dominate <= 512 and dominate >=128:
                    newx, newy = int(x1-(dominate-w)/2), int(y1-(dominate-h)/2)
                    try:
                        if img[newy:newy+dominate, newx:newx+dominate, :].any() != None:
                            crop_imgs.append(img[newy:newy+dominate, newx:newx+dominate, :])
                            annos.append(anno)
                            ori_names.append(init_dir + file)
                        # cv2.imwrite(out_dir+pre_dir+'/'+pre_filename+f'_{i-1}.jpg', img[newy:newy+dominate, newx:newx+dominate, :])
                    except:
                        pass
    return crop_imgs, annos, ori_names

                    
"""
Avoid capturing face with less features
"""            
# Calculate face position 
def face_angle(detector, predictor, imgs, annos, ori_names):

    # Find face angle
    found_imgs = []
    found_annos = []
    found_names = []
    for j in tqdm(range(len(imgs))):
        img = imgs[j]
        ps = []
        rects = detector(img, 0)
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
            for idx, point in enumerate(landmarks):

                pos = (point[0, 0], point[0, 1])
                ps.append(pos)
                #print(idx,pos)
                
        if not len(ps) == 0: 

            image_points = np.array([
                                    ps[30],     # Nose tip
                                    ps[8],      # Chin
                                    ps[0],      # Left eye left corner 36
                                    ps[16],     # Right eye right corne 45
                                    ps[4],      # 51
                                    ps[12]      # 57
                                ], dtype="double")
            
            ### Find angle
            left_center = (image_points[2][0] + image_points[4][0])/2
            right_center = (image_points[3][0] + image_points[5][0])/2
            
            left = np.abs(image_points[0][0] - left_center)
            right = np.abs(image_points[0][0] - right_center)
           
            # return image whose angle smaller than 20
            if np.abs(left - right) <= 20:
                found_imgs.append(img)
                found_annos.append(annos[j])
                found_names.append(ori_names[j])
    return found_imgs, found_annos, found_names


if __name__ == '__main__':
    crop_img()
    face_angle()
