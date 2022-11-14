import os
import shutil
import os.path as osp

query_path = 'occluded_body_images/'
gallery_path = 'whole_body_images/'
tquery_path = 'query/'
tgallery_path = 'bounding_box_test/'

if not osp.exists(tquery_path):
    os.makedirs(tquery_path)

if not osp.exists(tgallery_path):
    os.makedirs(tgallery_path)
    
for i in range(200):
    i = i + 1
    filename = str(i).zfill(3)
    tfilename = str(i).zfill(4)
    for j in range(5):
        j = j + 1
        img_num = str(j).zfill(2)
        
        qimg_path = query_path + filename + '/' + filename + '_' + img_num + '.tif'
        qout_path = tquery_path + tfilename + '_c1s1' + '_' + str(j * 50).zfill(6) + '_00' + '.tif'
        shutil.copyfile(qimg_path, qout_path)
        
        gimg_path = gallery_path + filename + '/' + filename + '_' + img_num + '.tif'
        gout_path = tgallery_path + tfilename + '_c3s3' + '_' + str(j * 50).zfill(6) + '_00' + '.tif'
        shutil.copyfile(gimg_path, gout_path)