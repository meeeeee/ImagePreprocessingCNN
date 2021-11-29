import sys
import glob
import h5py
import cv2
import numpy as np

IMG_WIDTH = 110
IMG_HEIGHT = 20

h5file = r'dataset/test.h5'

nfiles = len(glob.glob('english/noise/*.png'))
print(f'count of image files nfiles={nfiles}')

# resize all images and load into a single dataset
with h5py.File(h5file,'w') as h5f:
    img_ds = h5f.create_dataset('images',shape=(nfiles, IMG_HEIGHT, IMG_WIDTH,3), dtype=int)
    for cnt, ifile in enumerate(glob.iglob('english/noise/*.png')) :
        img = cv2.imread(ifile, cv2.IMREAD_COLOR)
        # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
        #img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_ds[cnt:cnt+1:,:,:] = img[np.newaxis, ...]#_resize