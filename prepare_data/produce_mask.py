import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm


image_dir = '/home/suoxin/Body/obj1/image/'
target_dir = '/home/suoxin/Body/obj1/mask/'
for i in range(1,31):
    image_file = image_dir + str(i) + '.png'
    image = cv2.imread(image_file)

    dst = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, silh = cv2.threshold(dst, 10, 255, cv2.THRESH_BINARY)
    cv2.imwrite(target_dir + str(i)+'.png',silh)


