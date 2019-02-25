import xml.etree.ElementTree as ET
import numpy as np
import h5py

def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    str = root[0][2].text
    str_array = str.split()
    keypoint = []
    for i in range(25):
        keypoint.append([float(str_array[i*3]),float(str_array[i*3+1]),float(str_array[i*3+2])])
    keypoint = np.array(keypoint)
    print(keypoint)
    return keypoint

out_file = '/home/suoxin/Body/obj/joints/keypoints.hdf5'

with h5py.File(out_file, 'w') as f:
    poses_dset = f.create_dataset("keypoints", (30, 54), 'f', chunks=True, compression="lzf")

    for i in range(30):
        path = '/home/suoxin/Body/openpose/keypoint/' + str(i) + '_pose.xml'
        keypoint = parse_xml(path)
        poses_dset[i] = keypoint[:18, :].reshape(1,54)

