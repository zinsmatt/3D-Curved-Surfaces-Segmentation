"""
@author: Matthieu Zins
"""

import glob
import os
import numpy as np
import subprocess
import segyio
import argparse
import sys


# data_folder = "/home/matt/dev/Seismic_3D_Volume/training/inputs/"
# output_folder = "/home/matt/dev/Seismic_3D_Volume/results/"

input_folder = sys.argv[1]
output_folder = sys.argv[2]

print("input folder: ", input_folder)
print("output folder: ", output_folder)
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

files = sorted(glob.glob(os.path.join(input_folder, "*.sgy")))


def write_obj(fname, pts):
    """
        This function writes an obj mesh from a list of points
    """
    assert pts.shape[1] == 3
    with open(fname, "w") as fout:
        for i, p in enumerate(pts):
            fout.write("v %f %f %f\n" % (p[0], p[1], p[2]))


temp_file = os.path.join(output_folder, "temporary.obj")

for f in files:
    data = segyio.tools.cube(f)
    dim_z, dim_y, dim_x = data.shape
    y, x, z = np.where(data > 0.7)
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    write_obj(temp_file, pts)
    print("Generate temporary.obj", flush=True)
    
    
    name = os.path.splitext(os.path.basename(f))[0]
    output = os.path.join(output_folder, name)
    if not os.path.isdir(output):
        os.mkdir(output)
        
    cmd = ["build/3D_segmentation", temp_file, output]
    print("Run ", " ".join(cmd), flush=True)
    subprocess.run(cmd)

os.remove(temp_file)
