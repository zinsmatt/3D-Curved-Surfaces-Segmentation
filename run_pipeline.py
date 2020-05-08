#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import glob
import os
import subprocess


data_folder = "/home/matt/dev/Seismic_3D_Volume/training/inputs/"
output_folder = "/home/matt/dev/Seismic_3D_Volume/results/"

files = sorted(glob.glob(os.path.join(data_folder, "*.obj")))

for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    output = os.path.join(output_folder, name)
    cmd = ["build/3D_segmentation", f, output]
    print("Run ", " ".join(cmd))
    subprocess.run(cmd)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # os.system(" ".join(cmd))
    # subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
               
