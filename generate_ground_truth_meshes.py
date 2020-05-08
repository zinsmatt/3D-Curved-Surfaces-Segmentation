#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""
import numpy as np
import os
import glob

output_folder = "/home/matt/dev/Seismic_3D_Volume/training/gt_meshes/"

for num in range(1, 11):
    filenames = glob.glob("/home/matt/dev/Seismic_3D_Volume/training/ground_truth/t%02d/*.ts" % num)
    n_files = len(filenames)
    
    colors = np.random.randint(0, 255, (n_files, 3))
    colors_str = [" ".join(map(str, c)) for c in colors]
    
    buf = []
    for it, filename in enumerate(filenames):
        with open(filename, "r") as fin:
            lines = fin.readlines()
    
        buf_loc = []
        for l in lines:
            if len(l) > 0 and l[0] == 'V':
                buf_loc.append(" ".join(l.split()[2:]) + " " + colors_str[it] + "\n")
                
        # with open("gt_%02d_%02d.obj" % (num, it), "w") as fout:
        #     for b in buf_loc:
        #         fout.write("v " + b)
            
        buf.extend(buf_loc)
        
    with open(os.path.join(output_folder, "gt_%02d.obj" % num), "w") as fout:
        for b in buf:
            fout.write("v " + b)