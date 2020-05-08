#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import segyio
import numpy as np
from mesh_io import write_obj, read_ply
import glob
import os


files = glob.glob("/home/matt/dev/Seismic_3D_Volume/training/inputs/*.sgy")

for f in files:
    data = segyio.tools.cube(f)
    dim_z, dim_y, dim_x = data.shape
    y, x, z = np.where(data > 0.7)
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    write_obj(os.path.splitext(f)[0] + ".obj", pts)
