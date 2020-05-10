#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

from geomdl import NURBS
import numpy as np

# Create a NURBS surface instance
surf = NURBS.Surface()

# Set degrees
surf.degree_u = 3
surf.degree_v = 2

# Set control points (weights vector will be 1 by default)
# Use curve.ctrlptsw is if you are using homogeneous points as Pw
control_points = [[0, 0, 0], [0, 4, 0], [0, 8, -3],
                  [2, 0, 6], [2, 4, 0], [2, 8, 0],
                  [4, 0, 0], [4, 4, 0], [4, 8, 3],
                  [6, 0, 0], [6, 4, -3], [6, 8, 0]]
surf.set_ctrlpts(control_points, 4, 3)

# Set knot vectors
surf.knotvector_u = [0, 0, 0, 0, 1, 1, 1, 1]
surf.knotvector_v = [0, 0, 0, 1, 1, 1]

# Set evaluation delta (control the number of surface points)
surf.delta = 0.05

# Get surface points (the surface will be automatically evaluated)
surface_points = surf.evalpts

ctrl_pts = np.vstack(control_points)

pts = np.vstack(surface_points)



with open("control_points.obj", "w") as fout:
    for p in ctrl_pts:
        fout.write("v " + " ".join(map(str, p)) + "\n")
        
with open("NURBS.obj", "w") as fout:
    for p in pts:
        fout.write("v " + " ".join(map(str, p)) + "\n")