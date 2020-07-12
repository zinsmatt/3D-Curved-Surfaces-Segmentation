#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import segyio
import numpy as np
from mesh_io import write_obj, read_ply

        
data = segyio.tools.cube('/home/matt/dev/Seismic_3D_Volume/training/inputs/t04.sgy')
dim_x, dim_y, dim_z = data.shape



x, y, z = np.where(data > 0.7)
pts = np.vstack((z.flatten(), y.flatten(), x.flatten())).T

write_obj("out.obj", pts)

#%%
# from sklearn.cluster import DBSCAN


# pts, _ = read_ply("build/planar_points.ply")

# clustering = DBSCAN(eps=2, min_samples=8).fit(pts)
# nb_clusters = np.max(clustering.labels_) + 2
# colors = np.random.randint(0, 255, (nb_clusters, 3))


# with open("planar_points_colored.obj", "w") as fout:
#     for i, p in enumerate(pts):
#         fout.write("v " + " ".join(map(str, p))+ " ")
#         fout.write(" ".join(map(str, colors[clustering.labels_[i]+1])) + "\n")


# j = 0
# for i in range(nb_clusters):
#     sub = pts[clustering.labels_ == i-1, :]
#     if sub.shape[0] > 50:
#         write_obj("build/parts/part_%03d.obj" % j, sub)
#         j += 1



# #%% GT meshes
# import os
# import glob
# num = 2
# filenames = glob.glob("/home/matt/dev/Seismic_3D_Volume/training/ground_truth/t%02d/*.ts" % num)
# n_files = len(filenames)

# colors = np.random.randint(0, 255, (n_files, 3))
# colors_str = [" ".join(map(str, c)) for c in colors]

# buf = []
# for it, filename in enumerate(filenames):
#     with open(filename, "r") as fin:
#         lines = fin.readlines()

#     buf_loc = []
#     for l in lines:
#         if len(l) > 0 and l[0] == 'V':
#             buf_loc.append(" ".join(l.split()[2:]) + " " + colors_str[it] + "\n")
            
#     # with open("gt_%02d_%02d.obj" % (num, it), "w") as fout:
#     #     for b in buf_loc:
#     #         fout.write("v " + b)
        
#     buf.extend(buf_loc)
    
# with open("gt_%02d.obj" % num, "w") as fout:
#     for b in buf:
#         fout.write("v " + b)
    
#%% Volumetric vizu

# data = np.transpose(data, (0, 1, 2)).copy()


dim_z, dim_y, dim_x = data.shape
occupancy = data.flatten()
import vtk

print("dim x = ", dim_x)
print("dim y = ", dim_y)
print("dim z = ", dim_z)



xCoords = vtk.vtkFloatArray()
for i in range(dim_x):
    xCoords.InsertNextValue(i)
    
yCoords = vtk.vtkFloatArray()
for i in range(dim_y):
    yCoords.InsertNextValue(i)
    
zCoords = vtk.vtkFloatArray()
for i in range(dim_z):
    zCoords.InsertNextValue(i)
    
values = vtk.vtkFloatArray()
for i in occupancy:
    values.InsertNextValue(i)
    
    
    

rgrid = vtk.vtkRectilinearGrid()
rgrid.SetDimensions(dim_x, dim_y, dim_z)
rgrid.SetXCoordinates(xCoords)
rgrid.SetYCoordinates(yCoords)
rgrid.SetZCoordinates(zCoords)
rgrid.GetPointData().SetScalars(values)

writer = vtk.vtkXMLRectilinearGridWriter()
writer.SetFileName("shape.vtr")
writer.SetInputData(rgrid)
writer.Write()