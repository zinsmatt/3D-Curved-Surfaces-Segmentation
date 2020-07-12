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