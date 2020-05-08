#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import vtk

# reader = vtk.vtkOBJReader()
# reader.SetFileName("/home/matt/dev/Seismic_3D_Volume/build/groups_meshes/group_18.obj")
# # reader.Update()


# filler = vtk.vtkFillHolesFilter()
# filler.SetHoleSize(20)
# filler.SetInputConnection(reader.GetOutputPort())
# filler.Update()
# mesh = filler.GetOutput()


# writer = vtk.vtkOBJWriter()
# writer.SetInputConnection(filler.GetOutputPort())
# writer.SetFileName("test.obj")
# writer.Write()

#%%

# pc_reader = vtk.vtkPLYReader()
# pc_reader.SetFileName("/home/matt/dev/Seismic_3D_Volume/build/groups/group_18.ply")

# rec = vtk.vtkSurfaceReconstructionFilter()
# rec.SetInputConnection(pc_reader.GetOutputPort())
# rec.SetNeighborhoodSize(3)
# rec.Update()
# mesh = rec.GetOutput()


# writer = vtk.vtkXMLImageDataWriter()
# # writer.SetInputConnection(rec.GetOutputPort())
# writer.SetInputData(mesh)
# writer.SetFileName("test.vti")
# writer.Write()


#%%

pc_reader = vtk.vtkPLYReader()
pc_reader.SetFileName("/home/matt/dev/Seismic_3D_Volume/build/groups_meshes/group_7_smooth_pc.ply")


rec = vtk.vtkDelaunay3D()
rec.SetInputConnection(pc_reader.GetOutputPort())
mesh = rec.GetOutput()

filt = vtk.vtkDataSetSurfaceFilter()
filt.SetInputConnection(rec.GetOutputPort())


# writer = vtk.vtkXMLUnstructuredGridWriter()
# writer.SetInputConnection(rec.GetOutputPort())
# writer.SetFileName("test.vtu")
# writer.Write()



writer = vtk.vtkOBJWriter()
writer.SetInputConnection(filt.GetOutputPort())
writer.SetFileName("test.obj")
writer.Write()
