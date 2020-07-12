import glob
import os
import numpy as np
import subprocess
import segyio
import argparse
import sys
from pymeshfix._meshfix import PyTMesh



def write_obj(fname, pts):
    """
        This function writes a point cloud in OBJ format from a list of points
    """
    assert pts.shape[1] == 3
    with open(fname, "w") as fout:
        for i, p in enumerate(pts):
            fout.write("v %f %f %f\n" % (p[0], p[1], p[2]))


def clean_mesh(fin, fout):
    """
        Fill some holes, clean the mesh and output it in TS format
    """
    mfix = PyTMesh(False)
    mfix.load_file(f)
    mfix.fill_small_boundaries(nbe=100, refine=True)
    mfix.clean(max_iters=10, inner_loops=0)

    vert, faces = mfix.return_arrays()
    faces += 1
    with open(fout, "w") as fileout:
        for i, v in enumerate(vert):
            fileout.write("VRTX %d %f %f %f\n" % (i+1, *v))
        for fa in faces:
            fileout.write("TRGL %d %d %d\n" % (fa[0], fa[1], fa[2]))




input_folder = sys.argv[1]
output_folder = sys.argv[2]

print("input folder: ", input_folder)
print("output folder: ", output_folder)

# create the output folder if needed
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# find all Seg-Y input files 
files = sorted(glob.glob(os.path.join(input_folder, "*.sgy")))

# path to the temporary file
temp_file = os.path.join(output_folder, "temporary.obj")


for f in files:
    # read the Seg-Y file
    data = segyio.tools.cube(f)
    dim_z, dim_y, dim_x = data.shape
    y, x, z = np.where(data > 0.7)
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    # write the temporary point cloud
    write_obj(temp_file, pts)
    print("Generate temporary.obj", flush=True)
    
    # create the output folder, where the generated meshes will be written
    name = os.path.splitext(os.path.basename(f))[0]
    output = os.path.join(output_folder, name)
    if not os.path.isdir(output):
        os.mkdir(output)

    # run C++ algorithms
    cmd = ["build/3D_segmentation", temp_file, output]
    print("Run ", " ".join(cmd), flush=True)
    subprocess.run(cmd)  # it outputs meshes in PLY format


    # final post-processing of these meshes and convertion to TS format
    all_ply_files = glob.glob(os.path.join(output, "*.ply"))
    for f in all_ply_files:
        file_ts = os.path.splitext(f)[0] + ".ts"
        print("Clean mesh", f, flush=True)
        clean_mesh(f, file_ts)
        print("Generate mesh", file_ts, flush=True)
        os.remove(f)
        print("Remove mesh", f, flush=True)

# delete the temporary point cloud file
os.remove(temp_file)
