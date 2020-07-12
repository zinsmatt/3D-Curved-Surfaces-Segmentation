# 3D Curved Surfaces SegmentationM

Extract surfaces from a probability volume and create separate meshes.

<img src="doc/seg.gif">


## Dependencies
- PCL
- Eigen3
- [segyio](https://github.com/equinor/segyio)
- NumPy
- PyMeshFix


## Building and Running Docker version

```bash
 docker build -t image_name .
 docker run -v path_to/training/inputs/:/input -v path_to/results/:/output image_name /input /output
 ```
