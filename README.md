# TopCoder Marathon Match: Fault Detection in a 3D Seismic Volume
([https://www.topcoder.com/challenges/5863dfd8-f36c-468a-8930-b2bc16b9a92a?tab=details](https://www.topcoder.com/challenges/5863dfd8-f36c-468a-8930-b2bc16b9a92a?tab=details))


The goal of this challenge is to detect and extract faults (curved surfaces) from seismic volume data.


<img src="doc/seismic_image.png" width=800 height=400>

The program takes a 3D Fault Likelihood volume as input, transforms it into a point cloud, segments curved surfaces and creates separate meshes.

<img src="doc/seg.gif" width=800 height=400>


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
