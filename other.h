#ifndef OTHER_H
#define OTHER_H


#include <pcl/PolygonMesh.h>
#include <Eigen/Dense>
#include <iostream>

#include "pcl_utils.h"


std::pair<Pointcloud::Ptr, Pointcloud::Ptr> classify_points(Pointcloud::Ptr pc, Pointcloud::Ptr ref);

std::pair<std::vector<pcl::PointIndices>, pcl::PointCloud <pcl::PointXYZRGB>::Ptr> segment_point_cloud(Pointcloud::Ptr pc);

Pointcloud::Ptr detect_contour_points_based_on_neighbourhood(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals);

bool try_merge(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a, pcl::PointCloud<pcl::Boundary>::Ptr boundaries_a,
               Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b, pcl::PointCloud<pcl::Boundary>::Ptr boundaries_b);


pcl::PolygonMesh::Ptr triangulate_old(Pointcloud::Ptr pc);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_euclidean_clusters(Pointcloud::Ptr pc);

std::vector<pcl::Vertices> remove_faces_from_contour(Pointcloud::Ptr pc, Pointcloud::Ptr contour, const std::vector<pcl::Vertices>& faces);

std::vector<int> order_points(Pointcloud::Ptr pc, int center_idx, const Eigen::Vector3d& normal, const std::vector<int>& indices);

std::vector<pcl::Vertices> surface_triangulation(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals);

std::vector<pcl::Vertices> remove_large_faces(Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces);

std::vector<pcl::Vertices> force_triangular_faces(Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& quad_faces);

#endif // OTHER_H
