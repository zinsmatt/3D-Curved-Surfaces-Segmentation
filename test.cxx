//=========================================================================
//
// Copyright 2019 Kitware, Inc.
// Author: Matthieu Zins
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//=========================================================================

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <unordered_set>
#include <filesystem>
#include <numeric>
 
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/boundary.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/grid_projection.h>

//#include <pcl/surface/on_nurbs/fitting_curve_pdm.h>
//#include <pcl/surface/on_nurbs/triangulation.h>

#include <vtkSmartPointer.h>
#include <vtkDelaunay3D.h>
#include <vtkOBJWriter.h>
#include <vtkPolyData.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkDataSetSurfaceFilter.h>

#include <Eigen/Dense>

#define PI 3.14159265359


using Point = pcl::PointXYZ;
using Pointcloud = pcl::PointCloud<Point>;
using KDTree = pcl::KdTreeFLANN<Point>;

namespace fs = std::filesystem;


class Match_debugger
{
  public:

  void add_edge(Point a, Point b)
  {
    points.push_back(a);
    points.push_back(b);
  }

  /// Add some points, not defining edges (this should be done after all the edges)
  void add_raw_point_cloud(Pointcloud::Ptr pc)
  {
    other_points.insert(other_points.end(), pc->points.begin(), pc->points.end());
  }

  void save_obj(const std::string& name) const
  {
    std::ofstream file(name);
    for (auto& p : points)
    {
      file << "v " << std::to_string(p.x) << " " << std::to_string(p.y) << " " << std::to_string(p.z) << "\n";
    }
    for (int i = 1; i <= points.size() / 2; i += 2)
    {
      file << "l " << i << " " << i+1 << "\n";
    }

    for (auto& p : other_points)
    {
      file << "v " << std::to_string(p.x) << " " << std::to_string(p.y) << " " << std::to_string(p.z) << "\n";
    }

    file.close();
  }

  private:
    std::vector<Point> points;
    std::vector<Point> other_points;

};

void voxelGridFilter(Pointcloud::Ptr& pc, double voxelSize)
{
  pcl::VoxelGrid<Point> vox_grid;
  vox_grid.setInputCloud(pc);
  vox_grid.setLeafSize(voxelSize, voxelSize, voxelSize);
  Pointcloud::Ptr tempCloud (new Pointcloud);
  vox_grid.filter(*tempCloud);
  pc.swap(tempCloud);
}


std::pair<Pointcloud::Ptr, Pointcloud::Ptr> classify_points(Pointcloud::Ptr pc, Pointcloud::Ptr ref)
{
  Pointcloud::Ptr planar_pts(new Pointcloud());
  Pointcloud::Ptr other_pts(new Pointcloud());

  KDTree::Ptr tree(new KDTree);
  tree->setInputCloud(pc);
  double radius = 4.0;

  for (unsigned int i = 0; i < ref->size(); ++i)
  {
    std::vector<int> nearest_index;
    std::vector<float> nearest_sq_dist;
    tree->radiusSearch(ref->points[i], radius, nearest_index, nearest_sq_dist);

    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(nearest_index.size(), 3);
    Eigen::Vector3d center(0.0, 0.0, 0.0);
    Eigen::Vector3d central_point(ref->points[i].x, ref->points[i].y, ref->points[i].z);
    for (unsigned int j = 0; j < nearest_index.size(); ++j)
    {
      pts(j, 0) = pc->points[nearest_index[j]].x;
      pts(j, 1) = pc->points[nearest_index[j]].y;
      pts(j, 2) = pc->points[nearest_index[j]].z;
      center += pts.row(j);
    }
    center /= nearest_index.size();

    pts = pts.rowwise() - central_point.transpose(); // center.transpose();     // maybe remove directly ref->points[i] ??
    Eigen::Matrix3d mat = (1.0 / nearest_index.size()) * pts.transpose() * pts;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat);
    Eigen::Vector3d vals = svd.singularValues();

    std::sort(vals.data(), vals.data() + vals.size(), std::greater<double>());
    //if (vals[2] < 20 * vals[1] / vals[0])
    if ((vals[1] - vals[2]) / (vals[0] - vals[2]) >= 0.6)
    {
      // planar
      planar_pts->push_back(ref->points[i]);
    }
    else
    {
      other_pts->push_back(ref->points[i]);
    }
  }

  return {planar_pts, other_pts};
}



std::pair<std::vector<pcl::PointIndices>, pcl::PointCloud <pcl::PointXYZRGB>::Ptr> segment_point_cloud(Pointcloud::Ptr pc)
{
  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud(pc);
  normal_estimator.setKSearch(10);
  normal_estimator.compute(*normals);

  //pcl::PassThrough<pcl::PointXYZ> pass;
  //pcl::IndicesPtr indices (new std::vector <int>);
  //pass.setInputCloud(cloud);
  //pass.setFilterFieldName("z");
  //pass.setFilterLimits(0.0, 1.0);
  //pass.filter(*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize(50);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(10);
  reg.setInputCloud(pc);

  //reg.setIndices (indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(5.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(2.0);

  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;


  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();

  

  return {clusters, colored_cloud};
}


pcl::PointCloud<pcl::Normal>::Ptr compute_normals(Pointcloud::Ptr pc)
{
  pcl::search::Search<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<Point, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(pc);
  //normal_estimator.setKSearch(10);
  normal_estimator.setRadiusSearch(5);
  normal_estimator.setViewPoint(-10, -10, -10);
  normal_estimator.compute(*normals);
 
  return normals;
}

pcl::PointCloud<pcl::Normal>::Ptr compute_normals2(Pointcloud::Ptr pc)
{
  pcl::search::Search<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<Point, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(pc);
  normal_estimator.setKSearch(10);
  //normal_estimator.setRadiusSearch(5);
  normal_estimator.setViewPoint(-10, -10, -10);
  normal_estimator.compute(*normals);
 
  return normals;
}


Pointcloud::Ptr detect_contour_points(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  Pointcloud::Ptr contour_pts(new Pointcloud());

  KDTree::Ptr tree(new KDTree);
  tree->setInputCloud(pc);

  for (unsigned int i = 0; i < pc->size(); ++i)
  {
    std::vector<int> nearest_index;
    std::vector<float> nearest_sq_dist;
    tree->radiusSearch(pc->points[i], 8.0, nearest_index, nearest_sq_dist);

    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(nearest_index.size(), 3);
    Eigen::Vector3d center(0.0, 0.0, 0.0);
    Eigen::Vector3d n(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z);
    n.normalize();
    double x = pc->points[i].x;
    double y = pc->points[i].y;
    double z = pc->points[i].z;

    for (unsigned int j = 0; j < nearest_index.size(); ++j)
    {
      pts(j, 0) = pc->points[nearest_index[j]].x - x;
      pts(j, 1) = pc->points[nearest_index[j]].y - y;
      pts(j, 2) = pc->points[nearest_index[j]].z - z;
      pts.row(j) -= (pts.row(j).dot(n)) * n;
      center += pts.row(j);
    }
    center /= nearest_index.size();

    std::vector<double> angles(nearest_index.size());
    Eigen::Vector3d vx = pts.row(nearest_index.size()-1);
    vx.normalize();
    Eigen::Vector3d vz = n;
    Eigen::Vector3d vy = vz.cross(vx);
    double min_angle = std::numeric_limits<double>::infinity();
    double max_angle = -min_angle;
    for (unsigned int j = 0; j < nearest_index.size(); ++j)
    {
      double cos = pts.row(j).dot(vx);
      double sin = pts.row(j).dot(vy);
      angles[j] = std::atan2(sin, cos);
      if (angles[j] < min_angle)
        min_angle = angles[j];
      if (angles[j] > max_angle)
        max_angle = angles[j];
    }

    if (max_angle - min_angle < PI )
    {
      contour_pts->push_back(pc->points[i]);
    }

    //double dist = sqrt(center[0]*center[0] + center[1]*center[1] + center[2]*center[2]);
    //if (dist > 2.0)
    //{
    //  contour_pts->push_back(pc->points[i]);
    //}
  }

  return contour_pts;
}

std::pair<std::vector<int>, pcl::PointCloud<pcl::Boundary>::Ptr> detect_contour_points2(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>());
  pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
  est.setInputCloud(pc);
  est.setInputNormals(normals);
  est.setRadiusSearch(15.0);
  est.setSearchMethod(typename pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
  est.compute(*boundaries);
  std::vector<int> ctrs_points_inds;
  for (unsigned int i = 0; i < boundaries->size(); ++i)
  {
    if (boundaries->points[i].boundary_point)
    {
      ctrs_points_inds.push_back(i);
    }
  }
  return {ctrs_points_inds, boundaries};
}

bool try_merge(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a, pcl::PointCloud<pcl::Boundary>::Ptr boundaries_a,
               Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b, pcl::PointCloud<pcl::Boundary>::Ptr boundaries_b)
{
  Match_debugger dbg;

  KDTree::Ptr tree_b(new KDTree());
  tree_b->setInputCloud(pc_b);

  std::vector<double> distances;
  int nb_on_ctr = 0;
  int nb_pts_ctr = 0;
  for (unsigned int i = 0; i < pc_a->size(); ++i)
  {
    if (boundaries_a->points[i].boundary_point)
    {
      std::vector<float> nearest_sq_dist;
      std::vector<int> nearest_idx;
      tree_b->nearestKSearch(pc_a->points[i], 1, nearest_idx, nearest_sq_dist);
      int j = nearest_idx.front();

      Eigen::Vector3d na(normals_a->points[i].normal_x, normals_a->points[i].normal_y, normals_a->points[i].normal_z);
      Eigen::Vector3d nb(normals_b->points[j].normal_x, normals_b->points[j].normal_y, normals_b->points[j].normal_z);
      na.normalize();
      nb.normalize();
      if (na.dot(nb) < 0.9)
      {
        continue;
      }

      Eigen::Vector3d a(pc_a->points[i].x, pc_a->points[i].y, pc_a->points[i].z);
      Eigen::Vector3d b(pc_b->points[j].x, pc_b->points[j].y, pc_b->points[j].z);
      Eigen::Vector3d v = b - a;
      if (std::abs(v.dot(na)) > 4.0)
      {
        continue;
      }


      if (boundaries_b->points[j].boundary_point)
      {
        nb_on_ctr++;
        distances.push_back(nearest_sq_dist.front());
      }
      nb_pts_ctr++;
      dbg.add_edge(pc_a->points[i], pc_b->points[j]);
      
    }
  }

  sort(distances.begin(), distances.end());
  dbg.save_obj("debug.obj");

  if (distances.size() > 10 && static_cast<double>(nb_on_ctr) / nb_pts_ctr > 0.9 && distances[10] < 16*16) 
  {
    return true;
  }
  else
  {
    return  false;
  }
}

Pointcloud::Ptr smooth_point_cloud(Pointcloud::Ptr pc)
{
  // Create a KD-Tree
  pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  Pointcloud::Ptr mls_points(new Pointcloud());
  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquares<Point, Point> mls;
  mls.setComputeNormals(true);
  mls.setInputCloud(pc);
  mls.setPolynomialOrder(2);
  mls.setSearchMethod(tree);
  mls.setNumberOfThreads(8);
  mls.setPolynomialFit(true);
  mls.setPointDensity(10);
  mls.setSqrGaussParam(4.0);
  mls.setSearchRadius(2.0);
  // Reconstruct
  mls.process (*mls_points);
  return mls_points;
}

Pointcloud::Ptr smooth_point_cloud2(Pointcloud::Ptr pc)
{
  // Create a KD-Tree
  pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  Pointcloud::Ptr mls_points(new Pointcloud());
  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquares<Point, Point> mls;
  //mls.setComputeNormals(true);
  mls.setInputCloud(pc);
  mls.setPolynomialOrder(2);
  mls.setSearchMethod(tree);
  mls.setNumberOfThreads(8);
  mls.setPolynomialFit(true);
  mls.setPointDensity(50);
  mls.setSqrGaussParam(2.0);
  mls.setSearchRadius(10.0);
  // Reconstruct
  mls.process (*mls_points);
  return mls_points;
}

auto triangulate(Pointcloud::Ptr pc)
{
 //   pcl::PointCloud<pcl::Normal>::Ptr normals = compute_normals(pc);
    pcl::search::Search<Point>::Ptr tree_n (new pcl::search::KdTree<Point>);
    tree_n->setInputCloud(pc);
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<Point, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree_n);
    normal_estimator.setInputCloud(pc);
    normal_estimator.setKSearch(10);
    // normal_estimator.setRadiusSearch(5);
    normal_estimator.setViewPoint(-10, -10, -10);
    normal_estimator.compute(*normals);
  

    pcl::PointCloud<pcl::PointNormal>::Ptr pc_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*pc, *normals, *pc_with_normals);

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>);
    tree->setInputCloud(pc_with_normals);

    //Initialize objects for triangulation
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp;
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh());

    //Max distance between connecting edge points
    gp.setSearchRadius(100);
    gp.setMu(1.5);
    gp.setConsistentVertexOrdering(true);
    //gp.setMaximumNearestNeighbors(30);
    //gp.setMaximumSurfaceAngle(4*M_PI/8); // 45 degrees
    gp.setMinimumAngle(M_PI/18); // 10 degrees
    gp.setMaximumAngle(2*M_PI/3); // 120 degrees
    //gp.setNormalConsistency(false);

    // Get result
    gp.setInputCloud(pc_with_normals);
    gp.setSearchMethod(tree);
    gp.reconstruct(*triangles);
    return triangles;


    // pcl::MarchingCubesRBF<pcl::PointNormal> mc;
    // mc.setInputCloud(pc_with_normals);
    // mc.setGridResolution(50, 50, 50);
    // mc.setSearchMethod(tree);
    // mc.reconstruct(*triangles);
    // return triangles;

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_euclidean_clusters(Pointcloud::Ptr pc)
{
  pcl::EuclideanClusterExtraction<Point> ec;
  std::vector<pcl::PointIndices> clusters;
  ec.setInputCloud(pc);
  ec.setClusterTolerance(2.0);
  ec.extract(clusters);
  
  srand (static_cast<unsigned int> (time (0)));
  std::vector<unsigned char> colors;
  for (size_t i_segment = 0; i_segment < clusters.size (); i_segment++)
  {
    colors.push_back(static_cast<unsigned char> (rand () % 256));
    colors.push_back(static_cast<unsigned char> (rand () % 256));
    colors.push_back(static_cast<unsigned char> (rand () % 256));
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_pc(new pcl::PointCloud<pcl::PointXYZRGB>());
  colored_pc->resize(pc->size());
  for (int cluster_id = 0; cluster_id < clusters.size(); ++cluster_id)
  {
    auto& v = clusters[cluster_id];
    for (auto i : v.indices)
    {
      colored_pc->points[i].x = pc->points[i].x;
      colored_pc->points[i].y = pc->points[i].y;
      colored_pc->points[i].z = pc->points[i].z;
      colored_pc->points[i].r = colors[cluster_id * 3];
      colored_pc->points[i].g = colors[cluster_id * 3 + 1];
      colored_pc->points[i].b = colors[cluster_id * 3 + 2];
    }
  }
  return colored_pc;
}


double sq_L2_dist(Point const& a, Point const& b)
{
  return std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2);
}


std::pair<bool, std::vector<std::pair<int, int>>> 
fusion_attempt(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a, KDTree::Ptr tree_a,
               Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b, KDTree::Ptr tree_b, int aa, int bb)
{
  std::vector<std::pair<int, int>> links;
  Match_debugger dbg;
  int count_matches = 0;

  for (unsigned int idx_a = 0; idx_a < pc_a->size(); ++idx_a)
  {
    std::vector<float> sq_dists;
    std::vector<int> nearest_idx;
    tree_b->nearestKSearch(pc_a->points[idx_a], 1, nearest_idx, sq_dists);

    auto min_dist_idx = nearest_idx.front();

    if (sq_dists.front() < 500.0)
    {
      Eigen::Vector3d na(normals_a->points[idx_a].normal_x, normals_a->points[idx_a].normal_y, normals_a->points[idx_a].normal_z);
      Eigen::Vector3d nb(normals_b->points[min_dist_idx].normal_x, normals_b->points[min_dist_idx].normal_y, normals_b->points[min_dist_idx].normal_z);
      na.normalize();
      nb.normalize();
      double dot_prod = na.dot(nb);
      double angle_diff = 180 * std::acos(dot_prod) / PI;

      if (angle_diff < 10 || angle_diff > 170)
      {
        dbg.add_edge(pc_a->points[idx_a], pc_b->points[min_dist_idx]);
        links.push_back({idx_a, min_dist_idx});
        count_matches++;
      }
    }
  }
  std::cout << "matches = " << count_matches << "\n";
  if (count_matches >= 10) {
    dbg.add_raw_point_cloud(pc_a);
    dbg.add_raw_point_cloud(pc_b);
    dbg.save_obj("matches/match_" + std::to_string(aa) + "_" + std::to_string(bb) + ".obj");
    return {true, links};
  }
  else
    return {false, links}; 
}




Eigen::Vector3d pca(Pointcloud::Ptr pc)
{
  const unsigned int n_points = pc->size();
  Eigen::Matrix<double, Eigen::Dynamic, 3> pts(n_points, 3);
  Eigen::Vector3d center(0.0, 0.0, 0.0);
  for (unsigned int j = 0; j < n_points; ++j)
  {
    pts(j, 0) = pc->points[j].x;
    pts(j, 1) = pc->points[j].y;
    pts(j, 2) = pc->points[j].z;
    center += pts.row(j);
  }
  center /= n_points;
  pts = pts.rowwise() - center.transpose();
  Eigen::Matrix3d mat = (1.0 / n_points) * pts.transpose() * pts;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat);
  Eigen::Vector3d eig_vals = svd.singularValues();
  std::sort(eig_vals.data(), eig_vals.data() + eig_vals.size(), std::greater<double>());
  return eig_vals;
}


void saveTS(const std::string& filename, Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces)
{
  std::ofstream file(filename);
  for (unsigned int i = 0; i < pc->size(); ++i)
  {
    file << "VRTX " << i+1 << " " << pc->points[i].x << " " << pc->points[i].y << " " << pc->points[i].z << "\n";
  }

  for (auto& f : faces)
  {
    if (f.vertices.size() != 3)
        std::cout << f.vertices.size() << "\n";
    file << "TRGL " << f.vertices[0]+1 << " " << f.vertices[1]+1 << " " << f.vertices[2]+1 << "\n";
  }

  file.close();
}

std::pair<Pointcloud::Ptr, pcl::PointCloud<pcl::Normal>::Ptr>
 interpolate_meshes(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a, 
                        Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b,
                        const std::vector<std::pair<int, int>>& links)
{
  Pointcloud::Ptr new_pc(new Pointcloud());
  pcl::PointCloud<pcl::Normal>::Ptr new_normals(new pcl::PointCloud<pcl::Normal>());
  unsigned int nb = 10;
  for (auto& p : links)
  {
    auto a = p.first;
    auto b = p.second;
    const auto& pa = pc_a->points[a];
    const auto& pb = pc_b->points[b];

    const auto& na = normals_a->points[a];

    double dx = pa.x - pb.x;
    double dy = pa.y - pb.y;
    double dz = pa.z - pb.z;

    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    dx /= nb;
    dy /= nb;
    dz /= nb;

    for (int i = 1 ; i < nb; ++i)
    {
      auto x = pb.x + i * dx;
      auto y = pb.y + i * dy;
      auto z = pb.z + i * dz;

      new_pc->push_back(Point(x, y, z));
      new_normals->push_back(na);
    }
    // TODO interpolte normal
  }
  return {new_pc, new_normals};
}
                        
std::vector<int> organize_contour(Pointcloud::Ptr pc, const std::vector<int>& ctr)
{
  int idx = 0;
  int start = idx;
  std::vector<int> starts = {start};
  std::vector<std::vector<int>> orders;
  orders.push_back({});
  std::vector<bool> done(ctr.size(), false);
  unsigned int k = 0;
  for (unsigned int k = 0; k < ctr.size(); ++k)
  {
    if (done[idx] && start == -1)
      continue; 
    double sq_min_dist = std::numeric_limits<double>::infinity();
    unsigned int sq_min_dist_idx = -1;
    for (unsigned int i = 0; i < ctr.size(); ++i)
    {
      if (i != idx && !done[i])
      {
        auto d = sq_L2_dist(pc->points[ctr[idx]], pc->points[ctr[i]]);
        if (d < sq_min_dist)
        {
          sq_min_dist = d;
          sq_min_dist_idx = i;
        }
      }
    }

    if (sq_min_dist > 1000) // if the next point is too far, it means we should create a new contour
    { // stuck

      if (start != -1)
      {
        // go in other direction from the start
        idx = start;
        start = -1;
      }
      else
      {
        // start a new contour
        int start_again = 0;
        while (start_again < ctr.size() && done[start_again])
          ++start_again;
        orders.push_back({});
        done[idx] = true;
        idx = start_again;
        start = start_again;
        starts.push_back(start);
        continue;
      }
    }
    orders.back().push_back(ctr[idx]);
    done[idx] = true;
    idx = sq_min_dist_idx;
  }

  int max_size = 0, max_size_idx = 0;
  for (unsigned int i = 0; i < orders.size(); ++i)
  {
    if (orders[i].size() > max_size)
    {
      max_size = orders[i].size();
      max_size_idx = i;
    }
  }

  if (max_size < 0.8 * ctr.size())
  {
    std::cerr << "Warning; order contour < 0.8 contour" << std::endl;
  }

  auto& best_order = orders[max_size_idx];
  int i = 0;
  while (i < best_order.size() && best_order[i] != best_order[0])
    ++i;
  if (i < best_order.size())
  {
    std::reverse(best_order.begin() + i, best_order.end());
  }
  return best_order;
}

Pointcloud::Ptr densify_contour(Pointcloud::Ptr pc, std::vector<int>& ctr, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  int idx = pc->size();
  Pointcloud::Ptr ctr_pc(new Pointcloud());
  int j = 0;
  //std::cout << "densify "<< ctr.size() << "\n";
  std::vector<int> new_ctr;
  for (unsigned int i = 0; i < ctr.size(); ++i)
  {
    //std::cout << i << " / " << ctr.size() << "\n";
    unsigned int i2 = (i+1) % ctr.size();
    Eigen::Vector3d a(pc->points[ctr[i]].x, pc->points[ctr[i]].y, pc->points[ctr[i]].z);
    Eigen::Vector3d b(pc->points[ctr[i2]].x, pc->points[ctr[i2]].y, pc->points[ctr[i2]].z);
    ctr_pc->push_back(pc->points[ctr[i]]);
    ctr_pc->push_back(pc->points[ctr[i2]]);

    if (sq_L2_dist(pc->points[ctr[i]], pc->points[ctr[i2]]) > 4.0)
     continue;
    Eigen::Vector3d d = b - a;
    d /= 10;
    for (int k = 1; k < 10; ++k)
    {
      Eigen::Vector3d p = a + k * d;
      pc->push_back(Point(p[0], p[1], p[2]));
      normals->push_back(normals->points[ctr[i]]);
      ctr_pc->push_back(Point(p[0], p[1], p[2]));
      new_ctr.push_back(k++);
    }
  }
  ctr.insert(ctr.end(), new_ctr.begin(), new_ctr.end());
  return ctr_pc;
}

Pointcloud::Ptr smooth_contour(Pointcloud::Ptr pc, const std::vector<int>& ctr)
{
  // Pointcloud::Ptr ctr_pc(new Pointcloud());
  // ctr_pc->resize(ctr.size());
  // for (unsigned int i = 0; i < ctr.size(); ++i)
  // { 
  //   ctr_pc->points[i] = pc->points[ctr[i]];
  // }

  Pointcloud::Ptr smooth_ctr(new Pointcloud());
  pcl::MovingLeastSquares<Point, Point> mls;
  mls.setComputeNormals(true);
  mls.setInputCloud(pc);
  mls.setPolynomialOrder(2);
   mls.setPolynomialFit(true);
  mls.setPointDensity(10);
  mls.setSqrGaussParam(4.0);
  mls.setSearchRadius(10.0);
  mls.process (*smooth_ctr);


  return smooth_ctr;
}


/// this functions does a BFS and keeps only the largest (in term of number of faces) connected component of the mesh
std::vector<pcl::Vertices> clean_mesh(Pointcloud::Ptr pc, std::vector<pcl::Vertices>& faces)
{
  std::vector<int> labels(pc->size(), -1);
  std::vector<std::vector<int>> edges(pc->size(), std::vector<int>());
  for (auto f : faces)
  {
    int a, b, c;
    a = f.vertices[0];
    b = f.vertices[1];
    c = f.vertices[2];
    edges[a].push_back(b);
    edges[a].push_back(c);
    edges[b].push_back(a);
    edges[b].push_back(c);
    edges[c].push_back(a);
    edges[c].push_back(b);
  }

  /// BFS search
  int cur = 0;
  int a = 0;
  while (cur < pc->size())
  {
    std::queue<int> q;
    q.push(cur);
    while (!q.empty())
    {
      auto x = q.front();
      q.pop();
      if (labels[x] != -1)
        continue;
      labels[x] = a;
      for (auto e : edges[x])
      {
        if (labels[e] == -1)
          q.push(e);
      }
    }
    ++a;
    while (cur < pc->size() && labels[cur] != -1)
      ++cur;
  }

  std::vector<int> count(a, 0);
  for (auto l : labels)
    count[l]++;
  int best_label = std::distance(count.begin(), std::max_element(count.begin(), count.end()));


  std::vector<bool> keep(faces.size(), true);
  for (unsigned int i = 0; i < faces.size(); ++i)
  {
    auto& f = faces[i].vertices;
    if (labels[f[0]] != best_label || labels[f[1]] != best_label || labels[f[2]] != best_label)
    {
      keep[i] = false;
    }
  }

  std::vector<pcl::Vertices> new_faces;
  for (unsigned int i = 0; i < faces.size(); ++i)
  {
    if (keep[i])
    {
      new_faces.push_back(faces[i]);
    }
  }

std::cout << "prev new faces " << faces.size() << " " << new_faces.size() << "\n";
  return new_faces;
}


/// This function first find the points of the mesh that are on the contour (i.e very close to points on the contour) and 
/// removes the faces that connect only points from the contours
std::vector<pcl::Vertices> remove_faces_from_contour(Pointcloud::Ptr pc, Pointcloud::Ptr contour, const std::vector<pcl::Vertices>& faces)
{

  KDTree::Ptr ctr_tree(new KDTree());
  ctr_tree->setInputCloud(contour);

  std::unordered_map<int, int> verts_on_ctr;
  std::vector<float> sq_dist;
  std::vector<int> nearest_idx;
  for (unsigned int i = 0; i < pc->size(); ++i)
  {
    ctr_tree->nearestKSearch(pc->points[i], 1, nearest_idx, sq_dist);
    if (sq_dist.front() < 4.0)
    {
      verts_on_ctr[i] = 1; // we do not use the associated values (just set it to 1)
    }
  }

  std::vector<pcl::Vertices> clean_faces;
  for (auto& f: faces)
  {
    if (verts_on_ctr.find(f.vertices[0]) != verts_on_ctr.end() &&
        verts_on_ctr.find(f.vertices[1]) != verts_on_ctr.end() &&
        verts_on_ctr.find(f.vertices[2]) != verts_on_ctr.end())
        {
          continue;
        }
      clean_faces.push_back(f);
  }

  return clean_faces;
}
std::vector<int> order_points(Pointcloud::Ptr pc, int center_idx, const Eigen::Vector3d& normal, const std::vector<int>& indices)
{
  std::vector<int> order = indices;
  Eigen::Matrix<double, Eigen::Dynamic, 3> points(indices.size(), 3);
  for (unsigned int i = 0; i < indices.size(); ++i)
  {
    points(i, 0) = pc->points[indices[i]].x;
    points(i, 1) = pc->points[indices[i]].y;
    points(i, 2) = pc->points[indices[i]].z;
  }

  Eigen::Vector3d center(pc->points[center_idx].x, pc->points[center_idx].y, pc->points[center_idx].z);
  points.rowwise() -= center.transpose();

  Eigen::Vector3d v = points.row(0).transpose();
  v.normalize();
  Eigen::Vector3d y_axis = normal.cross(v);
  Eigen::Vector3d x_axis = y_axis.cross(normal);

  Eigen::Matrix<double, Eigen::Dynamic, 1> x_proj = points * x_axis;
  Eigen::Matrix<double, Eigen::Dynamic, 1> y_proj = points * y_axis;

  std::vector<double> angle(indices.size());
  for (unsigned int i = 0; i < indices.size(); ++i)
  {
    angle[i] = std::atan2(y_proj(i, 0), x_proj(i, 0));
  }

  
  std::sort(order.begin(), order.end(), [&angle] (double a, double b) {
    return a < b;
  });
  return order;
}

#define create_pair(x, y) std::make_pair(std::min(x, y), std::max(x, y))

std::vector<pcl::Vertices> surface_triangulation(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  KDTree::Ptr tree(new KDTree());
  tree->setInputCloud(pc);

  double radius = 3.0;

  double cx = 0.0, cy = 0.0, cz = 0.0;
  for (auto p : *pc)
  {
    cx += p.x;
    cy += p.y;
    cz += p.z;
  }
  cx /= pc->size();
  cy /= pc->size();
  cz /= pc->size();

  std::vector<float> tmp_dist;
  std::vector<int> closest;
  tree->radiusSearch(Point(cx, cy, cz), 10, closest, tmp_dist);
  
  std::vector<bool> done(pc->size(), false);
  std::queue<int> q;
  q.push(closest.front());
  std::vector<int> neighbours;
  std::vector<float> sq_distances;
  std::map<std::pair<int, int>, bool> edges_map;
  std::vector<pcl::Vertices> faces;
  while (!q.empty())
  {
    auto x = q.front();
    q.pop();
    if (done[x])
      continue;
    done[x] = true;
    tree->nearestKSearch(pc->points[x], 5, neighbours, sq_distances);
    neighbours.erase(neighbours.begin());

    std::cout << "faces.size = " << faces.size() << "\n";

    Eigen::Vector3d n(normals->points[x].normal_x, normals->points[x].normal_y, normals->points[x].normal_z);

    if (neighbours.size() > 1)
    {
      auto neighbours_order = order_points(pc, x, n, neighbours);
      for (int k = 0; k < neighbours_order.size(); ++k)
      {
        pcl::Vertices f;
        f.vertices = {x, neighbours_order[k], neighbours_order[(k+1)%neighbours_order.size()]};
        auto e0 = create_pair(x, neighbours_order[k]);
        auto e1 = create_pair(neighbours_order[k], neighbours_order[(k+1)%neighbours_order.size()]);
        auto e2 = create_pair(x, neighbours_order[(k+1)%neighbours_order.size()]);

        auto it0 = edges_map.find(e0);
        auto it1 = edges_map.find(e1);
        auto it2 = edges_map.find(e2);

        if (it0 == edges_map.end() || it1 == edges_map.end() || it2 == edges_map.end())
        {
          faces.emplace_back(f);
        }
        if (it0 == edges_map.end())
          edges_map[e0] = true;
        if (it1 == edges_map.end())
          edges_map[e1] = true;
        if (it2 == edges_map.end())
          edges_map[e2] = true;

        if (!done[neighbours_order[k]])
        {
          q.push(neighbours_order[k]);
        }
      }
    }
  }

  return faces;  
}


std::vector<pcl::Vertices> remove_large_faces(Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces)
{
  std::vector<pcl::Vertices> small_faces;
  double threshold = 10.0;
  for (auto& f : faces)
  {
    auto v0 = f.vertices[0];
    auto v1 = f.vertices[1];
    auto v2 = f.vertices[2];

    if (sq_L2_dist(pc->points[v0], pc->points[v1]) > threshold ||
        sq_L2_dist(pc->points[v0], pc->points[v1]) > threshold ||
        sq_L2_dist(pc->points[v0], pc->points[v1]) > threshold)
        {
          continue;
        }
    small_faces.push_back(f);
  }
  return small_faces;
}


void saveOBJ(const std::string& filename, Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces)
{
  std::ofstream file(filename);
  for (auto p : *pc)
  {
    file << "v " << p.x << " " << p.y << " " << p.z << "\n";
  }

  for (auto& f : faces)
  {
    file << "f " << f.vertices[0]+1 << " " << f.vertices[1]+1 << " " << f.vertices[2]+1 << "\n";
  }

  file.close();
}

std::vector<pcl::Vertices> force_triangular_faces(Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& quad_faces)
{
  std::vector<pcl::Vertices> triangles(quad_faces.size() * 2);
  for (unsigned int i = 0; i < quad_faces.size(); ++i)
  {
    const auto& quad = quad_faces[i].vertices;
    double d1 = sq_L2_dist(pc->points[quad[0]], pc->points[quad[2]]);
    double d2 = sq_L2_dist(pc->points[quad[1]], pc->points[quad[3]]);
    pcl::Vertices f1;
    pcl::Vertices f2;
    if (d1 > d2)
    {
      f1.vertices = {quad[0], quad[1], quad[3]};
      f2.vertices = {quad[1], quad[2], quad[3]};
    }
    else
    {
      f1.vertices = {quad[0], quad[1], quad[2]};
      f2.vertices = {quad[2], quad[3], quad[0]};
    }
    triangles[i*2] = f1;
    triangles[i*2+1] = f2;
  }
  return triangles;
}

int main(int argc, char* argv[])
{
  srand (time(NULL));
  pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);


  Pointcloud::Ptr pc(new Pointcloud());
  pcl::io::loadPLYFile("groups_meshes/group_0_smooth_pc.ply", *pc);
    std::cout << pc->size() << "\n";
  auto pc_normals = compute_normals(pc);
  pcl::PointCloud<pcl::PointNormal>::Ptr pc_with_normals(new pcl::PointCloud<pcl::PointNormal>());
  pcl::concatenateFields(*pc, *pc_normals, *pc_with_normals);


  pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());

  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp;
  gp.setInputCloud(pc_with_normals);
  gp.setMu(2.0);
  gp.setSearchRadius(10.0);
  gp.reconstruct(*mesh);

  pcl::PolygonMesh::Ptr mesh2(new pcl::PolygonMesh());
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
  tree->setInputCloud(pc_with_normals);
  pcl::GridProjection<pcl::PointNormal> gp2;
  gp2.setInputCloud(pc_with_normals);
  gp2.setResolution(0.01);
  gp2.setPaddingSize(3);
  gp2.setMaxBinarySearchLevel(5);
  gp2.setSearchMethod(tree);
  gp2.reconstruct(*mesh2);

  std::cout << mesh->polygons.size() << "\n";
  std::cout << mesh2->polygons.size() << "\n";

  Pointcloud::Ptr tmp(new Pointcloud());
  pcl::fromPCLPointCloud2(mesh2->cloud, *tmp);

  auto triangles = force_triangular_faces(tmp, mesh2->polygons);
  mesh2->polygons = triangles;
  std::cout << pc_with_normals->size() << " " << tmp->size() << "\n";
  pcl::io::saveOBJFile("mesh_test.obj", *mesh);
  pcl::io::saveOBJFile("mesh_test2.obj", *mesh2);


  return 0;

}
