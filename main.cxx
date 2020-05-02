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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/boundary.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/segmentation/extract_clusters.h>

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
  est.setRadiusSearch(10.0);
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


auto triangulate(Pointcloud::Ptr pc)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals = compute_normals(pc);
    pcl::PointCloud<pcl::PointNormal>::Ptr pc_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*pc, *normals, *pc_with_normals);

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>);
    tree->setInputCloud(pc_with_normals);

    //Initialize objects for triangulation
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp;
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh());

    //Max distance between connecting edge points
    gp.setSearchRadius(5);
    gp.setMu(1);
    gp.setMaximumNearestNeighbors(10);
    gp.setMaximumSurfaceAngle(M_PI/8); // 45 degrees
    gp.setMinimumAngle(M_PI/18); // 10 degrees
    gp.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp.setNormalConsistency(false);

    // Get result
    gp.setInputCloud(pc_with_normals);
    gp.setSearchMethod(tree);
    gp.reconstruct(*triangles);
    return triangles;
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


bool fusion_attempt(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a, const std::vector<int>& ctr_a,
                    Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b, const std::vector<int>& ctr_b, int aa, int bb)
{

  Match_debugger dbg;
  int count_matches = 0;
  for (unsigned int i = 0; i < ctr_a.size(); ++i)
  {
    double min_dist = std::numeric_limits<double>::infinity();
    int min_dist_idx = -1;
    int idx_a = ctr_a[i];
    for (unsigned int j = 0; j < ctr_b.size(); ++j)
    {
      int idx_b = ctr_b[j];
      double d = sq_L2_dist(pc_a->points[idx_a], pc_b->points[idx_b]);
      if (d < min_dist)
      {
        min_dist = d;
        min_dist_idx = idx_b;
      }
    }


    if (min_dist < 200.0)
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
        count_matches++;
      }
    }
  }

  if (count_matches >= 10) {
    dbg.add_raw_point_cloud(pc_a);
    dbg.add_raw_point_cloud(pc_b);
    dbg.save_obj("matches/match_" + std::to_string(aa) + "_" + std::to_string(bb) + ".obj");
    return true;
  }
  else
    return false; 
}


int main(int argc, char* argv[])
{
  srand (time(NULL));
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  
  /* ---- Keep only planar points ------ */
  
  Pointcloud::Ptr pc(new Pointcloud());
  if (pcl::io::loadOBJFile("../out.obj", *pc) != 0)
  {
    throw std::runtime_error("Could not load point cloud");
  }



  auto smoothed_pc = smooth_point_cloud(pc);
  pcl::io::savePLYFile("smoothed.ply", *smoothed_pc);
  pc = smoothed_pc;


  pcl::PointCloud<pcl::Normal>::Ptr normals = compute_normals(pc);
  pcl::PointCloud<pcl::PointNormal>::Ptr pc_with_normals(new pcl::PointCloud<pcl::PointNormal>());
  pcl::concatenateFields(*pc, *normals, *pc_with_normals);
  pcl::io::savePLYFile("smoothed_with_normals.ply", *pc_with_normals);


  
  
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(pc);
  reg.setMinClusterSize(3);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(6);
  reg.setInputCloud(pc);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(10.0 / 180.0 * M_PI);
  //reg.setCurvatureThreshold(2.0);

  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);
  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();

  pcl::io::savePLYFile("regions_growing.ply", *colored_cloud); 


  /**
   * to continue, I need to merge some aprts together
   * for that:
   *  - look at each pair of parts
   *  - extract the contour (optionnaly interpolate the contour points to make it more dense)
   *  - try to find pair of points that has dist < threshold and angle(normal1, normal2) < threshold_angle, 
   *    (optionally, compute a score and sort so that one point can only be paired to one point in the otehr contour)
   *  - if there are sufficient connections, then merge the two parts together
   *  - while merging interpolate points  along the links between the contours
   * 
   *  - triangulate the big parts to create a mesh
   *  - apply hard smoothing before or after the triangulation
   * 
   * 
   * 
   * ** further checking to validate a fusion:
   *  - keep the "other" points and 
   *  - sample points along the found connections and check that they are "very" close to some "other" points
   * 
   * */


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
    if (v.indices.size() < 20) continue;
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
  pcl::io::savePLYFile("regions_growing_clusters.ply", *colored_pc); 


  std::vector<Pointcloud::Ptr> parts_pc; 
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> parts_normals;
  std::vector<std::vector<int>> parts_boundaries;
  for (unsigned int i = 0; i < clusters.size(); ++i)
  {
    auto& v = clusters[i];
    if (v.indices.size() < 20) continue;
    Pointcloud::Ptr p(new Pointcloud());
    pcl::PointCloud<pcl::Normal>::Ptr norms(new pcl::PointCloud<pcl::Normal>());
    p->resize(v.indices.size());
    norms->resize(v.indices.size());
    for (unsigned int j = 0; j < v.indices.size(); ++j)
    {
      p->points[j] = pc->points[v.indices[j]];
      norms->points[j] = normals->points[v.indices[j]];
    }
    parts_pc.push_back(p);
    parts_normals.push_back(norms);
    auto [ctr, bounds] = detect_contour_points2(p, norms);
    parts_boundaries.push_back(ctr);

    //if (parts_pc.size()  > 2) break;
  }


  for (unsigned int i = 0; i < parts_pc.size(); ++i)
  {
    pcl::io::savePLYFile("parts/part_" + std::to_string(i) + ".ply", *parts_pc[i]);
  }

  const unsigned int n_parts = parts_pc.size();
  std::vector<std::vector<int>> edges(n_parts, std::vector<int>());

  for (unsigned int a = 0; a < n_parts; ++a)
  {
    for (unsigned int b = a+1; b < n_parts; ++b)
    {
      auto ret = fusion_attempt(parts_pc[a], parts_normals[a], parts_boundaries[a],
                                parts_pc[b], parts_normals[b], parts_boundaries[b], a, b);
      std::cout << a << " " << b << "\n";
      if (ret)
      {
        edges[a].push_back(b);
        edges[b].push_back(a);
      }
    }
  }

  // BFS to create groups in an efficient way
  std::vector<int> groups(n_parts, -1);
  int g = 0;
  for (unsigned int i = 0; i < n_parts; ++i)
  {
    if (groups[i] != -1) 
      continue;
    std::queue<int> q;
    q.push(i);
    while (!q.empty())
    {
      auto x = q.front();
      q.pop();
      if (groups[x] != -1)
        continue;
      groups[x] = g;
      for (auto e : edges[x])
      {
        q.push(e);
      }
    }
    ++g;
  }

  std::vector<unsigned char> groups_colors;
  for (size_t i = 0; i < g; i++)
  {
    groups_colors.push_back(static_cast<unsigned char> (rand () % 256));
    groups_colors.push_back(static_cast<unsigned char> (rand () % 256));
    groups_colors.push_back(static_cast<unsigned char> (rand () % 256));
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_groups(new pcl::PointCloud<pcl::PointXYZRGB>());
  for (int i = 0; i < n_parts; ++i)
  {
    int group_idx = groups[i];

    for (unsigned int j = 0; j < parts_pc[i]->size(); ++j)
    {
      pcl::PointXYZRGB p;
      p.x = parts_pc[i]->points[j].x;
      p.y = parts_pc[i]->points[j].y;
      p.z = parts_pc[i]->points[j].z;
      p.r = groups_colors[group_idx * 3];
      p.g = groups_colors[group_idx * 3 + 1];
      p.b = groups_colors[group_idx * 3 + 2];
      colored_groups->push_back(p);
    }
  }

  pcl::io::savePLYFile("colored_groups.ply", *colored_groups);


  return 0;
  // auto mesh = triangulate(pc);
  // std::cout << "nb faces = " << mesh->polygons.size() << "\n";
  // pcl::io::savePLYFile("mesh.ply", *mesh);



//  Pointcloud::Ptr small_pc(new Pointcloud(*pc));
//  voxelGridFilter(small_pc, 2.0);
//  std::cout << pc->size() << "\n";
//  std::cout << small_pc->size() << "\n";
//  pcl::io::savePLYFile("filtered.ply", *pc);

  // auto [planar_pts, other_pts] = classify_points(pc, pc);
  // pcl::io::savePLYFile("planar_points.ply", *planar_pts);
  // pcl::io::savePLYFile("other_points.ply", *other_pts);


  // auto euclidean = extract_euclidean_clusters(planar_pts);
  // pcl::io::savePLYFile("euclidean_clusters.ply", *euclidean);
/*
  auto [clusters, colored_pc] = segment_point_cloud(planar_pts);
  pcl::io::savePLYFile("colored_pc.ply", *colored_pc);

  for (unsigned int i = 0; i < clusters.size(); ++i)
  {
    auto sub_pc(new Pointcloud());
    sub_pc->resize(clusters[i].indices.size());
    for (unsigned int j = 0; j < clusters[i].indices.size(); ++j)
    {
      sub_pc->points[j] = planar_pts->points[clusters[i].indices[j]];
    }
    pcl::io::savePLYFile("seg_" + std::to_string(i) + ".ply", *sub_pc);
  }
*/


  /* ------ Process parts ------*/

  // std::vector<Pointcloud::Ptr> parts;
  // std::vector<std::string> parts_filenames;
  // for(auto& p: fs::recursive_directory_iterator("parts"))
  // {
  //   if (p.path().extension().string() == ".obj")
  //   {
  //     parts_filenames.push_back(p.path().string());
  //   }
  // }

  // sort(parts_filenames.begin(), parts_filenames.end());

  // for (auto const& f : parts_filenames)
  // {
  //   Pointcloud::Ptr pc(new Pointcloud());
  //   pcl::io::loadOBJFile(f, *pc);
  //   parts.push_back(pc);
  // }

  // const int nb_parts = parts.size();
  // std::vector<Pointcloud::Ptr> parts_ctr(nb_parts);
  // std::vector<pcl::PointCloud<pcl::Normal>::Ptr> parts_normal(nb_parts);
  // std::vector<pcl::PointCloud<pcl::Boundary>::Ptr> parts_boundaries(nb_parts);
  // for (int i = 0; i < nb_parts; ++i)
  // {
  //   parts_normal[i] = compute_normals(parts[i]);
  //   auto [ctr, bounds] = detect_contour_points2(parts[i], parts_normal[i]);
  //   parts_ctr[i] = ctr;
  //   parts_boundaries[i] = bounds;
  //   pcl::io::savePLYFile("parts_ctr/parts_" + std::to_string(i) + ".ply", *parts_ctr[i]);
  // }


  // //int i = 3, j = 5;
  // //auto res = try_merge(parts[i], parts_normal[i], parts_boundaries[i],
  // //                     parts[j], parts_normal[j], parts_boundaries[j]);
  // //std::cout << "merge " << res << "\n";

  // return 0;

  // std::vector<int> labels(nb_parts, -1);

  // int a = 0;
  // for (int i = 1; i < nb_parts; ++i)
  // {
  //   if (labels[i] == -1)
  //   {
  //     labels[i] = a++;      
  //   }
  //   for (int j = i+1; j < nb_parts; ++j)
  //   {

  //     std::cout << "Try merge parts " << i << " and " << j << "\n";
  //     auto rep = try_merge(parts[i], parts_normal[i], parts_boundaries[i],
  //                          parts[j], parts_normal[j], parts_boundaries[j]);
  //     auto rep2 = try_merge(parts[j], parts_normal[j], parts_boundaries[j],
  //                           parts[i], parts_normal[i], parts_boundaries[i]);
  //     if (rep && rep2)
  //     {
  //       /*if (i == 1 && j == 17 || i == 3 && j == 27 || i == 1 && j == 28 || i == 1 && j == 58 || i == 2 && j == 17 || i == 9 && j == 37 || i == 10 && j == 15 ||
  //       i == 24 && j == 25 || i == 24 && j == 26)
  //         continue;
  //       else
  //         return 0;
  //       */  
  //       if (labels[j] == -1)
  //       {
  //         labels[j] = labels[i];
  //       }
  //       else
  //       {
  //         for (auto& x : labels)
  //         {
  //           if (x == labels[j])
  //             x = labels[i];
  //         }
  //       }
  //     }
  //   }
  // }

  // std::unordered_set<int> labels_set(labels.begin(), labels.end());
  // std::unordered_map<int, Pointcloud::Ptr> parts_map;
  // for (auto l : labels_set)
  // {
  //   parts_map[l].reset(new Pointcloud());
  // }

  // for (int i = 1; i < nb_parts; ++i)
  // {
  //   *(parts_map[labels[i]]) += *(parts[i]);
  // }

  // for (auto v : parts_map)
  // {
  //   pcl::io::savePLYFile("out/out_" + std::to_string(v.first) + ".ply", *v.second);
  // }

  // return 0;
}
