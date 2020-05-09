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
#include <thread>
#include <chrono>

 
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

#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkFillHolesFilter.h>

#include <Eigen/Dense>

#include "pcl_utils.h"
#include "io.h"

namespace fs = std::filesystem;


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

      Eigen::Vector3d vec_ab(pc_b->points[min_dist_idx].x - pc_a->points[idx_a].x,
                             pc_b->points[min_dist_idx].x - pc_a->points[idx_a].y,
                             pc_b->points[min_dist_idx].x - pc_a->points[idx_a].z);
     // if (std::abs(vec_ab.dot(na)) > 5.0)
     //   continue;

      if (angle_diff < 10 || angle_diff > 170)
      {
        dbg.add_edge(pc_a->points[idx_a], pc_b->points[min_dist_idx]);
        links.push_back({idx_a, min_dist_idx});
        count_matches++;
      }
    }
  }
  //std::cout << "matches = " << count_matches << "\n";
  if (count_matches >= 10) {
    dbg.add_raw_point_cloud(pc_a);
    dbg.add_raw_point_cloud(pc_b);
    dbg.save_obj("matches/match_" + std::to_string(aa) + "_" + std::to_string(bb) + ".obj");
    return {true, links};
  }
  else
    return {false, links}; 
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

  return new_faces;
}




std::vector<pcl::Vertices> remove_non_manifold_faces(Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces)
{
  std::map<std::pair<int ,int>, int> edges;
  std::vector<pcl::Vertices> good_faces;
  for (auto& f : faces)
  {
    auto v0 = f.vertices[0];
    auto v1 = f.vertices[1];
    auto v2 = f.vertices[2];

    auto e0 = create_pair(v0, v1);
    auto e1 = create_pair(v1, v2);
    auto e2 = create_pair(v0, v2);

    auto it0 = edges.find(e0);
    auto it1 = edges.find(e1);
    auto it2 = edges.find(e2);

    if (it0 != edges.end() && edges[e0] >= 2)
    {
      continue;
    }
    else
    {
      if (it0 == edges.end())
        edges[e0] = 1;
      else
        edges[e0]++;
    }

    if (it1 != edges.end() && edges[e1] >= 2)
    {
      continue;
    }
    else
    {
      if (it1 == edges.end())
        edges[e1] = 1;
      else
        edges[e1]++;
    }

    if (it2 != edges.end() && edges[e2] >= 2)
    {
      continue;
    }
    else
    {
      if (it2 == edges.end())
        edges[e2] = 1;
      else
        edges[e2]++;
    }
    good_faces.push_back(f);
  }
  return good_faces;
}

void analyze_non_manifold_edges(const std::vector<pcl::Vertices>& faces)
{
  std::map<std::pair<int, int>, int> edges;
  for (auto& f : faces)
  {
    auto v0 = f.vertices[0];
    auto v1 = f.vertices[1];
    auto v2 = f.vertices[2];

    auto e0 = create_pair(v0, v1);
    auto e1 = create_pair(v0, v2);
    auto e2 = create_pair(v1, v2);

    if (edges.find(e0) == edges.end())
      edges[e0] = 1;
    else
      edges[e0]++;

    if (edges.find(e1) == edges.end())
      edges[e1] = 1;
    else
      edges[e1]++;

    if (edges.find(e2) == edges.end())
      edges[e2] = 1;
    else
      edges[e2]++;
  }

  for (auto& p : edges)
  {
    if (p.second > 2)
    {
      std::cout << p.first.first << " " << p.first.second << " appears " << p.second << " times" << std::endl;
    }
  }
}


void print_step(const std::string& msg)
{
  std::cout << "---> " << msg << " ...\n";
}



int main(int argc, char* argv[])
{
  std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

  srand (time(NULL));
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  std::string input_file("/home/matt/dev/Seismic_3D_Volume/training/inputs/t09.obj");
  std::string output_folder("/home/matt/dev/Seismic_3D_Volume/results/t09/");
  if (argc > 1)
  {
    input_file = std::string(argv[1]);
  }
  if (argc > 2)
  {
    output_folder = std::string(argv[2]);
    if (output_folder[output_folder.size()-1] != '/')
      output_folder += '/';

  }
  std::cout << "Process " << input_file << std::endl;
  std::cout << "Output folder = " << output_folder << std::endl;

  
  /* ---- Keep only planar points ------ */
  
  Pointcloud::Ptr pc(new Pointcloud());
  if (pcl::io::loadOBJFile(input_file, *pc) != 0)
  {
    throw std::runtime_error("Could not load point cloud");
  }


  // Smooth point cloud
  print_step("Initial smoothing");
  auto smoothed_pc = smooth_point_cloud(pc, 2.0, 2, 10, 4.0);
  pc = smoothed_pc;
  //pcl::io::savePLYFile("smoothed.ply", *smoothed_pc);



  // Region Growing Clustering based on curvature
  print_step("Region Growing Clustering absed on local Curvature");

  pcl::PointCloud<pcl::Normal>::Ptr normals = compute_normals_from_radius(pc);
  // pcl::PointCloud<pcl::PointNormal>::Ptr pc_with_normals(new pcl::PointCloud<pcl::PointNormal>());
  // pcl::concatenateFields(*pc, *normals, *pc_with_normals);
  // pcl::io::savePLYFile("smoothed_with_normals.ply", *pc_with_normals);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(pc);
  reg.setMinClusterSize(3);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(6);
  reg.setInputCloud(pc);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(7.0 / 180.0 * M_PI);
  //reg.setCurvatureThreshold(2.0);

  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  // save the point cloud with one color per cluster
  // pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
  // pcl::io::savePLYFile("regions_growing.ply", *colored_cloud); 

  // save only the cluster with at least a certain number of points
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



  // Extract each part separately and computer normals and boundaries
  print_step("Parts contours extraction");

  std::vector<Pointcloud::Ptr> parts_pc; 
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> parts_normals;
  std::vector<std::vector<int>> parts_boundaries;

  for (unsigned int i = 0; i < clusters.size(); ++i)
  {
    auto& v = clusters[i];
    if (v.indices.size() < 20)
      continue;
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

    // Save the contour points
    // Pointcloud::Ptr ctr_pc(new Pointcloud());
    // for (auto k : ctr)
    // {
    //   ctr_pc->push_back(p->points[k]);
    // }
    // pcl::io::savePLYFile("parts_ctr/part_" + std::to_string(i) + ".ply", *ctr_pc);

    // pcl::io::savePLYFile("parts/part_" + std::to_string(i) + ".ply", *parts_pc[i]);
  }
  const unsigned int n_parts = parts_pc.size();
  std::cout << "found " << n_parts << "parts.\n";



/******************  Process contours (define order + densify)  ***********************/

  print_step("Contours smoothing");

  // Organize contour and extract them as point cloud and normals
  std::vector<Pointcloud::Ptr> contours_pc;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> contours_normals;
  for (unsigned int i = 0; i < parts_pc.size(); ++i)
  {
    auto ordered_ctr = organize_contour(parts_pc[i], parts_boundaries[i]);
    Pointcloud::Ptr ctr_pc(new Pointcloud());
    pcl::PointCloud<pcl::Normal>::Ptr ctr_norms(new pcl::PointCloud<pcl::Normal>());
    ctr_pc->resize(ordered_ctr.size());
    ctr_norms->resize(ordered_ctr.size());
    for (int j = 0; j < ordered_ctr.size(); ++j)
    {
      ctr_pc->points[j] = parts_pc[i]->points[ordered_ctr[j]];
      ctr_norms->points[j] = parts_normals[i]->points[ordered_ctr[j]];
    }
    contours_pc.push_back(ctr_pc);
    contours_normals.push_back(ctr_norms);

    // pcl::io::savePLYFile("parts_ctr/part_" + std::to_string(i) + "_ordered.ply", *ctr_pc);
  }


  // mayeb should densify contour HERE

  // Contour points smoothing
  std::vector<double> weights = {1.0/16, 1.0/16, 2.0/16, 4.0/16, 0.0, 4.0/16, 2.0/16, 1.0/16, 1.0/16};
  for (unsigned int i = 0; i < n_parts; ++i)
  {
      Pointcloud::Ptr smooth(new Pointcloud());
      *smooth = *contours_pc[i];
      for (int j = 0; j < contours_pc[i]->size(); ++j)
      {
        double tot_x = 0.0;
        double tot_y = 0.0;
        double tot_z = 0.0;
        for (int d = -4; d <= 4; ++d)
        {
          int idx = (j + d) % contours_pc[i]->size();
          if (idx < 0)
            idx += contours_pc[i]->size();
          tot_x += contours_pc[i]->points[idx].x; 
          tot_y += contours_pc[i]->points[idx].y; 
          tot_z += contours_pc[i]->points[idx].z; 
        }
        smooth->points[j].x = tot_x / 9;
        smooth->points[j].y = tot_y / 9;
        smooth->points[j].z = tot_z / 9;
      }
      *contours_pc[i] = *smooth;

      // pcl::io::savePLYFile("parts_ctr/part_" + std::to_string(i) + "_smooth.ply", *contours_pc[i]);
  }


  // Creation of contour trees for fast search
  std::vector<KDTree::Ptr> contours_trees;
  for (unsigned int i = 0; i < n_parts; ++i)
  {
    KDTree::Ptr tree(new KDTree());
    tree->setInputCloud(contours_pc[i]);
    contours_trees.push_back(tree);
  }




  // Matching between parts
  print_step("Parts matching");
  std::vector<std::vector<int>> edges(n_parts, std::vector<int>());
  unsigned int nb_fusions = 0;
  for (unsigned int a = 0; a < n_parts; ++a)
  {
    for (unsigned int b = a+1; b < n_parts; ++b)
    {
      //std::cout << a << " " << b << "\n";
      auto [ret1, links1] = fusion_attempt(contours_pc[a], contours_normals[a], contours_trees[a],
                                         contours_pc[b], contours_normals[b], contours_trees[b], a, b);
      auto [ret2, links2] = fusion_attempt(contours_pc[b], contours_normals[b], contours_trees[b],
                                           contours_pc[a], contours_normals[a], contours_trees[a], b, a);
      
      if (ret1 && ret2)
      {      
        auto [new_pc, new_normals] = interpolate_meshes(contours_pc[a], contours_normals[a], contours_pc[b], contours_normals[b], links1);
        *parts_pc[a] += *new_pc;
        *parts_normals[a] += *new_normals;
        auto [new_pc2, new_normals2] = interpolate_meshes(contours_pc[b], contours_normals[b], contours_pc[a], contours_normals[a], links2);
        *parts_pc[b] += *new_pc2;
        *parts_normals[b] += *new_normals2;

        edges[a].push_back(b);
        edges[b].push_back(a);
        nb_fusions++;
      }
    }
  }
  std::cout << nb_fusions << " fusions\n";


  // BFS to identify groups of connected parts in an efficient way
  print_step("Connected groups extraction");
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

  // Save the point cloud with one color per group
  // std::vector<unsigned char> groups_colors;
  // for (size_t i = 0; i < g; i++)
  // {
  //   groups_colors.push_back(static_cast<unsigned char> (rand () % 256));
  //   groups_colors.push_back(static_cast<unsigned char> (rand () % 256));
  //   groups_colors.push_back(static_cast<unsigned char> (rand () % 256));
  // }
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_groups(new pcl::PointCloud<pcl::PointXYZRGB>());
  // for (int i = 0; i < n_parts; ++i)
  // {
  //   int group_idx = groups[i];

  //   for (unsigned int j = 0; j < parts_pc[i]->size(); ++j)
  //   {
  //     pcl::PointXYZRGB p;
  //     p.x = parts_pc[i]->points[j].x;
  //     p.y = parts_pc[i]->points[j].y;
  //     p.z = parts_pc[i]->points[j].z;
  //     p.r = groups_colors[group_idx * 3];
  //     p.g = groups_colors[group_idx * 3 + 1];
  //     p.b = groups_colors[group_idx * 3 + 2];
  //     colored_groups->push_back(p);
  //   }
  // }
  // pcl::io::savePLYFile("colored_groups.ply", *colored_groups);


  // Create point clouds and normals per group
  std::vector<Pointcloud::Ptr> groups_pc(g, nullptr);
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> groups_normals(g, nullptr);
  for (unsigned int i = 0; i < n_parts; ++i)
  {
    int gr = groups[i];
    if (!groups_pc[gr]) 
    {
      groups_pc[gr].reset(new Pointcloud());
      groups_normals[gr].reset(new pcl::PointCloud<pcl::Normal>());
    }

    *groups_pc[gr] += *parts_pc[i];
    *groups_normals[gr] += *parts_normals[i];
  }


  // Final filtering and smoothing before surface triangulation
  print_step("Final filtering, smoothing and triangulation");
  for (unsigned int i = 0; i < groups_pc.size(); ++i)
  {
    std::cout << "group " << i << ": \n";
    
    Eigen::Vector3d eig_vals_old = pca_axes_old(groups_pc[i]);
    std::cout << "------------------> old axes = " << eig_vals_old.transpose() << " ----> ";
    if (eig_vals_old[0] >= 100) std::cout << "PASSED\n";
    else std::cout << "NO\n";

    Eigen::Vector3d eig_vals = pca_axes(groups_pc[i]);
    std::cout << "------------------> new axes = " << eig_vals.transpose() << " ----> ";
    if (eig_vals[0] >= 100) std::cout << "PASSED\n";
    else std::cout << "NO\n\n";
    pcl::io::savePLYFile("groups_pc/group_" + std::to_string(i) + ".ply", *groups_pc[i]);
    if (eig_vals[0] >= 100)
    {
      voxelGridFilter(groups_pc[i], 3.0);
      auto smooth_pc = smooth_point_cloud(groups_pc[i], 10, 2, 50, 2.0);

      // pcl::io::savePLYFile("groups_meshes/group_" + std::to_string(i) + "_smooth_pc.ply", *smooth_pc);

      auto smooth_pc_normals = compute_normals_from_radius(smooth_pc);
      pcl::PointCloud<pcl::PointNormal>::Ptr smooth_pc_with_normals(new pcl::PointCloud<pcl::PointNormal>());
      pcl::concatenateFields(*smooth_pc, *smooth_pc_normals, *smooth_pc_with_normals);

      pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
      pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp;
      gp.setInputCloud(smooth_pc_with_normals);
      gp.setMu(2.0);
      gp.setSearchRadius(10.0);
      gp.reconstruct(*mesh);

      auto cleaned_faces = clean_mesh(smooth_pc, mesh->polygons);
      auto manifold_faces = remove_non_manifold_faces(smooth_pc, cleaned_faces);
      mesh->polygons = manifold_faces;
      // analyze_non_manifold_edges(manifold_faces);

      saveOBJ("groups_meshes/group_" + std::to_string(i) + "_clean_mesh_no_fill.obj", smooth_pc, manifold_faces);

      // MAYBE REENABLE AFTER
      // vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();
      // pcl::VTKUtils::convertToVTK(*mesh, pd);
      // vtkSmartPointer<vtkFillHolesFilter> fill_holes = vtkSmartPointer<vtkFillHolesFilter>::New();
      // fill_holes->SetInputData(pd);
      // fill_holes->SetHoleSize(100);
      // fill_holes->Update();
      // pcl::PolygonMesh::Ptr filled_mesh(new pcl::PolygonMesh());
      // vtkSmartPointer<vtkPolyData> filled_pd = vtkSmartPointer<vtkPolyData>::New();
      // filled_pd->DeepCopy(fill_holes->GetOutput());
      // pcl::VTKUtils::convertToPCL(filled_pd, *filled_mesh);

      // manifold_faces = remove_non_manifold_faces(smooth_pc, filled_mesh->polygons);
      // filled_mesh->polygons = manifold_faces;

      // pcl::io::saveOBJFile("groups_meshes/group_" + std::to_string(i) + "_clean_mesh.obj", *filled_mesh);
      // saveTS(output_folder + "group_" + std::to_string(i) + ".ts", smooth_pc, filled_mesh->polygons);
      
      saveTS(output_folder + "group_" + std::to_string(i) + ".ts", smooth_pc, manifold_faces);


      std::cout << manifold_faces.size() << " faces\n";
    }
    else 
    {
      std::cout << "Ignore group (too small)\n";

    }
  }

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
  std::cout << "Done in " << time_span.count() << " s.\n";

  return 0;

}
