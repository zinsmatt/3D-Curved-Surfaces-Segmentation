//=========================================================================
//
// Copyright 2020
// Author: Matthieu Zins
//
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
#include <numeric>
#include <thread>
#include <chrono>


#include <pcl/features/boundary.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>

#include <Eigen/Dense>

#include "io.h"
#include "pcl_utils.h"
#include "processing.h"




/// This function does a second pass to re-fuse the parts of a group. It is called only on the groups that are suspected to contain multiple surfaces.
/// The fusion is more restrictive than in the first pass in order to split the different surfaces into different groups.
std::pair<std::vector<Pointcloud::Ptr>, std::vector<pcl::PointCloud<pcl::Normal>::Ptr>>
 second_pass_fusion(const std::vector<Pointcloud::Ptr>& parts_pc, const std::vector<pcl::PointCloud<pcl::Normal>::Ptr> parts_normals)
{
  const unsigned int n_parts = parts_pc.size();

  // Detect the parts contours
  print_step("Parts contours extraction");
  std::vector<std::vector<int>> parts_boundaries;
  for (unsigned int i = 0; i < n_parts; ++i)
  {
    auto [ctr, bounds] = detect_contour_points(parts_pc[i], parts_normals[i]);
    parts_boundaries.push_back(ctr);
  }

  // Organize contours and extract them as point cloud and normals
  print_step("Contours smoothing");
  std::vector<Pointcloud::Ptr> contours_pc;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> contours_normals;
  for (unsigned int i = 0; i < n_parts; ++i)
  {
    auto ordered_ctr = organize_contour(parts_pc[i], parts_boundaries[i]);
    Pointcloud::Ptr ctr_pc(new Pointcloud());
    pcl::PointCloud<pcl::Normal>::Ptr ctr_norms(new pcl::PointCloud<pcl::Normal>());
    ctr_pc->resize(ordered_ctr.size());
    ctr_norms->resize(ordered_ctr.size());
    for (unsigned int j = 0; j < ordered_ctr.size(); ++j)
    {
      ctr_pc->points[j] = parts_pc[i]->points[ordered_ctr[j]];
      ctr_norms->points[j] = parts_normals[i]->points[ordered_ctr[j]];
    }
    contours_pc.push_back(ctr_pc);
    contours_normals.push_back(ctr_norms);
  }

  // Contour points smoothing
  smooth_contours(contours_pc);

  // Creation of contour trees for fast search
  std::vector<KDTree::Ptr> contours_trees;
  for (unsigned int i = 0; i < n_parts; ++i)
  {
    KDTree::Ptr tree(new KDTree());
    tree->setInputCloud(contours_pc[i]);
    contours_trees.push_back(tree);
  }

  // Force clustering into N sub-groups. The parts are sorted by size and the initial N sub-groups are created from the N largest parts.
  // Then, each remainging parts are merged into the sub-group with the strongest links.
  int N = std::min(5, (int)n_parts);

  std::vector<int> idx(n_parts);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&parts_pc] (int a, int b) {
    return parts_pc[a]->size() > parts_pc[b]->size();
  });

  std::vector<std::vector<Pointcloud::Ptr>> per_groups_pc(N, std::vector<Pointcloud::Ptr>());
  std::vector<std::vector<pcl::PointCloud<pcl::Normal>::Ptr>> per_groups_normals(N, std::vector<pcl::PointCloud<pcl::Normal>::Ptr>());
  for (int i = 0; i < N; ++i)
  {
    per_groups_pc[i].push_back(parts_pc[idx[i]]);
    per_groups_normals[i].push_back(parts_normals[idx[i]]);
  }

  for (unsigned int i = N; i < n_parts; ++i)
  {
    int a = idx[i];

    std::vector<Match_ab> poss_matches;
    for (int k = 0; k < N; ++k)
    {
      int b = idx[k];
      auto [ret1, links1] = find_contours_matches(contours_pc[a], contours_normals[a], contours_trees[a],
                                           contours_pc[b], contours_normals[b], contours_trees[b]);
      auto [ret2, links2] = find_contours_matches(contours_pc[b], contours_normals[b], contours_trees[b],
                                           contours_pc[a], contours_normals[a], contours_trees[a]);
      if (ret1 && ret2)
      {
        poss_matches.emplace_back(a, b, k, links1, links2);
      }
    }

    if (poss_matches.size() == 0)
      continue;

    // Sort the possible matches by strength of links
    std::sort(poss_matches.begin(), poss_matches.end(), [] (const Match_ab& a, const Match_ab& b) {
      return a.links_ba.size() + a.links_ab.size() > b.links_ab.size() + b.links_ba.size();
    });

    // Do fusion with the first sub-group
    const auto& best_match = poss_matches.front();
    auto [new_pc, new_normals] = interpolate_meshes(contours_pc[best_match.a], contours_normals[best_match.a], contours_pc[best_match.b], contours_normals[best_match.b], best_match.links_ab);
    *parts_pc[best_match.a] += *new_pc;
    *parts_normals[best_match.a] += *new_normals;
    auto [new_pc2, new_normals2] = interpolate_meshes(contours_pc[best_match.b], contours_normals[best_match.b], contours_pc[best_match.a], contours_normals[best_match.a], best_match.links_ba);
    *parts_pc[best_match.a] += *new_pc2;
    *parts_normals[best_match.a] += *new_normals2;

    per_groups_pc[best_match.x].push_back(parts_pc[best_match.a]);
    per_groups_normals[best_match.x].push_back(parts_normals[best_match.a]);
  }


  // Recompose the sub-groups point cloud and normals
  std::vector<Pointcloud::Ptr> groups_pc(N, nullptr);
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> groups_normals(N, nullptr);
  for (int i = 0; i < N; ++i)
  {
    for (unsigned int j = 1; j < per_groups_pc[i].size(); ++j)
    {
      *per_groups_pc[i][0] += *per_groups_pc[i][j];
      *per_groups_normals[i][0] += *per_groups_normals[i][j];
    }
    groups_pc[i] = per_groups_pc[i][0];
    groups_normals[i] = per_groups_normals[i][0];
  }

  return {groups_pc, groups_normals};
}





int main(int argc, char* argv[])
{
  std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

  srand (time(NULL));
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  std::string input_file;
  std::string output_folder;
  if (argc != 3)
  {
    std::cout << "Usage: ./3D_segmentation input_file_obj output_folder" << std::endl;
    return -1;
  }

  input_file = std::string(argv[1]);
  output_folder = std::string(argv[2]);
  if (output_folder[output_folder.size()-1] != '/')
    output_folder += '/';

  std::cout << "Process " << input_file << std::endl;
  std::cout << "Output folder = " << output_folder << std::endl;



  /****** Start of processing ******/

  // Load point cloud
  Pointcloud::Ptr pc(new Pointcloud());
  if (pcl::io::loadOBJFile(input_file, *pc) != 0)
  {
    throw std::runtime_error("Could not load point cloud");
  }


  // Smooth point cloud
  print_step("Initial smoothing");
  auto smoothed_pc = smooth_point_cloud(pc, 2.0, 2, 10, 4.0);
  pc = smoothed_pc;


  // Remove invalid points
  Pointcloud::Ptr good_pc(new Pointcloud());
  good_pc->resize(pc->size());
  int j = 0;
  for (auto& p : *pc)
  {
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
    {
      good_pc->points[j++] = p;
    }
  }
  good_pc->resize(j);
  pc = good_pc;



  // Region Growing Clustering based on curvature
  print_step("Region Growing Clustering absed on local Curvature");

  pcl::PointCloud<pcl::Normal>::Ptr normals = compute_normals_from_radius(pc);
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

  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  // Remove too small clusters
  std::vector<pcl::PointIndices> filtered_clusters;
  for (const auto c : clusters)
  {
    if (c.indices.size() > 100)
    {
      filtered_clusters.push_back(c);
    }
  }
  clusters = filtered_clusters;


  // Extract each part separately and computer normals and boundaries
  print_step("Parts contours extraction");

  std::vector<Pointcloud::Ptr> parts_pc;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> parts_normals;
  std::vector<std::vector<int>> parts_boundaries;

  for (unsigned int i = 0; i < clusters.size(); ++i)
  {
    auto& v = clusters[i];

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
    auto [ctr, bounds] = detect_contour_points(p, norms);
    parts_boundaries.push_back(ctr);
  }
  const unsigned int n_parts = parts_pc.size();


  // Organize contours and extract them as point cloud and normals
  print_step("Contours smoothing");
  std::vector<Pointcloud::Ptr> contours_pc;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> contours_normals;
  for (unsigned int i = 0; i < parts_pc.size(); ++i)
  {
    auto ordered_ctr = organize_contour(parts_pc[i], parts_boundaries[i]);
    Pointcloud::Ptr ctr_pc(new Pointcloud());
    pcl::PointCloud<pcl::Normal>::Ptr ctr_norms(new pcl::PointCloud<pcl::Normal>());
    ctr_pc->resize(ordered_ctr.size());
    ctr_norms->resize(ordered_ctr.size());
    for (unsigned int j = 0; j < ordered_ctr.size(); ++j)
    {
      ctr_pc->points[j] = parts_pc[i]->points[ordered_ctr[j]];
      ctr_norms->points[j] = parts_normals[i]->points[ordered_ctr[j]];
    }
    contours_pc.push_back(ctr_pc);
    contours_normals.push_back(ctr_norms);
  }

  // Contours points smoothing
  smooth_contours(contours_pc);


  // Creation of contours trees for fast search
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
      auto [ret1, links1] = find_contours_matches(contours_pc[a], contours_normals[a], contours_trees[a],
                                           contours_pc[b], contours_normals[b], contours_trees[b]);
      auto [ret2, links2] = find_contours_matches(contours_pc[b], contours_normals[b], contours_trees[b],
                                           contours_pc[a], contours_normals[a], contours_trees[a]);

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


  // Second pass to try to split wrongs groups
  std::vector<Pointcloud::Ptr> group_refused_pc;
  for (unsigned int i = 0; i < groups_pc.size(); ++i)
  {

    auto norms = compute_normals_from_radius(groups_pc[i], 9);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(groups_pc[i]);
    reg.setMinClusterSize(3);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(6);
    reg.setInputCloud(groups_pc[i]);
    reg.setInputNormals(norms);
    reg.setSmoothnessThreshold(10.0 / 180.0 * M_PI);
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    std::vector<pcl::PointIndices> filtered_clusters;
    for (const auto c : clusters)
    {
      if (c.indices.size() > 300)
      {
        filtered_clusters.push_back(c);
      }
    }
    clusters = filtered_clusters;

    // Create the point cloud and normals of each sub-group
    std::vector<Pointcloud::Ptr> group_sub_pc;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> group_sub_normals;
    for (auto& v : clusters)
    {
      Pointcloud::Ptr pts(new Pointcloud());
      pcl::PointCloud<pcl::Normal>::Ptr norms(new pcl::PointCloud<pcl::Normal>());
      pts->resize(v.indices.size());
      norms->resize(v.indices.size());
      int a = 0;
      for (auto idx : v.indices)
      {
        pts->points[a] = groups_pc[i]->points[idx];
        norms->points[a] = groups_normals[i]->points[idx];
        ++a;
      }
      group_sub_pc.push_back(pts);
      group_sub_normals.push_back(norms);
    }

    // If the initial group has been split in enough sub-groups, we try to remerge them differently
    if (clusters.size() >= 7)
    {
      auto [refused_pc, refused_normals] = second_pass_fusion(group_sub_pc, group_sub_normals);
      group_refused_pc.insert(group_refused_pc.end(), refused_pc.begin(), refused_pc.end());
    }
    else
    {
      group_refused_pc.push_back(groups_pc[i]);
    }
  }


  groups_pc = group_refused_pc;

  // Final filtering and smoothing before surface triangulation
  print_step("Final filtering, smoothing and triangulation");
  for (unsigned int i = 0; i < groups_pc.size(); ++i)
  {
    std::cout << "group " << i << ": ";

    // Threshold on the groups global metric size
    Eigen::Vector3d eig_vals = pca_axes(groups_pc[i]);
    if (eig_vals[0] >= 65)
    {
      // Voxelize the point cloud and smooth it in order to facilitates the meshing
      voxelGridFilter(groups_pc[i], 3.0);
      auto smooth_pc = smooth_point_cloud(groups_pc[i], 10, 2, 50, 2.0);

      // Compute normals needed for the meshing algorithm
      auto smooth_pc_normals = compute_normals_from_radius(smooth_pc);
      pcl::PointCloud<pcl::PointNormal>::Ptr smooth_pc_with_normals(new pcl::PointCloud<pcl::PointNormal>());
      pcl::concatenateFields(*smooth_pc, *smooth_pc_normals, *smooth_pc_with_normals);

      pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
      pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp;
      gp.setInputCloud(smooth_pc_with_normals);
      gp.setMu(2.0);
      gp.setSearchRadius(10.0);
      gp.reconstruct(*mesh);

      // Final filtering to ensure valid meshes
      auto cleaned_faces = keep_largest_connected_component(smooth_pc, mesh->polygons);
      auto manifold_faces = remove_non_manifold_faces(smooth_pc,cleaned_faces);
      cleaned_faces = remove_disconnected_faces(manifold_faces);
      mesh->polygons = cleaned_faces;

      pcl::io::savePLYFile(output_folder + "group_" + std::to_string(i) + ".ply", *mesh);

      // saveTS(output_folder + "group_" + std::to_string(i) + ".ts", smooth_pc, cleaned_faces); // uncomment to directly save to TS format

      std::cout << mesh->polygons.size() << " faces\n";
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


