//=========================================================================
//
// Copyright 2020
// Author: Matthieu Zins
//
//=========================================================================

#ifndef PROCESSING_H
#define PROCESSING_H

#include <fstream>

#include <pcl/features/boundary.h>

#include <Eigen/Dense>

#include "pcl_utils.h"



/// This structure is used to keep matching information between sub-groups
struct Match_ab
{
  int a;    // index of sub-group a
  int b;    // index of sub-group b
  int x;    // direct index of sub-group b

  std::vector<std::pair<int, int>> links_ab;        // indices of matched points in matching a to b
  std::vector<std::pair<int, int>> links_ba;        // indices of matched points in matching b to a
  Match_ab(int aa, int bb, int xx,  const std::vector<std::pair<int, int>>& la, const std::vector<std::pair<int, int>>& lb) :
    a(aa), b(bb), x(xx), links_ab(la), links_ba(lb)
    {}
  Match_ab() {}
};



/**
 * @brief remove_non_manifold_faces: this function removes each face which has an edge with more
 * than 2 neighbouring faces
 * @param pc
 * @param faces
 * @return new list of only manifold faces
 */
std::vector<pcl::Vertices> remove_non_manifold_faces(Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces);


/**
 * @brief detect_contour_points: this function detect the boundary points of a point cloud
 * @param pc
 * @param normals
 * @return [list of boundary points indices, pcl::pointcloud of pcl::Boundary indicating if a point is
 * a boundary point of not]
 */
std::pair<std::vector<int>, pcl::PointCloud<pcl::Boundary>::Ptr> detect_contour_points(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals);


/**
 * @brief interpolate_meshes: this function interpolates new points between two pointcloud based on
 * pairs of matched points in the two point clouds.
 * @param pc_a
 * @param normals_a
 * @param pc_b
 * @param normals_b
 * @param links
 * @return pair of new points and the corresponding normals
 */
std::pair<Pointcloud::Ptr, pcl::PointCloud<pcl::Normal>::Ptr>
 interpolate_meshes(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a,
                    Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b,
                    const std::vector<std::pair<int, int>>& links);


/**
 * @brief organize_contour: this function arrange the contour points in a geometrically meaningful order
 * @param pc
 * @param ctr: indices of the contour points
 * @return re-ordered indices of the contour points
 */
std::vector<int> organize_contour(Pointcloud::Ptr pc, const std::vector<int>& ctr);


/**
 * @brief organize_contour2: clearner version of organize_contour
 * @param pc
 * @param ctr: indices of the contour points
 * @return re-ordered indices of the contour points
 */
std::vector<int> organize_contour2(Pointcloud::Ptr pc, const std::vector<int>& ctr);


/**
 * @brief find_contours_links: this function finds links between two contours by matching points based on distances and normals
 * @param pc_a
 * @param normals_a
 * @param tree_a
 * @param pc_b
 * @param normals_b
 * @param tree_b
 * @return [boolean indicating if enough links were found, a list of pairs of indices corresponding
 *  to the matched points in each point cloud]
 */
std::pair<bool, std::vector<std::pair<int, int>>>
find_contours_matches(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a, KDTree::Ptr tree_a,
                    Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b, KDTree::Ptr tree_b);


/**
 * @brief remove_disconnected_faces: this function remove disconnected faces (i.e faces without any connected neighbour)
 * @param pc
 * @param faces
 * @return a new list of faces
 */
std::vector<pcl::Vertices> remove_disconnected_faces(std::vector<pcl::Vertices>& faces);


/**
 * @brief keep_largest_connected_component: this functions explores the mesh similarly to a graph
 *  and keeps only the largest connected component
 * @param pc
 * @param faces
 * @return a list of the faces which are in the largest connected component
 */
std::vector<pcl::Vertices> keep_largest_connected_component(Pointcloud::Ptr pc, std::vector<pcl::Vertices>& faces);



/**
 * @brief smooth_contours: this function smoothes (in-place) the contour points with a kind of Gaussian kernel.
 * Similar to Gaussian blurring of a 2D image. Warning: the contour points have to be ordered
 * in a geometrically meaningful way.
 * @param contours_pc
 */
void smooth_contours(std::vector<Pointcloud::Ptr>& contours_pc);




#endif // PROCESSING_H
