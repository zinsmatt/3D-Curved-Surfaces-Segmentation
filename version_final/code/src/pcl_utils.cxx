#include "pcl_utils.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>

#include <thread>

void voxelGridFilter(Pointcloud::Ptr& pc, double voxelSize)
{
  pcl::VoxelGrid<Point> vox_grid;
  vox_grid.setInputCloud(pc);
  vox_grid.setLeafSize(voxelSize, voxelSize, voxelSize);
  Pointcloud::Ptr tempCloud (new Pointcloud);
  vox_grid.filter(*tempCloud);
  pc.swap(tempCloud);
}



pcl::PointCloud<pcl::Normal>::Ptr compute_normals_from_radius(Pointcloud::Ptr pc, double radius)
{
  pcl::search::Search<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimationOMP<Point, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(pc);
  normal_estimator.setNumberOfThreads(std::thread::hardware_concurrency()); // set to the number of availbale cores
  normal_estimator.setRadiusSearch(radius);
  normal_estimator.setViewPoint(-100, -100, -100);
  normal_estimator.compute(*normals);
 
  return normals;
}

pcl::PointCloud<pcl::Normal>::Ptr compute_normals_from_nearest(Pointcloud::Ptr pc, double k)
{
  pcl::search::Search<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimationOMP<Point, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(pc);
  normal_estimator.setNumberOfThreads(std::thread::hardware_concurrency()); // set to the number of availbale cores
  normal_estimator.setKSearch(k);
  normal_estimator.setViewPoint(-100, -100, -100);
  normal_estimator.compute(*normals);
 
  return normals;
}


Pointcloud::Ptr smooth_point_cloud(Pointcloud::Ptr pc, double radius, int polynomial_order, int point_density, double sqr_gauss_param)
{
  // Create a KD-Tree
  pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  tree->setInputCloud(pc);
  Pointcloud::Ptr mls_points(new Pointcloud());
  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquaresOMP<Point, Point> mls;
  mls.setComputeNormals(true);
  mls.setInputCloud(pc);
  mls.setSearchRadius(radius);
  mls.setPolynomialOrder(polynomial_order);
  mls.setSearchMethod(tree);
  mls.setNumberOfThreads(std::thread::hardware_concurrency());
  mls.setPointDensity(point_density);
  mls.setSqrGaussParam(sqr_gauss_param);
  // Reconstruct
  mls.process (*mls_points);
  return mls_points;
}





Eigen::Vector3d pca_axes(Pointcloud::Ptr pc)
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

  Eigen::EigenSolver<Eigen::Matrix3d> eig(mat);
  Eigen::Vector3d eig_vals = eig.eigenvalues().real();
  std::sort(eig_vals.data(), eig_vals.data() + eig_vals.size(), std::greater<double>());
  Eigen::Vector3d axes = eig_vals.cwiseSqrt();
  return 4 * axes;  // 2 times for the diameter and 2 times for 2 stddev
}
