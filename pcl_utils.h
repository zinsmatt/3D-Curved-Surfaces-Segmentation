#ifndef PCLUTILS_H
#define PCLUTILS_H

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/Vertices.h>

#include <Eigen/Dense>

#include <fstream>

#define PI 3.14159265359
#define create_pair(x, y) std::make_pair(std::min(x, y), std::max(x, y))


using Point = pcl::PointXYZ;
using Pointcloud = pcl::PointCloud<Point>;
using KDTree = pcl::KdTreeFLANN<Point>;


void voxelGridFilter(Pointcloud::Ptr& pc, double voxelSize);

pcl::PointCloud<pcl::Normal>::Ptr compute_normals_from_radius(Pointcloud::Ptr pc, double radius=5.0);

pcl::PointCloud<pcl::Normal>::Ptr compute_normals_from_nearest(Pointcloud::Ptr pc, double k=10);

Pointcloud::Ptr smooth_point_cloud(Pointcloud::Ptr pc, double radius, int polynomial_order, int point_density, double sqr_gauss_param);

Eigen::Vector3d pca(Pointcloud::Ptr pc);


inline double sq_L2_dist(Point const& a, Point const& b)
{
  return std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2);
}



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
    for (unsigned int i = 1; i <= points.size() / 2; i += 2)
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

#endif // PCLUTILS_H
