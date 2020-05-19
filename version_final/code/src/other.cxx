#include "other.h"

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

#include <vector>


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



Pointcloud::Ptr detect_contour_points_based_on_neighbourhood(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals)
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


pcl::PolygonMesh::Ptr triangulate_old(Pointcloud::Ptr pc)
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