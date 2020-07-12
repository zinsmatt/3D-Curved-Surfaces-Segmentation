//=========================================================================
//
// Copyright 2020
// Author: Matthieu Zins
//
//=========================================================================

#include "processing.h"



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




std::pair<std::vector<int>, pcl::PointCloud<pcl::Boundary>::Ptr> detect_contour_points(Pointcloud::Ptr pc, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
  tree->setInputCloud(pc);
  pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>());
  pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
  est.setInputCloud(pc);
  est.setInputNormals(normals);
  est.setRadiusSearch(15.0);
  est.setSearchMethod(tree);
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





std::pair<Pointcloud::Ptr, pcl::PointCloud<pcl::Normal>::Ptr>
 interpolate_meshes(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a,
                        Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b,
                        const std::vector<std::pair<int, int>>& links)
{
  Pointcloud::Ptr new_pc(new Pointcloud());
  pcl::PointCloud<pcl::Normal>::Ptr new_normals(new pcl::PointCloud<pcl::Normal>());
  int nb = 10;
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
    // normal interpolation could be added
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
  for (int k = 0; k < static_cast<int>(ctr.size()); ++k)
  {
    if (done[idx] && start == -1)
      continue;
    double sq_min_dist = std::numeric_limits<double>::infinity();
    unsigned int sq_min_dist_idx = -1;
    for (int i = 0; i < static_cast<int>(ctr.size()); ++i)
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

    if (sq_min_dist > 1000) // if the next point is too far
    {

      if (start != -1)
      {
        // first, try to go in other direction from the start
        idx = start;
        start = -1;
      }
      else
      {
        // or start a new contour
        int start_again = 0;
        while (start_again < static_cast<int>(ctr.size()) && done[start_again])
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

  unsigned int max_size = 0, max_size_idx = 0;
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
    std::cerr << "Warning; ordered contour < 0.8 contour" << std::endl;
  }

  auto& best_order = orders[max_size_idx];
  unsigned int i = 0;
  while (i < best_order.size() && best_order[i] != best_order[0])
    ++i;
  if (i < best_order.size())
  {
    std::reverse(best_order.begin() + i, best_order.end());
  }
  return best_order;
}


std::vector<int> organize_contour2(Pointcloud::Ptr pc, const std::vector<int>& ctr)
{
  std::vector<int> best_order;
  std::vector<bool> done(ctr.size(), false);
  unsigned int s = 0;
  while (s < ctr.size())
  {
    while (s < ctr.size() && done[s])
      ++s;
    if (s >= ctr.size())
      break;
    std::vector<int> links;
    links.push_back(s);
    done[s] = true;
    for (unsigned int k = 0; k < ctr.size(); ++k)
    {
      int cur = links.back();

      // find the closest point to cur
      double sq_min_dist = std::numeric_limits<double>::infinity();
      int sq_min_dist_idx = -1;
      for (int i = 0; i < static_cast<int>(ctr.size()); ++i)
      {
        if (i != cur && !done[i])
        {
          auto d = sq_L2_dist(pc->points[ctr[cur]], pc->points[ctr[i]]);
          if (d < sq_min_dist)
          {
            sq_min_dist = d;
            sq_min_dist_idx = i;
          }
        }
      }

      if (sq_min_dist > 500)
        break;

      links.push_back(sq_min_dist_idx);
      done[sq_min_dist_idx] = true;
    }

    // second loop fo the other sens
    std::vector<int> links2;
    links2.push_back(s);
    for (unsigned int k = 0; k < ctr.size(); ++k)
    {
      int cur = links2.back();

      // find the closest point to cur
      double sq_min_dist = std::numeric_limits<double>::infinity();
      int sq_min_dist_idx = -1;
      for (int i = 0; i < static_cast<int>(ctr.size()); ++i)
      {
        if (i != cur && !done[i])
        {
          auto d = sq_L2_dist(pc->points[ctr[cur]], pc->points[ctr[i]]);
          if (d < sq_min_dist)
          {
            sq_min_dist = d;
            sq_min_dist_idx = i;
          }
        }
      }

      if (sq_min_dist > 500)
        break;

      links2.push_back(sq_min_dist_idx);
      done[sq_min_dist_idx] = true;
    }


    // merge the two links
    std::reverse(links2.begin(), links2.end());
    links2.pop_back();
    if (links2.size() > 0)
    {
      links.insert(links.end(), links2.begin(), links2.end());
    }


    if (links.size() >= best_order.size())
    {
      best_order = links;
    }
  }
  std::vector<int> points_indices(best_order.size());
  for (unsigned int i = 0; i < best_order.size(); ++i)
  {
    points_indices[i] = ctr[best_order[i]];
  }
  return points_indices;
}



std::pair<bool, std::vector<std::pair<int, int>>>
find_contours_matches(Pointcloud::Ptr pc_a, pcl::PointCloud<pcl::Normal>::Ptr normals_a, KDTree::Ptr tree_a,
                    Pointcloud::Ptr pc_b, pcl::PointCloud<pcl::Normal>::Ptr normals_b, KDTree::Ptr tree_b)
{
  std::vector<std::pair<int, int>> links;
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
                             pc_b->points[min_dist_idx].y - pc_a->points[idx_a].y,
                             pc_b->points[min_dist_idx].z - pc_a->points[idx_a].z);
      if (std::abs(vec_ab.dot(na)) > 5.0)
        continue;

      if (angle_diff < 10 || angle_diff > 170)
      {
        links.push_back({idx_a, min_dist_idx});
        count_matches++;
      }
    }
  }

  if (count_matches >= 10)
    return {true, links};
  else
    return {false, links};
}



/// This function removes faces which have no connections
std::vector<pcl::Vertices> remove_disconnected_faces(std::vector<pcl::Vertices>& faces)
{
  std::map<std::pair<int, int>, int> counter;

  for (auto f : faces)
  {
    auto v0 = f.vertices[0];
    auto v1 = f.vertices[1];
    auto v2 = f.vertices[2];

    auto e0 = create_pair(v0, v1);
    auto e1 = create_pair(v1, v2);
    auto e2 = create_pair(v0, v2);

    if (!counter.count(e0)) counter[e0] = 1;
    else ++counter[e0];
    if (!counter.count(e1)) counter[e1] = 1;
    else ++counter[e1];
    if (!counter.count(e2)) counter[e2] = 1;
    else ++counter[e2];
  }

  std::vector<pcl::Vertices> new_faces;
  for (auto f : faces)
  {
    auto v0 = f.vertices[0];
    auto v1 = f.vertices[1];
    auto v2 = f.vertices[2];

    auto e0 = create_pair(v0, v1);
    auto e1 = create_pair(v1, v2);
    auto e2 = create_pair(v0, v2);

    if (counter[e0] < 2 && counter[e1] < 2 && counter[e2] < 2)
    {
      continue;
    }
    else
    {
      new_faces.push_back(f);
    }
  }
  return new_faces;
}



std::vector<pcl::Vertices> keep_largest_connected_component(Pointcloud::Ptr pc, std::vector<pcl::Vertices>& faces)
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
  unsigned int cur = 0;
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


void smooth_contours(std::vector<Pointcloud::Ptr>& contours_pc)
{
  // Contour points smoothing
  // Could use weights
  // std::vector<double> weights = {1.0/16, 1.0/16, 2.0/16, 4.0/16, 0.0, 4.0/16, 2.0/16, 1.0/16, 1.0/16};
  for (unsigned int i = 0; i < contours_pc.size(); ++i)
  {
    Pointcloud::Ptr smooth(new Pointcloud());
    *smooth = *contours_pc[i];
    for (int j = 0; j < static_cast<int>(contours_pc[i]->size()); ++j)
    {
      double tot_x = 0.0;
      double tot_y = 0.0;
      double tot_z = 0.0;
      for (int d = -4; d <= 4; ++d)
      {
        int idx = (j + d) % static_cast<int>(contours_pc[i]->size());
        if (idx < 0)
          idx += static_cast<int>(contours_pc[i]->size());
        tot_x += contours_pc[i]->points[idx].x;
        tot_y += contours_pc[i]->points[idx].y;
        tot_z += contours_pc[i]->points[idx].z;
      }
      smooth->points[j].x = tot_x / 9;
      smooth->points[j].y = tot_y / 9;
      smooth->points[j].z = tot_z / 9;
    }
    *contours_pc[i] = *smooth;
  }
}


