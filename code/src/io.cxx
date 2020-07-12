#include "io.h"


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
