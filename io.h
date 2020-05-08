#ifndef IO_H
#define IO_H

#include "pcl_utils.h"

void saveTS(const std::string& filename, Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces);

void saveOBJ(const std::string& filename, Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces);

#endif // IO_H
