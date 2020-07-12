//=========================================================================
//
// Copyright 2020
// Author: Matthieu Zins
//
//=========================================================================

#ifndef IO_H
#define IO_H

#include "pcl_utils.h"


/**
 * @brief saveTS: this functions saves a mesh in TS format
 * @param filename
 * @param pc
 * @param faces
 */
void saveTS(const std::string& filename, Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces);


/**
 * @brief saveOBJ: this function saves a mesh in OBJ format
 * @param filename
 * @param pc
 * @param faces
 */
void saveOBJ(const std::string& filename, Pointcloud::Ptr pc, const std::vector<pcl::Vertices>& faces);

#endif // IO_H
