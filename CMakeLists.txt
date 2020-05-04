#=========================================================================
#
# Copyright 2020
# Author: Matthieu Zins
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=========================================================================

cmake_minimum_required (VERSION 3.5.0)
project(3D_segmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)


add_definitions(${PCL_DEFINITIONS})
add_definitions("-DHAVE_QHULL")


add_executable(${PROJECT_NAME}
    main.cxx
    )

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${PCL_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_KDTREE_LIBRARY} ${PCL_SEARCH_LIBRARY} ${PCL_SURFACE_LIBRARY} ${PCL_FEATURES_LIBRARY}
    Eigen3::Eigen
    stdc++fs
    )