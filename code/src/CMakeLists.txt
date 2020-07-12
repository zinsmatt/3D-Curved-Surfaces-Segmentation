#=========================================================================
#
# Copyright 2020
# Author: Matthieu Zins
#
#=========================================================================

cmake_minimum_required (VERSION 3.5.0)
project(3D_segmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)


set(SOURCES_DIR "./")
# Headers
set(${PROJECT_NAME}_headers
    ${SOURCES_DIR}/io.h
    ${SOURCES_DIR}/pcl_utils.h
    ${SOURCES_DIR}/processing.h
    )

# Sources
set(${PROJECT_NAME}_sources
    ${SOURCES_DIR}/io.cxx
    ${SOURCES_DIR}/main.cxx
    ${SOURCES_DIR}/pcl_utils.cxx
    ${SOURCES_DIR}/processing.cxx
    )


add_definitions(${PCL_DEFINITIONS})


add_executable(${PROJECT_NAME}
    ${${PROJECT_NAME}_sources}
    ${${PROJECT_NAME}_headers}
    )



target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${PCL_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_KDTREE_LIBRARY} ${PCL_SEARCH_LIBRARY} ${PCL_SURFACE_LIBRARY} ${PCL_FEATURES_LIBRARY}
    Eigen3::Eigen
    )

