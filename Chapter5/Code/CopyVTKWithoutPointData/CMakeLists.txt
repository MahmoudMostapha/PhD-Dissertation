
cmake_minimum_required(VERSION 2.8)
 
PROJECT(CopyVTKWithoutPointData)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
add_executable(CopyVTKWithoutPointData MACOSX_BUNDLE CopyVTKWithoutPointData.cxx )
 
target_link_libraries(CopyVTKWithoutPointData ${VTK_LIBRARIES})
