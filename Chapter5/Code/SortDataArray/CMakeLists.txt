
cmake_minimum_required(VERSION 2.8)
 
PROJECT(SortDataArray)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
add_executable(SortDataArray MACOSX_BUNDLE SortDataArray.cxx )
 
target_link_libraries(SortDataArray ${VTK_LIBRARIES})
