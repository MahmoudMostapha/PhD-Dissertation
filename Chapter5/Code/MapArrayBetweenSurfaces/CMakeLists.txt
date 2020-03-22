
cmake_minimum_required(VERSION 2.8)
 
PROJECT(MapArrayBetweenSurfaces)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
add_executable(MapArrayBetweenSurfaces MACOSX_BUNDLE MapArrayBetweenSurfaces.cxx )
 
target_link_libraries(MapArrayBetweenSurfaces ${VTK_LIBRARIES})
