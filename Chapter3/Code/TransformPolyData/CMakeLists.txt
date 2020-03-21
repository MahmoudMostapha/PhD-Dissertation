
cmake_minimum_required(VERSION 2.8)

PROJECT(TransformPolyData)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(TransformPolyData MACOSX_BUNDLE TransformPolyData)

if(VTK_LIBRARIES)
  target_link_libraries(TransformPolyData ${VTK_LIBRARIES})
else()
  target_link_libraries(TransformPolyData vtkHybrid vtkWidgets)
endif()
