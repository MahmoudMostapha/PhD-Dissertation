project( vtkMeshGeodesicExamples )

find_package( VTK REQUIRED )
include( ${VTK_USE_FILE} )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../Source  )

add_executable( GeodesicRings GeodesicRings.cxx )
target_link_libraries( GeodesicRings vtkMeshGeodesics ${VTK_LIBRARIES} )

add_executable( VisualizeGrid VisualizeGrid.cxx )
target_link_libraries( VisualizeGrid vtkMeshGeodesics ${VTK_LIBRARIES} )

add_executable( MultiResolutionSurfaceMappring MultiResolutionSurfaceMappring.cxx )
target_link_libraries( MultiResolutionSurfaceMappring vtkMeshGeodesics ${VTK_LIBRARIES} )

add_executable( GenerateSurfaceMask GenerateSurfaceMask.cxx )
target_link_libraries( GenerateSurfaceMask vtkMeshGeodesics ${VTK_LIBRARIES} )

