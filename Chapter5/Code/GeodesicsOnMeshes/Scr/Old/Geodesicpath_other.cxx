

#include "vtkSmartPointer.h"
#include <vtkPolyDataReader.h>

#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkTimerLog.h"
#include "vtkCamera.h"
#include "vtkProperty.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkPolyDataNormals.h"
#include "vtkRendererCollection.h"
#include "vtkPolyDataCollection.h"
#include "vtkObjectFactory.h"
#include "vtkIdList.h"
#include "vtkXMLPolyDataReader.h"
#include "vtkXMLPolyDataWriter.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkContourWidget.h"
#include "vtkOrientedGlyphContourRepresentation.h"
#include "vtkPolygonalSurfacePointPlacer.h"
#include "vtkPolygonalSurfaceContourLineInterpolator2.h"


int main(int argc, char*argv[])
{
  if (argc < 2)
    {
    std::cerr
      << "Demonstrates editing capabilities of a contour widget on polygonal \n"
      << "data.\n"
      << "Usage args: mesh.vtk [Method 0=Dijkstra,1=FastMarching] [InterpolationOrder 0=NearestNeighbor,1=Linear] [height_offset]."
      << std::endl;
    return EXIT_FAILURE;
    }


  // Get all surface data from the file
  vtkSmartPointer<vtkPolyDataReader> reader =
  vtkSmartPointer<vtkPolyDataReader>::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  vtkPolyData* inputPolyData = reader->GetOutput();
  std::cout << "Input surface has " << inputPolyData->GetNumberOfPoints() << " points." << std::endl;

  vtkNew<vtkPolyDataNormals> normals;

  const int geodesicMethod = (argc > 2 ? atoi(argv[2]) : 0);
  const int interpolationOrder = (argc > 3 ? atoi(argv[3]) : 0);
  const double distanceOffset = (argc > 4 ? atof(argv[4]) : 0);

  // We need to ensure that the dataset has normals if a distance offset was
  // specified.
  if (fabs(distanceOffset) > 1e-6)
    {
    normals->SetInputConnection(reader->GetOutputPort());
    normals->SplittingOff();

    // vtkPolygonalSurfacePointPlacer needs cell normals
    // vtkPolygonalSurfaceContourLineInterpolator needs vertex normals
    normals->ComputeCellNormalsOn();
    normals->ComputePointNormalsOn();
    normals->Update();
    }

  vtkPolyData *pd = (fabs(distanceOffset) > 1e-6) ? 
      normals->GetOutput() : reader->GetOutput();

  vtkNew<vtkPolyDataMapper> mapper;
  mapper->SetInputConnection(fabs(distanceOffset) > 1e-6 ? 
      normals->GetOutputPort() : reader->GetOutputPort());


  vtkNew<vtkPolygonalSurfaceContourLineInterpolator2> interpolator;
  interpolator->GetPolys()->AddItem( pd );
  interpolator->SetGeodesicMethod(geodesicMethod);
  interpolator->SetInterpolationOrder(interpolationOrder);
  if (fabs(distanceOffset) > 1e-6)
    {
    interpolator->SetDistanceOffset( distanceOffset );
    }

  interpolator->LastInterpolatedVertexIds[0] = -1;
  interpolator->LastInterpolatedVertexIds[1] = -1;



  return EXIT_SUCCESS;
}
