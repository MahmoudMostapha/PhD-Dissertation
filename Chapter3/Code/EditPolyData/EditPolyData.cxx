#include <vtkGenericDataObjectReader.h>
#include <vtkStructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPointData.h>
#include <vtkIdList.h>

#include <cmath>        // std::abs
#include <string>
#include <limits>       // std::numeric_limits

int main ( int argc, char *argv[] )
{
  // Ensure a filename was specified
  if(argc < 6)
    {
    std::cerr << "Usage: " << argv[0] << " InputSurfaceFileName OutputSurfaceFileName flipx flipy flipz" << endl;
    return EXIT_FAILURE;
    }

  // Get the Surface filename from the command line
  std::string inputSurfaceFilename = argv[1];

  // Get all surface data from the file
  vtkSmartPointer<vtkGenericDataObjectReader> reader = 
      vtkSmartPointer<vtkGenericDataObjectReader>::New();
  reader->SetFileName(inputSurfaceFilename.c_str());
  reader->Update();

  vtkPolyData* inputPolyData = reader->GetPolyDataOutput();
  std::cout << "Input surface has " << inputPolyData->GetNumberOfPoints() << " points." << std::endl;
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

 for(vtkIdType j = 0; j < inputPolyData->GetNumberOfPoints(); j++)
    {

	    double p[3];
	    inputPolyData->GetPoint(j,p);
            double P_M[3];
            P_M[0] = atoi(argv[3])*p[0];    // x coordinate
            P_M[1] = atoi(argv[4])*p[1];    // y coordinate
            P_M[2] = atoi(argv[5])*p[2];    // z coordinate
            points->InsertNextPoint(P_M);
    }

    inputPolyData->SetPoints(points);
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName(argv[2]);
    writer->SetInputData(inputPolyData);
    writer->Write();

    return EXIT_SUCCESS;
}
