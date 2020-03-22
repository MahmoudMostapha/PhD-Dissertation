#include <vtkSmartPointer.h>
#include <vtkCellLocator.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkIdList.h>
#include <vtkCell.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataReader.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

int main ( int argc, char *argv[] )
{
  // Ensure a filename was specified
  if(argc < 4)
  {
  std::cerr << "Usage: " << argv[0] << "InputVtkFileName OutputVtkFileName ArrayNumber" << endl;
  return EXIT_FAILURE;
  }

  // Read the Surfaces
  vtkSmartPointer<vtkPolyDataReader> reader1 =
  vtkSmartPointer<vtkPolyDataReader>::New();
  reader1->SetFileName(argv[1]);
  reader1->Update();
  vtkPolyData* InputSurface = reader1->GetOutput();
  std::cout << "Input Surface has " << InputSurface->GetNumberOfPoints() << " points." << std::endl;

  vtkIdType numberOfPointArrays = InputSurface->GetPointData()->GetNumberOfArrays();
  std::cout << "Number of PointData arrays: " << numberOfPointArrays << std::endl;

  int dataTypeID;
  for(vtkIdType i = 0; i < numberOfPointArrays; i++)
    {
    dataTypeID = InputSurface->GetPointData()->GetArray(i)->GetDataType();
    std::cout << "Array " << i << ": " << InputSurface->GetPointData()->GetArrayName(i)
              << " (type: " << dataTypeID << ")" << std::endl;
    }

  std::cout << "Selected Float Array Name is " << InputSurface->GetPointData()->GetArrayName(atoi(argv[3]))  << std::endl;
  vtkSmartPointer<vtkFloatArray> Array = vtkFloatArray::SafeDownCast(InputSurface->GetPointData()->GetArray(atoi(argv[3])));

  if (Array == 0)
     {
        std::cout << "Error Reading the array" << std::endl;
     }

  vtkSmartPointer<vtkPolyDataReader> reader2 =
  vtkSmartPointer<vtkPolyDataReader>::New();
  reader2->SetFileName(argv[2]);
  reader2->Update();
  vtkPolyData* OutputSurface = reader2->GetOutput();
  std::cout << "Output Surface has " << OutputSurface->GetNumberOfPoints() << " points." << std::endl;

  vtkSmartPointer<vtkFloatArray> NewArray = vtkSmartPointer<vtkFloatArray>::New();
  NewArray->SetNumberOfComponents(1);
  NewArray->SetName(InputSurface->GetPointData()->GetArrayName(atoi(argv[3])));

  // Create the tree
  vtkSmartPointer<vtkCellLocator> cellLocator = 
    vtkSmartPointer<vtkCellLocator>::New();
  cellLocator->SetDataSet(InputSurface);
  cellLocator->BuildLocator();

  for(vtkIdType i = 0; i < OutputSurface->GetNumberOfPoints(); i++)
  {
  //std::cout << "Current VertexId: " << i << std::endl;
  double CurrentPoint[3];
  OutputSurface->GetPoint(i,CurrentPoint);
  
  //Find the closest points to TestPoint
  double closestPoint[3];//the coordinates of the closest point will be returned here
  double closestPointDist2; //the squared distance to the closest point will be returned here
  vtkIdType cellId; //the cell id of the cell containing the closest point will be returned here
  int subId; //this is rarely used (in triangle strips only, I believe)
  cellLocator->FindClosestPoint(CurrentPoint, closestPoint, cellId, subId, closestPointDist2);
  
  //std::cout << "Coordinates of closest point: " << closestPoint[0] << " " << closestPoint[1] << " " << closestPoint[2] << std::endl;
  //std::cout << "Squared distance to closest point: " << closestPointDist2 << std::endl;
  //std::cout << "CellId: " << cellId << std::endl;

  vtkSmartPointer<vtkIdList> cellPointIds =
    vtkSmartPointer<vtkIdList>::New();
  InputSurface->GetCellPoints(cellId, cellPointIds);

  double AvgValue = 0.0;
  for(vtkIdType j = 0; j < cellPointIds->GetNumberOfIds(); j++)
  {
  //std::cout << "VertexId: " << cellPointIds->GetId(j) << std::endl;
  AvgValue = AvgValue + Array->GetValue(cellPointIds->GetId(j));
  }
  AvgValue = AvgValue / cellPointIds->GetNumberOfIds();
  NewArray->InsertNextValue(AvgValue);
  }	 

  OutputSurface->GetPointData()->AddArray(NewArray);

  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  polywriter->SetFileName(argv[2]);
  polywriter->SetInputData(OutputSurface);
  polywriter->Write();
  
  return EXIT_SUCCESS;
}
