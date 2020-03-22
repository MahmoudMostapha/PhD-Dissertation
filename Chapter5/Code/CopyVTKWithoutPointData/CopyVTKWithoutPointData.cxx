#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

int main ( int argc, char *argv[] )
{

  // Ensure a filename was specified
  if(argc < 3)
  {
  std::cerr << "Usage: " << argv[0] << "InputVtkFileName OutputVtkFileName" << endl;
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

  // Create a polydata object and add everything to it
  vtkSmartPointer<vtkPolyData> OutputSurface = vtkSmartPointer<vtkPolyData>::New();
  OutputSurface->SetPoints(InputSurface->GetPoints());
  OutputSurface->SetPolys(InputSurface->GetPolys());
  OutputSurface->GetPointData()->SetNormals(OutputSurface->GetPointData()->GetNormals());

  numberOfPointArrays = OutputSurface->GetPointData()->GetNumberOfArrays();
  std::cout << "Number of PointData arrays: " << numberOfPointArrays << std::endl;

  for(vtkIdType i = 0; i < numberOfPointArrays; i++)
    {
    dataTypeID = OutputSurface->GetPointData()->GetArray(i)->GetDataType();
    std::cout << "Array " << i << ": " << OutputSurface->GetPointData()->GetArrayName(i)
              << " (type: " << dataTypeID << ")" << std::endl;
    }

  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  polywriter->SetFileName(argv[2]);
  polywriter->SetInputData(OutputSurface);
  polywriter->Write();

  return EXIT_SUCCESS;
}
