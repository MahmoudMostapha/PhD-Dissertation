#include <vtkVersion.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>

int main(int argc, char* argv[])
{

if (argc < 3)
{
std::cerr << "Usage: " << argv[0] << "SurfaceMesh.vtk NormalThreshold" << std::endl;
return EXIT_FAILURE;
}

ofstream Result;
Result.open("Mask.csv");
  
// Read File
std::string inputSurfaceFilename = argv[1]; //first command line argument
std::cout << "Reading file " << inputSurfaceFilename << "..." << std::endl;

// Get all surface data from the file
vtkSmartPointer<vtkPolyDataReader> surfacereader =
vtkSmartPointer<vtkPolyDataReader>::New();
surfacereader->SetFileName(inputSurfaceFilename.c_str());
surfacereader->Update();

double Threshold = atof(argv[2]);

vtkPolyData* polydata = surfacereader->GetOutput();
std::cout << "Input surface has " << polydata->GetNumberOfPoints() << " points." << std::endl;

// Generate normals
vtkSmartPointer<vtkPolyDataNormals> normalGenerator = vtkSmartPointer<vtkPolyDataNormals>::New();
normalGenerator->SetInputData(polydata);
normalGenerator->ComputePointNormalsOn();
normalGenerator->ComputeCellNormalsOff();
normalGenerator->Update();
polydata = normalGenerator->GetOutput();

vtkFloatArray* normalDataFloat =
vtkFloatArray::SafeDownCast(polydata->GetPointData()->GetArray("Normals"));

if(normalDataFloat)
{
int nc = normalDataFloat->GetNumberOfTuples();
std::cout << "There are " << nc
    << " components in normalDataFloat" << std::endl;
}

vtkSmartPointer<vtkDoubleArray> Mask = vtkSmartPointer<vtkDoubleArray>::New();
Mask->SetNumberOfComponents(1);
Mask->SetName("Mask");

for(vtkIdType Vertex_ID = 0; Vertex_ID < polydata->GetNumberOfPoints(); Vertex_ID++)
{

double testDouble[3];
normalDataFloat->GetTuple(Vertex_ID, testDouble);

std::cout << "Double: " << testDouble[0] << " "
      << testDouble[1] << " " << testDouble[2] << std::endl;

if (testDouble[0] < Threshold)
{
Mask->InsertNextValue(0);	
Result << 0;
}
else
{
Mask->InsertNextValue(1);	
Result << 1;	
}
Result << endl;

}
Result.close();
polydata->GetPointData()->AddArray(Mask);

vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
polywriter->SetFileName("Mask.vtk");
polywriter->SetInputData(polydata);
polywriter->Write();
  
  return EXIT_SUCCESS;
}


