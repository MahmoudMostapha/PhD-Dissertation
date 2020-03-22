#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

int main(int argc, char* argv[])
{
  if (argc < 4)
    {
    std::cerr << "Usage: " << argv[0] << "SurfaceMesh.vtk CSVFileName.csv VertexID" << std::endl;
    return EXIT_FAILURE;
    }

  int ROWS = 7;
  int COLS = 7;
  int BUFFSIZE = 100;
  int array[ROWS][COLS];
  char buff[BUFFSIZE]; // a buffer to temporarily park the data
  ifstream infile(argv[2]);

for ( int i = 0; i <= atoi(argv[3]); ++i )
{
  stringstream ss;
  for( int row = 0; row < ROWS; ++row ) {
	infile.getline( buff,  BUFFSIZE );
	ss << buff;
	for( int col = 0; col < COLS; ++col ) {
	  ss.getline( buff, 10, ',' );
	  array[row][col] = atoi( buff );
	}
	ss << ""; 
	ss.clear();
  }
}

  infile.close();

  // Get all surface data from the file
  vtkSmartPointer<vtkPolyDataReader> surfacereader =
  vtkSmartPointer<vtkPolyDataReader>::New();
  surfacereader->SetFileName(argv[1]);
  surfacereader->Update();
  vtkPolyData* inputPolyData = surfacereader->GetOutput();
  std::cout << "Input surface has " << inputPolyData->GetNumberOfPoints() << " points." << std::endl;

  vtkSmartPointer<vtkFloatArray> R1 = vtkSmartPointer<vtkFloatArray>::New();
  R1->SetNumberOfComponents(1);
  R1->SetName("R1");

  vtkSmartPointer<vtkFloatArray> R2 = vtkSmartPointer<vtkFloatArray>::New();
  R2->SetNumberOfComponents(1);
  R2->SetName("R2");

  vtkSmartPointer<vtkFloatArray> R3 = vtkSmartPointer<vtkFloatArray>::New();
  R3->SetNumberOfComponents(1);
  R3->SetName("R3");

  vtkSmartPointer<vtkFloatArray> R4 = vtkSmartPointer<vtkFloatArray>::New();
  R4->SetNumberOfComponents(1);
  R4->SetName("R4");

  vtkSmartPointer<vtkFloatArray> R5 = vtkSmartPointer<vtkFloatArray>::New();
  R5->SetNumberOfComponents(1);
  R5->SetName("R5");

  vtkSmartPointer<vtkFloatArray> R6 = vtkSmartPointer<vtkFloatArray>::New();
  R6->SetNumberOfComponents(1);
  R6->SetName("R6");

  vtkSmartPointer<vtkFloatArray> R7 = vtkSmartPointer<vtkFloatArray>::New();
  R7->SetNumberOfComponents(1);
  R7->SetName("R7");

 for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
     {
       R1->InsertNextValue(0);
       R2->InsertNextValue(0);
       R3->InsertNextValue(0);
       R4->InsertNextValue(0);
       R5->InsertNextValue(0);
       R6->InsertNextValue(0);
       R7->InsertNextValue(0);
     } 

 for( int col = 1; col < COLS; ++col ) 
     {
       R1->SetValue(array[0][col], 1);
       R2->SetValue(array[1][col], 1);
       R3->SetValue(array[2][col], 1);
       R4->SetValue(array[3][col], 1);
       R5->SetValue(array[4][col], 1);
       R6->SetValue(array[5][col], 1);
       R7->SetValue(array[6][col], 1);
     }	

  inputPolyData->GetPointData()->AddArray(R1);
  inputPolyData->GetPointData()->AddArray(R2);
  inputPolyData->GetPointData()->AddArray(R3);
  inputPolyData->GetPointData()->AddArray(R4);
  inputPolyData->GetPointData()->AddArray(R5);
  inputPolyData->GetPointData()->AddArray(R6);
  inputPolyData->GetPointData()->AddArray(R7);

  // Write Results
  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  polywriter->SetFileName("Result.vtk");
  polywriter->SetInputData(inputPolyData);
  polywriter->Write();

}


