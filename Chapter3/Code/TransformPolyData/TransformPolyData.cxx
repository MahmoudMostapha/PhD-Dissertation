#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkPolyDataWriter.h>


int main(int argc, char ** argv)
{  

    if(argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " InputSurfacename OutputSurfacename [ShiftX] [ShiftY] [ShiftZ]" << endl;
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
  // Set up the transform filter

  vtkSmartPointer<vtkTransform> translation =
    vtkSmartPointer<vtkTransform>::New();
  translation->Translate(atof(argv[3]), atof(argv[4]), atof(argv[5]));

  vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter =
    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  transformFilter->SetInputConnection(reader->GetOutputPort());
  transformFilter->SetTransform(translation);
  transformFilter->Update();

  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  polywriter->SetFileName(argv[2]);
  polywriter->SetInputData(transformFilter->GetOutput());
  polywriter->Write();

  return EXIT_SUCCESS;
}
