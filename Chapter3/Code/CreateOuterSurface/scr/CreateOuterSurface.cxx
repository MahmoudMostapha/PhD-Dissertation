#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkPolyDataWriter.h>
#include <vtkMarchingCubes.h>
#include <vtkDecimatePro.h>
#include <vtkSmoothPolyDataFilter.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include <itkImageToVTKImageFilter.h>


int main(int argc, char ** argv)
{  

    if(argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << "InputBinaryImage OutputSurfacename NumberOfIterations" << endl;
        return EXIT_FAILURE;
    }

  const unsigned int Dimension = 3;
  typedef unsigned char PixelType;
  typedef itk::Image<PixelType, Dimension>  ImageType;
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();		
  reader->SetFileName(argv[1]);
  reader->Update();
  ImageType::Pointer image = reader->GetOutput();

  typedef itk::ImageToVTKImageFilter<ImageType> itkVtkConverter;
  itkVtkConverter::Pointer conv = itkVtkConverter::New();
  conv->SetInput(image);
  conv->Update(); 

  vtkSmartPointer<vtkMarchingCubes> outputsurface = 
  vtkSmartPointer<vtkMarchingCubes>::New();
  outputsurface->SetInputData(conv->GetOutput());
  outputsurface->ComputeNormalsOn();
  outputsurface->SetValue(0, 1);
  outputsurface->Update();
  std::cout << "Marching Cube finished...." << std::endl;
 
  std::cout << "Before decimation" << std::endl << "------------" << std::endl;
  std::cout << "There are " << outputsurface->GetOutput()->GetNumberOfPoints() << " points." << std::endl;
  std::cout << "There are " << outputsurface->GetOutput()->GetNumberOfPolys() << " polygons." << std::endl;
 

  vtkSmartPointer<vtkDecimatePro> decimate =
  vtkSmartPointer<vtkDecimatePro>::New();
  decimate->SetInputData(outputsurface->GetOutput());
  decimate->SetTargetReduction(.1); //10% reduction (if there was 100 triangles, now there will be 90)
  decimate->Update();
 
  std::cout << "After decimation" << std::endl << "------------" << std::endl;
  std::cout << "There are " << decimate->GetOutput()->GetNumberOfPoints() << " points." << std::endl;
  std::cout << "There are " << decimate->GetOutput()->GetNumberOfPolys() << " polygons." << std::endl;

  vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter =
  vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
  smoothFilter->SetInputConnection( decimate->GetOutputPort());
    smoothFilter->SetNumberOfIterations(atoi(argv[3]));
    smoothFilter->SetRelaxationFactor(0.5);
    smoothFilter->FeatureEdgeSmoothingOff();
    smoothFilter->BoundarySmoothingOn();
  smoothFilter->Update();
  std::cout << "VTK Smoothing mesh finished...." << std::endl;
  vtkPolyData* SmoothedPolyData = smoothFilter->GetOutput();

  // Get the Surface filename from the command line

  std::string outputSurfaceFilename = argv[2];

  vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
  writer->SetFileName(outputSurfaceFilename.c_str());
  writer->SetInputData(SmoothedPolyData);
  writer->SetFileTypeToASCII();
  writer->Write();

  return EXIT_SUCCESS;
}

