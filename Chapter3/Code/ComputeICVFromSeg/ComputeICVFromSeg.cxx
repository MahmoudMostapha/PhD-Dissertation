#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>        // std::abs
#include <string>
#include <limits>       // std::numeric_limits
#include <ctime>// include this header 
#include <algorithm>// include this header 


int main ( int argc, char *argv[] )
{
  // Ensure a filename was specified
  if(argc < 3)
    {
    std::cerr << "Usage: " << argv[0] << "InputSegmentationFileName OutputFilename" << std::endl;
    return EXIT_FAILURE;
    }

  int start_s=clock();

  // Get the segmentation filename from the command line
  std::string inputMaskFilename = argv[1];
  const unsigned int Dimension = 3;
  typedef double                      PixelType;
  typedef itk::Image< PixelType, Dimension > ImageType;
  typedef itk::ImageFileReader< ImageType >  MaskReaderType;
  MaskReaderType::Pointer Maskreader = MaskReaderType::New();
  Maskreader->SetFileName(inputMaskFilename.c_str());
  Maskreader->Update();
  ImageType::Pointer inputMask = Maskreader->GetOutput();

  typedef itk::ImageRegionIterator< ImageType> IteratorType;
  IteratorType inputIt(inputMask, inputMask->GetRequestedRegion());
  int voulme = 0;
  inputIt.GoToBegin();
  while (!inputIt.IsAtEnd())
   {
        if (inputIt.Get() > 0)
	  {
	     voulme+= 1;
          }
	++inputIt;
   }

  std::string FileName = argv[2];
  std::ofstream Result;
  Result.open (FileName.c_str());
  Result << voulme << std::endl;

  int stop_s=clock();
  std::cout << "time: " << (((float)(stop_s-start_s))/CLOCKS_PER_SEC)/60 <<" min" << std::endl;
  return EXIT_SUCCESS;

}