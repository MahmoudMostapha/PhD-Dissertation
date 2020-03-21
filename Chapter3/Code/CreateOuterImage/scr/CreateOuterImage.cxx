#include "itkImage.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkImageFileReader.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageRegionIterator.h"
 
 
int main(int argc, char *argv[])
{
  if(argc < 3)
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " InputImageFile OutputImageFile [closingradius] [dilationradius] [Reverse]" << std::endl;
    return EXIT_FAILURE;
    }
 
  typedef itk::Image<unsigned char, 3>    ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);

  ImageType::SpacingType Spacing = reader->GetOutput()->GetSpacing();

  unsigned int radius = 60/Spacing[0];
  if (argc > 3)
    {
    radius = atoi(argv[3])/Spacing[0];
    }

  typedef itk::BinaryThresholdImageFilter <ImageType, ImageType>
    BinaryThresholdImageFilterType;
 
  BinaryThresholdImageFilterType::Pointer thresholdFilter
    = BinaryThresholdImageFilterType::New();
  thresholdFilter->SetInput(reader->GetOutput());
  thresholdFilter->SetLowerThreshold(1);
  thresholdFilter->SetUpperThreshold(4);
  thresholdFilter->SetInsideValue(1);
  thresholdFilter->SetOutsideValue(0);
  thresholdFilter->Update();

  ImageType::Pointer image = thresholdFilter->GetOutput();

  ImageType::SizeType regionSize;
  regionSize[0] = int(image->GetLargestPossibleRegion().GetSize()[0]/2);
  regionSize[1] = image->GetLargestPossibleRegion().GetSize()[1];
  regionSize[2] = image->GetLargestPossibleRegion().GetSize()[2];
 
  ImageType::IndexType regionIndex;
  regionIndex[0] = 0;
  if (argc == 6)
{
  regionIndex[0] = image->GetLargestPossibleRegion().GetSize()[0]/2;
}
  regionIndex[1] = 0;
  regionIndex[2] = 0;
 
  ImageType::RegionType region;
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  itk::ImageRegionIterator<ImageType> imageIterator(image,region);
 
  while(!imageIterator.IsAtEnd())
    {
    imageIterator.Set(0);
    ++imageIterator;
    }
 
/*
 typedef itk::ImageFileWriter< ImageType >  WriterType1;
  WriterType1::Pointer writer1 = WriterType1::New();
  writer1->SetFileName( "test.nrrd" );
  writer1->SetInput( image );
  writer1->UseCompressionOn();
  writer1->Update();
*/

  typedef itk::BinaryBallStructuringElement<
    ImageType::PixelType,3>                  StructuringElementType;
  StructuringElementType structuringElement;
  structuringElement.SetRadius(radius);
  structuringElement.CreateStructuringElement();

  typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;
  RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
  rescaleFilter->SetInput(image);
  rescaleFilter->SetOutputMinimum(0);
  rescaleFilter->SetOutputMaximum(255);
  rescaleFilter->Update();

  typedef itk::BinaryMorphologicalClosingImageFilter <ImageType, ImageType, StructuringElementType>
          BinaryMorphologicalClosingImageFilterType;
  BinaryMorphologicalClosingImageFilterType::Pointer closingFilter
          = BinaryMorphologicalClosingImageFilterType::New();
  closingFilter->SetInput(rescaleFilter->GetOutput());
  closingFilter->SetKernel(structuringElement);
  closingFilter->Update();

  radius = 5/Spacing[0];
  if (argc > 4)
    {
    radius = atoi(argv[4])/Spacing[0];
    }

  structuringElement.SetRadius(radius);
  structuringElement.CreateStructuringElement();

typedef itk::BinaryDilateImageFilter <ImageType, ImageType, StructuringElementType>
          BinaryDilateImageFilterType;
 
  BinaryDilateImageFilterType::Pointer dilateFilter
          = BinaryDilateImageFilterType::New();
  dilateFilter->SetInput(closingFilter->GetOutput());
  dilateFilter->SetKernel(structuringElement);
  dilateFilter->Update();

  rescaleFilter->SetInput(dilateFilter->GetOutput());
  rescaleFilter->SetOutputMinimum(0);
  rescaleFilter->SetOutputMaximum(1);
  rescaleFilter->Update();
 
  typedef itk::ImageFileWriter< ImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[2] );
  writer->SetInput( rescaleFilter->GetOutput() );
  writer->UseCompressionOn ();

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & e )
    {
    std::cerr << "Error: " << e << std::endl;
    return EXIT_FAILURE;
    }
 
  return EXIT_SUCCESS;
}
