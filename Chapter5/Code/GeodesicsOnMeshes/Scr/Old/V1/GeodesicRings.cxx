#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include "vtkIdList.h"
#include "vtkFastMarchingGeodesicDistance.h"
#include <vtkPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkFastMarchingGeodesicPath.h>
#include <vtkContourFilter.h>
#include <vtkStripper.h>
#include <vtkSplineFilter.h>
#include <vtkKochanekSpline.h>
#include <vtkPointLocator.h>
#include <vtkLine.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkAppendPolyData.h>
#include <vtkSelectPolyData.h>
#include <vtkCenterOfMass.h>
#include <vtkMath.h>
#include <stdlib.h>     /* abs */
#include <math.h>       /* cos */
#define PI 3.14159265


int main(int argc, char* argv[])
{
  if (argc < 3)
    {
    std::cerr << "Usage: " << argv[0] << "SurfaceMesh.vtk SeedID" << std::endl;
    return EXIT_FAILURE;
    }

  // Get all surface data from the file
  vtkSmartPointer<vtkPolyDataReader> surfacereader =
  vtkSmartPointer<vtkPolyDataReader>::New();
  surfacereader->SetFileName(argv[1]);
  surfacereader->Update();
  vtkPolyData* inputPolyData = surfacereader->GetOutput();
  std::cout << "Input surface has " << inputPolyData->GetNumberOfPoints() << " points." << std::endl;

//-----------------------------------------------Estimate Local Orientation---------------------------------------------------------

  vtkSmartPointer<vtkFloatArray> Local_Direction = vtkSmartPointer<vtkFloatArray>::New();
  Local_Direction->SetNumberOfComponents(1);
  Local_Direction->SetName("Local_Direction");

  // Add the Seed
  vtkSmartPointer<vtkIdList> seed =
      vtkSmartPointer<vtkIdList>::New();
  seed->InsertNextId(atoi(argv[2]));
  std::cout << "Starting from Vertex " << seed->GetId(0) << std::endl;

  vtkSmartPointer<vtkFloatArray> Phi = vtkFloatArray::SafeDownCast(inputPolyData->GetPointData()->GetArray("Phi"));
  vtkSmartPointer<vtkFloatArray> Theta = vtkFloatArray::SafeDownCast(inputPolyData->GetPointData()->GetArray("Theta"));
  double lat1 = Phi->GetValue(seed->GetId(0));
  double lon1 = Theta->GetValue(seed->GetId(0));

  for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
     {

          double lat2 = Phi->GetValue(ID);
          double lon2 = Theta->GetValue(ID);
	  double dlat = lat2 - lat1;
	  double dlon = lon2 - lon1;
	  double y = sin(lon2-lon1)*cos(lat2);
	  double x = cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1);
	  double tc1;
	  if (y > 0) 
	    {
	    if (x > 0)  { tc1 = atan(y/x);}
	    if (x < 0)  { tc1 = PI - atan(-y/x);}
	    if (x == 0) { tc1 = PI/2;}
	    }
	  if (y < 0) 
	    {
	    if (x > 0)  { tc1 = -atan(-y/x);}
	    if (x < 0)  { tc1 = atan(y/x)-PI;}
	    if (x == 0) { tc1 = 3*PI/2;}
	    }
	  if (y == 0) 
	    {
	    if (x > 0)  { tc1 = 0;}
	    if (x < 0)  { tc1 = PI;}
	    if (x == 0) { tc1 = -1;}
	    }
          
          Local_Direction->InsertNextValue(tc1+PI);
 
     }

  inputPolyData->GetPointData()->AddArray(Local_Direction);

//-----------------------------------------------Geodesic Distance Computation---------------------------------------------------------

  // Geodesic Filter
  vtkSmartPointer<vtkFastMarchingGeodesicDistance> Geodesic =
  vtkSmartPointer<vtkFastMarchingGeodesicDistance>::New();
  Geodesic->SetInputData(inputPolyData);
  Geodesic->SetFieldDataName("FMMDist");
  //Geodesic->SetDistanceStopCriterion(10);
  Geodesic->SetSeeds(seed);
  Geodesic->Update();
  std::cout << "Geodesic Distances Computation Done.. " << std::endl;

   vtkPolyData* outputPolyData = Geodesic->GetOutput();
  std::cout << "Output surface has " << outputPolyData->GetNumberOfPoints() << " points." << std::endl;;

   // Create cutter
  vtkSmartPointer<vtkContourFilter> cutter =
    vtkSmartPointer<vtkContourFilter>::New();
  cutter->SetInputData(outputPolyData);
  cutter->SetValue(0, 1);
  cutter->SetValue(1, 2);
  cutter->SetValue(2, 3);
  cutter->Update();

  vtkSmartPointer<vtkStripper> stripper =
    vtkSmartPointer<vtkStripper>::New();
  stripper->SetInputConnection(cutter->GetOutputPort());
  stripper->Update();

  vtkPolyData* ContoursPolyData = stripper->GetOutput();
  std::cout << "Contours file has " << ContoursPolyData->GetNumberOfLines() << " lines." << std::endl;

  vtkSmartPointer<vtkKochanekSpline> spline =
    vtkSmartPointer<vtkKochanekSpline>::New();
  spline->SetDefaultTension(.5);
 
  // Subdivide Contours
  vtkSmartPointer<vtkSplineFilter> divide =
    vtkSmartPointer<vtkSplineFilter>::New();
  divide->SetInputData(ContoursPolyData);
  divide->SetSubdivideToSpecified();
  divide->SetNumberOfSubdivisions(8);
  divide->SetSpline(spline);
  divide->GetSpline()->ClosedOn();
  divide->Update();

  vtkPolyData* ContoursDividedPolyData = divide->GetOutput();
  std::cout << "Contours Divided file has " << ContoursDividedPolyData->GetNumberOfLines() << " lines." << std::endl;
  
  ContoursDividedPolyData->GetLines()->InitTraversal();
  vtkSmartPointer<vtkIdList> idList3 = vtkSmartPointer<vtkIdList>::New();
  ContoursDividedPolyData->GetLines()->GetNextCell(idList3);
  double S1[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(0),S1);
  double S2[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(1),S2);
  double S3[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(2),S3);
  double S4[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(3),S4);
  double S5[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(4),S5);
  double S6[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(5),S6);
  double S7[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(6),S7);
  double S8[3];
  ContoursDividedPolyData->GetPoint(idList3->GetId(7),S8);

  vtkSmartPointer<vtkIdList> idList2 = vtkSmartPointer<vtkIdList>::New();
  ContoursDividedPolyData->GetLines()->GetNextCell(idList2);
  double SM1[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(0),SM1);
  double SM2[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(1),SM2);
  double SM3[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(2),SM3);
  double SM4[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(3),SM4);
  double SM5[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(4),SM5);
  double SM6[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(5),SM6);
  double SM7[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(6),SM7);
  double SM8[3];
  ContoursDividedPolyData->GetPoint(idList2->GetId(7),SM8);

  vtkSmartPointer<vtkIdList> idList1 = vtkSmartPointer<vtkIdList>::New();
  ContoursDividedPolyData->GetLines()->GetNextCell(idList1);
  double SI1[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(0),SI1);
  double SI2[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(1),SI2);
  double SI3[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(2),SI3);
  double SI4[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(3),SI4);
  double SI5[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(4),SI5);
  double SI6[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(5),SI6);
  double SI7[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(6),SI7);
  double SI8[3];
  ContoursDividedPolyData->GetPoint(idList1->GetId(7),SI8);

 // Build a locator
  vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
  pointLocator->SetDataSet(inputPolyData);
  pointLocator->BuildLocator();
  pointLocator->Update();

  // Create a point set of First Region
  vtkSmartPointer<vtkPoints> points =
    vtkSmartPointer<vtkPoints>::New();
  points->InsertNextPoint(S1);
  points->InsertNextPoint(SM1);
  points->InsertNextPoint(S2);
  points->InsertNextPoint(SM2);
 
  vtkSmartPointer<vtkPolyData> polydata =
    vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);

// Compute the center of mass
  vtkSmartPointer<vtkCenterOfMass> centerOfMassFilter =
    vtkSmartPointer<vtkCenterOfMass>::New();
  centerOfMassFilter->SetInputData(polydata);
  centerOfMassFilter->SetUseScalarsAsWeights(false);
  centerOfMassFilter->Update();
 
  double center[3];
  centerOfMassFilter->GetCenter(center);

  vtkSmartPointer<vtkFloatArray> Region1 = vtkSmartPointer<vtkFloatArray>::New();
  Region1->SetNumberOfComponents(1);
  Region1->SetName("R1");

  for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
     {
       Region1->InsertNextValue(0);
     } 

  vtkIdType R1;
  R1 = pointLocator->FindClosestPoint(S1);
  Region1->SetValue(R1, 1);
  vtkIdType R2;
  R2 = pointLocator->FindClosestPoint(S2);
  Region1->SetValue(R2, 1);
  vtkIdType R3;
  R3 = pointLocator->FindClosestPoint(SM1);
  Region1->SetValue(R3, 1);
  vtkIdType R4;
  R4 = pointLocator->FindClosestPoint(SM2);
  Region1->SetValue(R4, 1);
  vtkIdType R5;
  R5 = pointLocator->FindClosestPoint(center);
  Region1->SetValue(R5, 1);

  inputPolyData->GetPointData()->AddArray(Region1);

  vtkSmartPointer<vtkPolyDataWriter> Contourspolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  Contourspolywriter->SetFileName("Contour.vtk");
  Contourspolywriter->SetInputData(ContoursPolyData);
  Contourspolywriter->Write();

  //Write Surface Results
  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  polywriter->SetFileName("Region1.vtk");
  polywriter->SetInputData(inputPolyData);
  polywriter->Write();


  return EXIT_SUCCESS;
}

