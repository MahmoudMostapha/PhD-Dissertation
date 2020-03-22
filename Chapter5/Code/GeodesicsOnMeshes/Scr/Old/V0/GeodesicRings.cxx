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
  cutter->SetValue(0, 5);
  cutter->SetValue(1, 10);
  cutter->SetValue(2, 15);
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

vtkSmartPointer<vtkFloatArray> Contours_Labels = vtkSmartPointer<vtkFloatArray>::New();
  Contours_Labels->SetNumberOfComponents(1);
  Contours_Labels->SetName("Contours_Labels");

 for(vtkIdType ID = 0; ID < ContoursPolyData->GetNumberOfPoints(); ID++)
     {
       Contours_Labels->InsertNextValue(0);
     } 

  ContoursPolyData->GetLines()->InitTraversal();
  vtkSmartPointer<vtkIdList> ContoursidList = vtkSmartPointer<vtkIdList>::New();
  ContoursPolyData->GetLines()->GetNextCell(ContoursidList);
for(vtkIdType pointId = 0; pointId < ContoursidList->GetNumberOfIds(); pointId++)
{
      double P[3];
      ContoursPolyData->GetPoint(ContoursidList->GetId(pointId),P);
      if (vtkMath::Distance2BetweenPoints(P, S1) + vtkMath::Distance2BetweenPoints(P, S2) < vtkMath::Distance2BetweenPoints(S1, S2))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 1);
	}
      if (vtkMath::Distance2BetweenPoints(P, S2) + vtkMath::Distance2BetweenPoints(P, S3) < vtkMath::Distance2BetweenPoints(S2, S3))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 2);
	}
      if (vtkMath::Distance2BetweenPoints(P, S3) + vtkMath::Distance2BetweenPoints(P, S4) < vtkMath::Distance2BetweenPoints(S3, S4))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 3);
	}
      if (vtkMath::Distance2BetweenPoints(P, S4) + vtkMath::Distance2BetweenPoints(P, S5) < vtkMath::Distance2BetweenPoints(S4, S5))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 4);
	}
      if (vtkMath::Distance2BetweenPoints(P, S5) + vtkMath::Distance2BetweenPoints(P, S6) < vtkMath::Distance2BetweenPoints(S5, S6))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 5);
	}
      if (vtkMath::Distance2BetweenPoints(P, S6) + vtkMath::Distance2BetweenPoints(P, S7) < vtkMath::Distance2BetweenPoints(S6, S7))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 6);
	}
      if (vtkMath::Distance2BetweenPoints(P, S7) + vtkMath::Distance2BetweenPoints(P, S8) < vtkMath::Distance2BetweenPoints(S7, S8))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 7);
        }
      if (vtkMath::Distance2BetweenPoints(P, S8) + vtkMath::Distance2BetweenPoints(P, S1) < vtkMath::Distance2BetweenPoints(S8, S1))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 8);
	}
}    

    ContoursPolyData->GetLines()->GetNextCell(ContoursidList);
for(vtkIdType pointId = 0; pointId < ContoursidList->GetNumberOfIds(); pointId++)
{
      double P[3];
      ContoursPolyData->GetPoint(ContoursidList->GetId(pointId),P);
      if (vtkMath::Distance2BetweenPoints(P, SM1) + vtkMath::Distance2BetweenPoints(P, SM2) < vtkMath::Distance2BetweenPoints(SM1,SM2))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 1);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM2) + vtkMath::Distance2BetweenPoints(P, SM3) < vtkMath::Distance2BetweenPoints(SM2,SM3))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 2);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM3) + vtkMath::Distance2BetweenPoints(P, SM4) < vtkMath::Distance2BetweenPoints(SM3,SM4))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 3);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM4) + vtkMath::Distance2BetweenPoints(P, SM5) < vtkMath::Distance2BetweenPoints(SM4,SM5))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 4);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM5) + vtkMath::Distance2BetweenPoints(P, SM6) < vtkMath::Distance2BetweenPoints(SM5,SM6))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 5);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM6) + vtkMath::Distance2BetweenPoints(P, SM7) < vtkMath::Distance2BetweenPoints(SM6,SM7))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 6);
	}
     if (vtkMath::Distance2BetweenPoints(P, SM7) + vtkMath::Distance2BetweenPoints(P, SM8) < vtkMath::Distance2BetweenPoints(SM7, SM8))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 7);
        }
      if (vtkMath::Distance2BetweenPoints(P, SM8) + vtkMath::Distance2BetweenPoints(P, SM1) < vtkMath::Distance2BetweenPoints(SM8,SM1))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 8);
	}
}    

    ContoursPolyData->GetLines()->GetNextCell(ContoursidList);
for(vtkIdType pointId = 0; pointId < ContoursidList->GetNumberOfIds(); pointId++)
{
      double P[3];
      ContoursPolyData->GetPoint(ContoursidList->GetId(pointId),P);
      if (vtkMath::Distance2BetweenPoints(P, SI1) + vtkMath::Distance2BetweenPoints(P, SI2) < vtkMath::Distance2BetweenPoints(SI1,SI2))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 11);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI2) + vtkMath::Distance2BetweenPoints(P, SI3) < vtkMath::Distance2BetweenPoints(SI2,SI3))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 22);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI3) + vtkMath::Distance2BetweenPoints(P, SI4) < vtkMath::Distance2BetweenPoints(SI3,SI4))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 33);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI4) + vtkMath::Distance2BetweenPoints(P, SI5) < vtkMath::Distance2BetweenPoints(SI4,SI5))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 44);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI5) + vtkMath::Distance2BetweenPoints(P, SI6) < vtkMath::Distance2BetweenPoints(SI5,SI6))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 55);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI6) + vtkMath::Distance2BetweenPoints(P, SI7) < vtkMath::Distance2BetweenPoints(SI6,SI7))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 66);
	}
     if (vtkMath::Distance2BetweenPoints(P, SI7) + vtkMath::Distance2BetweenPoints(P, SI8) < vtkMath::Distance2BetweenPoints(SI7, SI8))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 77);
        }
      if (vtkMath::Distance2BetweenPoints(P, SI8) + vtkMath::Distance2BetweenPoints(P, SI1) < vtkMath::Distance2BetweenPoints(SI8,SI1))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 88);
	}
}

ContoursPolyData->GetPointData()->AddArray(Contours_Labels);

// Write Results
vtkSmartPointer<vtkPolyDataWriter> Contourspolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
Contourspolywriter->SetFileName("Contour.vtk");
Contourspolywriter->SetInputData(ContoursPolyData);
Contourspolywriter->Write();

//-----------------------------------------------Geodesic Rays Computation---------------------------------------------------------

 // Build a locator
  vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
  pointLocator->SetDataSet(inputPolyData);
  pointLocator->BuildLocator();
  pointLocator->Update();

  //***************************************************Outer eight rays*************************************************************

  // Outer 1
  vtkIdType Start1;
  Start1 = pointLocator->FindClosestPoint(S1);
  std::cout << "Starting Path from Vertex " << Start1 << std::endl;
  vtkIdType stop1;
  stop1 = pointLocator->FindClosestPoint(SM1);
  vtkSmartPointer<vtkIdList> Stop1 = vtkSmartPointer<vtkIdList>::New();
  Stop1->InsertNextId(stop1);
  std::cout << "Stopping at Vertex " << Stop1->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath1 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath1->SetBeginPointId(Start1);
  Geodesicpath1->SetSeeds(Stop1);
  Geodesicpath1->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath1-> SetInterpolationOrder(1);
  Geodesicpath1->Update();

  // Outer 2
  vtkIdType Start2;
  Start2 = pointLocator->FindClosestPoint(S2);
  std::cout << "Starting Path from Vertex " << Start2 << std::endl;
  vtkIdType stop2;
  stop2 = pointLocator->FindClosestPoint(SM2);
  vtkSmartPointer<vtkIdList> Stop2 = vtkSmartPointer<vtkIdList>::New();
  Stop2->InsertNextId(stop2);
  std::cout << "Stopping at Vertex " << Stop2->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath2 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath2->SetBeginPointId(Start2);
  Geodesicpath2->SetSeeds(Stop2);
  Geodesicpath2->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath2-> SetInterpolationOrder(1);
  Geodesicpath2->Update();

  // Outer 3
  vtkIdType Start3;
  Start3 = pointLocator->FindClosestPoint(S3);
  std::cout << "Starting Path from Vertex " << Start3 << std::endl;
  vtkIdType stop3;
  stop3 = pointLocator->FindClosestPoint(SM3);
  vtkSmartPointer<vtkIdList> Stop3 = vtkSmartPointer<vtkIdList>::New();
  Stop3->InsertNextId(stop3);
  std::cout << "Stopping at Vertex " << Stop3->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath3 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath3->SetBeginPointId(Start3);
  Geodesicpath3->SetSeeds(Stop3);
  Geodesicpath3->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath3-> SetInterpolationOrder(1);
  Geodesicpath3->Update();

  // Outer 4
  vtkIdType Start4;
  Start4 = pointLocator->FindClosestPoint(S4);
  std::cout << "Starting Path from Vertex " << Start4 << std::endl;
  vtkIdType stop4;
  stop4 = pointLocator->FindClosestPoint(SM4);
  vtkSmartPointer<vtkIdList> Stop4 = vtkSmartPointer<vtkIdList>::New();
  Stop4->InsertNextId(stop4);
  std::cout << "Stopping at Vertex " << Stop4->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath4 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath4->SetBeginPointId(Start4);
  Geodesicpath4->SetSeeds(Stop4);
  Geodesicpath4->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath4-> SetInterpolationOrder(1);
  Geodesicpath4->Update();

  // Outer 5
  vtkIdType Start5;
  Start5 = pointLocator->FindClosestPoint(S5);
  std::cout << "Starting Path from Vertex " << Start5 << std::endl;
  vtkIdType stop5;
  stop5 = pointLocator->FindClosestPoint(SM5);
  vtkSmartPointer<vtkIdList> Stop5 = vtkSmartPointer<vtkIdList>::New();
  Stop5->InsertNextId(stop5);
  std::cout << "Stopping at Vertex " << Stop5->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath5 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath5->SetBeginPointId(Start5);
  Geodesicpath5->SetSeeds(Stop5);
  Geodesicpath5->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath5-> SetInterpolationOrder(1);
  Geodesicpath5->Update();

  // Outer 6
  vtkIdType Start6;
  Start6 = pointLocator->FindClosestPoint(S6);
  std::cout << "Starting Path from Vertex " << Start6 << std::endl;
  vtkIdType stop6;
  stop6 = pointLocator->FindClosestPoint(SM6);
  vtkSmartPointer<vtkIdList> Stop6 = vtkSmartPointer<vtkIdList>::New();
  Stop6->InsertNextId(stop6);
  std::cout << "Stopping at Vertex " << Stop6->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath6 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath6->SetBeginPointId(Start6);
  Geodesicpath6->SetSeeds(Stop6);
  Geodesicpath6->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath6-> SetInterpolationOrder(1);
  Geodesicpath6->Update();

  // Outer 7
  vtkIdType Start7;
  Start7 = pointLocator->FindClosestPoint(S7);
  std::cout << "Starting Path from Vertex " << Start7 << std::endl;
  vtkIdType stop7;
  stop7 = pointLocator->FindClosestPoint(SM7);
  vtkSmartPointer<vtkIdList> Stop7 = vtkSmartPointer<vtkIdList>::New();
  Stop7->InsertNextId(stop7);
  std::cout << "Stopping at Vertex " << Stop7->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath7 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath7->SetBeginPointId(Start7);
  Geodesicpath7->SetSeeds(Stop7);
  Geodesicpath7->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath7-> SetInterpolationOrder(1);
  Geodesicpath7->Update();

  // Outer 8
  vtkIdType Start8;
  Start8 = pointLocator->FindClosestPoint(S8);
  std::cout << "Starting Path from Vertex " << Start8 << std::endl;
  vtkIdType stop8;
  stop8 = pointLocator->FindClosestPoint(SM8);
  vtkSmartPointer<vtkIdList> Stop8 = vtkSmartPointer<vtkIdList>::New();
  Stop8->InsertNextId(stop8);
  std::cout << "Stopping at Vertex " << Stop8->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath8 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath8->SetBeginPointId(Start8);
  Geodesicpath8->SetSeeds(Stop8);
  Geodesicpath8->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath8-> SetInterpolationOrder(1);
  Geodesicpath8->Update();


 //***************************************************Middle Four rays*************************************************************

  // Middle 1
  vtkIdType StartM1;
  StartM1 = pointLocator->FindClosestPoint(SM1);
  std::cout << "Starting Path from Vertex " << StartM1 << std::endl;
  vtkIdType stopM1;
  stopM1 = pointLocator->FindClosestPoint(SI1);
  vtkSmartPointer<vtkIdList> StopM1 = vtkSmartPointer<vtkIdList>::New();
  StopM1->InsertNextId(stopM1);
  std::cout << "Stopping at Vertex " << StopM1->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathM1 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  GeodesicpathM1->SetBeginPointId(StartM1);
  GeodesicpathM1->SetSeeds(StopM1);
  GeodesicpathM1->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathM1-> SetInterpolationOrder(1);
  GeodesicpathM1->Update();

  // Middle 2
  vtkIdType StartM2;
  StartM2 = pointLocator->FindClosestPoint(SM3);
  std::cout << "Starting Path from Vertex " << StartM2 << std::endl;
  vtkIdType stopM2;
  stopM2 = pointLocator->FindClosestPoint(SI3);
  vtkSmartPointer<vtkIdList> StopM2 = vtkSmartPointer<vtkIdList>::New();
  StopM2->InsertNextId(stopM2);
  std::cout << "Stopping at Vertex " << StopM2->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathM2 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  GeodesicpathM2->SetBeginPointId(StartM2);
  GeodesicpathM2->SetSeeds(StopM2);
  GeodesicpathM2->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathM2-> SetInterpolationOrder(1);
  GeodesicpathM2->Update();

  // Middle 3
  vtkIdType StartM3;
  StartM3 = pointLocator->FindClosestPoint(SM5);
  std::cout << "Starting Path from Vertex " << StartM3 << std::endl;
  vtkIdType stopM3;
  stopM3 = pointLocator->FindClosestPoint(SI5);
  vtkSmartPointer<vtkIdList> StopM3 = vtkSmartPointer<vtkIdList>::New();
  StopM3->InsertNextId(stopM3);
  std::cout << "Stopping at Vertex " << StopM3->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathM3 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  GeodesicpathM3->SetBeginPointId(StartM3);
  GeodesicpathM3->SetSeeds(StopM3);
  GeodesicpathM3->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathM3-> SetInterpolationOrder(1);
  GeodesicpathM3->Update();

  // Middle 4
  vtkIdType StartM4;
  StartM4 = pointLocator->FindClosestPoint(SM7);
  std::cout << "Starting Path from Vertex " << StartM4 << std::endl;
  vtkIdType stopM4;
  stopM4 = pointLocator->FindClosestPoint(SI7);
  vtkSmartPointer<vtkIdList> StopM4 = vtkSmartPointer<vtkIdList>::New();
  StopM4->InsertNextId(stopM4);
  std::cout << "Stopping at Vertex " << StopM4->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathM4 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  GeodesicpathM4->SetBeginPointId(StartM4);
  GeodesicpathM4->SetSeeds(StopM4);
  GeodesicpathM4->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathM4-> SetInterpolationOrder(1);
  GeodesicpathM4->Update();

 //***************************************************Inner Two rays*************************************************************

  // Inner 1;
  vtkIdType StartI1;
  StartI1 = pointLocator->FindClosestPoint(SI1);
  std::cout << "Starting Path from Vertex " << StartI1 << std::endl;
  std::cout << "Stopping at Vertex " << seed->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathI1 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  GeodesicpathI1->SetBeginPointId(StartI1);
  GeodesicpathI1->SetSeeds(seed);
  GeodesicpathI1->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathI1-> SetInterpolationOrder(1);
  GeodesicpathI1->Update();

  // Inner 2
  vtkIdType StartI2;
  StartI2 = pointLocator->FindClosestPoint(SI5);
  std::cout << "Starting Path from Vertex " << StartI2 << std::endl;
  std::cout << "Stopping at Vertex " << seed->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathI2 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  GeodesicpathI2->SetBeginPointId(StartI2);
  GeodesicpathI2->SetSeeds(seed);
  GeodesicpathI2->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathI2-> SetInterpolationOrder(1);
  GeodesicpathI2->Update();

  //Append the meshes 
  vtkSmartPointer<vtkAppendPolyData> appendFilter =
    vtkSmartPointer<vtkAppendPolyData>::New();
  appendFilter->AddInputData(Geodesicpath1->GetOutput());
  appendFilter->AddInputData(Geodesicpath2->GetOutput());
  appendFilter->AddInputData(Geodesicpath3->GetOutput());
  appendFilter->AddInputData(Geodesicpath4->GetOutput());
  appendFilter->AddInputData(Geodesicpath5->GetOutput());
  appendFilter->AddInputData(Geodesicpath6->GetOutput());
  appendFilter->AddInputData(Geodesicpath7->GetOutput());
  appendFilter->AddInputData(Geodesicpath8->GetOutput());
  appendFilter->AddInputData(GeodesicpathM1->GetOutput());
  appendFilter->AddInputData(GeodesicpathM2->GetOutput());
  appendFilter->AddInputData(GeodesicpathM3->GetOutput());
  appendFilter->AddInputData(GeodesicpathM4->GetOutput());
  appendFilter->AddInputData(GeodesicpathI1->GetOutput());
  appendFilter->AddInputData(GeodesicpathI2->GetOutput());
  appendFilter->Update();
  vtkPolyData* Grid_Rays = appendFilter->GetOutput();
  std::cout << "Final Grid Rays file has " << Grid_Rays->GetNumberOfLines() << " lines." << std::endl;

  vtkSmartPointer<vtkFloatArray> Grid_Rays_Labels = vtkSmartPointer<vtkFloatArray>::New();
  Grid_Rays_Labels->SetNumberOfComponents(1);
  Grid_Rays_Labels->SetName("Grid_Rays_Labels");

for(vtkIdType ID = 0; ID < Grid_Rays->GetNumberOfPoints(); ID++)
     {
       Grid_Rays_Labels->InsertNextValue(0);
     } 

  Grid_Rays->GetLines()->InitTraversal();
  vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
  float Line_ID = 1;
  while(Grid_Rays->GetLines()->GetNextCell(idList))
    {
    for(vtkIdType pointId = 0; pointId < idList->GetNumberOfIds(); pointId++)
      {
      Grid_Rays_Labels->SetValue(idList->GetId(pointId), Line_ID);
      }    
      Line_ID+= 1;
    }
  Grid_Rays->GetPointData()->AddArray(Grid_Rays_Labels);

/*
  //Append the meshes 
  vtkSmartPointer<vtkAppendPolyData> appendFilter =
    vtkSmartPointer<vtkAppendPolyData>::New();
  appendFilter->AddInputData(ContoursPolyData);
  appendFilter->AddInputData(Geodesicpath1->GetOutput());
  appendFilter->AddInputData(Geodesicpath2->GetOutput());
  appendFilter->AddInputData(Geodesicpath3->GetOutput());
  appendFilter->AddInputData(Geodesicpath4->GetOutput());
  appendFilter->AddInputData(Geodesicpath5->GetOutput());
  appendFilter->AddInputData(Geodesicpath6->GetOutput());
  appendFilter->AddInputData(Geodesicpath7->GetOutput());
  appendFilter->AddInputData(Geodesicpath8->GetOutput());
  appendFilter->AddInputData(GeodesicpathM1->GetOutput());
  appendFilter->AddInputData(GeodesicpathM2->GetOutput());
  appendFilter->AddInputData(GeodesicpathM3->GetOutput());
  appendFilter->AddInputData(GeodesicpathM4->GetOutput());
  appendFilter->AddInputData(GeodesicpathI1->GetOutput());
  appendFilter->AddInputData(GeodesicpathI2->GetOutput());
  appendFilter->Update();
  vtkPolyData* Grid = appendFilter->GetOutput();
  std::cout << "Final Grid file has " << Grid->GetNumberOfLines() << " lines." << std::endl;

  vtkSmartPointer<vtkFloatArray> Grid_Labels = vtkSmartPointer<vtkFloatArray>::New();
  Grid_Labels->SetNumberOfComponents(1);
  Grid_Labels->SetName("Grid_Labels");

 for(vtkIdType ID = 0; ID < Grid->GetNumberOfPoints(); ID++)
     {
       Grid_Labels->InsertNextValue(0);
     } 

  Grid->GetLines()->InitTraversal();
  vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
  float Line_ID = 1;
  while(Grid->GetLines()->GetNextCell(idList))
    {
    for(vtkIdType pointId = 0; pointId < idList->GetNumberOfIds(); pointId++)
      {
      Grid_Labels->SetValue(idList->GetId(pointId), Line_ID);
      }    
      Line_ID+= 1;
    }
  Grid->GetPointData()->AddArray(Grid_Labels);

*/

//-----------------------------------------------Check Enclosed Points---------------------------------------------------------

   vtkSmartPointer<vtkPoints> RegionPoints =
    vtkSmartPointer<vtkPoints>::New();

    for(vtkIdType ID = 0; ID < Grid_Rays->GetNumberOfPoints(); ID++)
     {
       if (Grid_Rays_Labels->GetValue(ID) == 1 || Grid_Rays_Labels->GetValue(ID) == 2)
       {
          double S[3];
          Grid_Rays->GetPoint(ID,S);
          RegionPoints->InsertNextPoint (S);
       }
     } 


 for(vtkIdType ID = 0; ID < ContoursPolyData->GetNumberOfPoints(); ID++)
     {
       if (Contours_Labels->GetValue(ID) == 2)
       {
          double S2[3];
          ContoursPolyData->GetPoint(ID,S2);
          RegionPoints->InsertNextPoint (S2);
       }
       Contours_Labels->InsertNextValue(0);
     } 

  //Points inside test
  vtkSmartPointer<vtkSelectPolyData> loop = 
  vtkSmartPointer<vtkSelectPolyData>::New();
  loop->SetInputData(outputPolyData);
  loop->SetLoop(RegionPoints);
  loop->GenerateSelectionScalarsOn();
  loop->SetSelectionModeToSmallestRegion();
  loop->Update();


  //Write Surface Results
  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  polywriter->SetFileName(argv[1]);
  polywriter->SetInputData(loop->GetOutput());
  polywriter->Write();

  // Write Results
  vtkSmartPointer<vtkPolyDataWriter> Rayspolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  Rayspolywriter->SetFileName("Grid_Rays.vtk");
  Rayspolywriter->SetInputData(Grid_Rays);
  Rayspolywriter->Write();

  return EXIT_SUCCESS;
}

