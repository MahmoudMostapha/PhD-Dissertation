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


#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>



int main(int argc, char* argv[])
{
  if (argc < 5)
    {
    std::cerr << "Usage: " << argv[0] << "SurfaceMesh.vtk StartVertex EndVertex Step" << std::endl;
    return EXIT_FAILURE;
    }


  int start_s=clock();
  // Get all surface data from the file
  vtkSmartPointer<vtkPolyDataReader> surfacereader =
  vtkSmartPointer<vtkPolyDataReader>::New();
  surfacereader->SetFileName(argv[1]);
  surfacereader->Update();
  vtkPolyData* inputPolyData = surfacereader->GetOutput();
  std::cout << "Input surface has " << inputPolyData->GetNumberOfPoints() << " points." << std::endl;

  ofstream Result;
  Result.open ("Result.csv");
  vtkIdType StartVertex = atoi(argv[2]);
  vtkIdType EndVertex = atoi(argv[3])+1;

for(vtkIdType Vertex = StartVertex; Vertex < EndVertex; Vertex++)
{

  // Add the Seed
  vtkSmartPointer<vtkIdList> seed =
      vtkSmartPointer<vtkIdList>::New();
  seed->InsertNextId(Vertex);
  std::cout << "Starting from Vertex " << seed->GetId(0) << std::endl;

//-----------------------------------------------Estimate Local Orientation---------------------------------------------------------
/*
  vtkSmartPointer<vtkFloatArray> Local_Direction = vtkSmartPointer<vtkFloatArray>::New();
  Local_Direction->SetNumberOfComponents(1);
  Local_Direction->SetName("Local_Direction");

  vtkSmartPointer<vtkFloatArray> Phi = vtkFloatArray::SafeDownCast(inputPolyData->GetPointData()->GetArray("Color_Map_Phi"));
  vtkSmartPointer<vtkFloatArray> Theta = vtkFloatArray::SafeDownCast(inputPolyData->GetPointData()->GetArray("Color_Map_Theta"));
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
*/
//-----------------------------------------------Geodesic Contours Extraction--------------------------------------------------------

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
  std::cout << "Output surface has " << outputPolyData->GetNumberOfPoints() << " points." << std::endl;

   // Create cutter
  vtkSmartPointer<vtkContourFilter> cutter =
    vtkSmartPointer<vtkContourFilter>::New();
  cutter->SetInputData(outputPolyData);
  cutter->SetValue(0, atof(argv[4]));
  cutter->SetValue(1, 2*atof(argv[4]));
  cutter->SetValue(2, 3*atof(argv[4]));
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

  vtkSmartPointer<vtkFloatArray> Level = vtkFloatArray::SafeDownCast(ContoursDividedPolyData->GetPointData()->GetArray("FMMDist"));
  ContoursDividedPolyData->GetLines()->InitTraversal();
  vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();

  double S1[3];
  double S2[3];
  double S3[3];
  double S4[3];
  double S5[3];
  double S6[3];
  double S7[3];
  double S8[3];
  double SM1[3];
  double SM2[3];
  double SM3[3];
  double SM4[3];
  double SM5[3];
  double SM6[3];
  double SM7[3];
  double SM8[3];
  double SI1[3];
  double SI2[3];
  double SI3[3];
  double SI4[3];
  double SI5[3];
  double SI6[3];
  double SI7[3];
  double SI8[3];

  vtkSmartPointer<vtkFloatArray> ContoursDivided_Labels = vtkSmartPointer<vtkFloatArray>::New();
  ContoursDivided_Labels->SetNumberOfComponents(1);
  ContoursDivided_Labels->SetName("ContoursDivided_Labels");

 for(vtkIdType ID = 0; ID < ContoursDividedPolyData->GetNumberOfPoints(); ID++)
     {
       ContoursDivided_Labels->InsertNextValue(0);
     } 

 for(int Line_ID = 0;  Line_ID < 3; Line_ID++)
     {

	      ContoursDividedPolyData->GetLines()->GetNextCell(idList);

	      if ( abs( Level->GetValue(idList->GetId(0)) - 3*atof(argv[4]) ) < 0.00001 )
      		 {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;
      		   ContoursDividedPolyData->GetPoint(idList->GetId(0),S1);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(1),S2);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(2),S3);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(3),S4);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(4),S5);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(5),S6);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(6),S7);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(7),S8);

      		   ContoursDivided_Labels->SetValue(idList->GetId(0), 1);
      		   ContoursDivided_Labels->SetValue(idList->GetId(1), 2);
      		   ContoursDivided_Labels->SetValue(idList->GetId(2), 3);
      		   ContoursDivided_Labels->SetValue(idList->GetId(3), 4);
      		   ContoursDivided_Labels->SetValue(idList->GetId(4), 5);
      		   ContoursDivided_Labels->SetValue(idList->GetId(5), 6);
      		   ContoursDivided_Labels->SetValue(idList->GetId(6), 7);
      		   ContoursDivided_Labels->SetValue(idList->GetId(7), 8);
      		 }

	      if ( abs( Level->GetValue(idList->GetId(0)) - 2*atof(argv[4]) ) < 0.00001 )
    		  {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;
      		   ContoursDividedPolyData->GetPoint(idList->GetId(0),SM1);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(1),SM2);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(2),SM3);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(3),SM4);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(4),SM5);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(5),SM6);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(6),SM7);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(7),SM8);

      		   ContoursDivided_Labels->SetValue(idList->GetId(0), 1);
      		   ContoursDivided_Labels->SetValue(idList->GetId(1), 2);
      		   ContoursDivided_Labels->SetValue(idList->GetId(2), 3);
      		   ContoursDivided_Labels->SetValue(idList->GetId(3), 4);
      		   ContoursDivided_Labels->SetValue(idList->GetId(4), 5);
      		   ContoursDivided_Labels->SetValue(idList->GetId(5), 6);
      		   ContoursDivided_Labels->SetValue(idList->GetId(6), 7);
      		   ContoursDivided_Labels->SetValue(idList->GetId(7), 8);
    		  }

	      if ( abs( Level->GetValue(idList->GetId(0)) - atof(argv[4]) ) < 0.00001 )
    		  {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;
      		   ContoursDividedPolyData->GetPoint(idList->GetId(0),SI1);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(1),SI2);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(2),SI3);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(3),SI4);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(4),SI5);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(5),SI6);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(6),SI7);
      		   ContoursDividedPolyData->GetPoint(idList->GetId(7),SI8);

      		   ContoursDivided_Labels->SetValue(idList->GetId(0), 1);
      		   ContoursDivided_Labels->SetValue(idList->GetId(1), 2);
      		   ContoursDivided_Labels->SetValue(idList->GetId(2), 3);
      		   ContoursDivided_Labels->SetValue(idList->GetId(3), 4);
      		   ContoursDivided_Labels->SetValue(idList->GetId(4), 5);
      		   ContoursDivided_Labels->SetValue(idList->GetId(5), 6);
      		   ContoursDivided_Labels->SetValue(idList->GetId(6), 7);
      		   ContoursDivided_Labels->SetValue(idList->GetId(7), 8);
    		  }
        }

  ContoursDividedPolyData->GetPointData()->AddArray(ContoursDivided_Labels);

  // Write Results
  vtkSmartPointer<vtkPolyDataWriter> ContoursDividedPolyDatapolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  ContoursDividedPolyDatapolywriter->SetFileName("ContoursDividedPolyData.vtk");
  ContoursDividedPolyDatapolywriter->SetInputData(ContoursDividedPolyData);
  ContoursDividedPolyDatapolywriter->Write();


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
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 9);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM2) + vtkMath::Distance2BetweenPoints(P, SM3) < vtkMath::Distance2BetweenPoints(SM2,SM3))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 10);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM3) + vtkMath::Distance2BetweenPoints(P, SM4) < vtkMath::Distance2BetweenPoints(SM3,SM4))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 11);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM4) + vtkMath::Distance2BetweenPoints(P, SM5) < vtkMath::Distance2BetweenPoints(SM4,SM5))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 12);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM5) + vtkMath::Distance2BetweenPoints(P, SM6) < vtkMath::Distance2BetweenPoints(SM5,SM6))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 13);
	}
      if (vtkMath::Distance2BetweenPoints(P, SM6) + vtkMath::Distance2BetweenPoints(P, SM7) < vtkMath::Distance2BetweenPoints(SM6,SM7))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 14);
	}
     if (vtkMath::Distance2BetweenPoints(P, SM7) + vtkMath::Distance2BetweenPoints(P, SM8) < vtkMath::Distance2BetweenPoints(SM7, SM8))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 15);
        }
      if (vtkMath::Distance2BetweenPoints(P, SM8) + vtkMath::Distance2BetweenPoints(P, SM1) < vtkMath::Distance2BetweenPoints(SM8,SM1))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 16);
	}
}    

    ContoursPolyData->GetLines()->GetNextCell(ContoursidList);
for(vtkIdType pointId = 0; pointId < ContoursidList->GetNumberOfIds(); pointId++)
{
      double P[3];
      ContoursPolyData->GetPoint(ContoursidList->GetId(pointId),P);
      if (vtkMath::Distance2BetweenPoints(P, SI1) + vtkMath::Distance2BetweenPoints(P, SI2) < vtkMath::Distance2BetweenPoints(SI1,SI2))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 17);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI2) + vtkMath::Distance2BetweenPoints(P, SI3) < vtkMath::Distance2BetweenPoints(SI2,SI3))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 18);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI3) + vtkMath::Distance2BetweenPoints(P, SI4) < vtkMath::Distance2BetweenPoints(SI3,SI4))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 19);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI4) + vtkMath::Distance2BetweenPoints(P, SI5) < vtkMath::Distance2BetweenPoints(SI4,SI5))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 20);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI5) + vtkMath::Distance2BetweenPoints(P, SI6) < vtkMath::Distance2BetweenPoints(SI5,SI6))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 21);
	}
      if (vtkMath::Distance2BetweenPoints(P, SI6) + vtkMath::Distance2BetweenPoints(P, SI7) < vtkMath::Distance2BetweenPoints(SI6,SI7))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 22);
	}
     if (vtkMath::Distance2BetweenPoints(P, SI7) + vtkMath::Distance2BetweenPoints(P, SI8) < vtkMath::Distance2BetweenPoints(SI7, SI8))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 23);
        }
      if (vtkMath::Distance2BetweenPoints(P, SI8) + vtkMath::Distance2BetweenPoints(P, SI1) < vtkMath::Distance2BetweenPoints(SI8,SI1))
	{
         Contours_Labels->SetValue(ContoursidList->GetId(pointId), 24);
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
  
   std::cout << "Outer Paths Calculation. " << std::endl;

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
  
  std::cout << "Middle Paths Calculation. " << std::endl;


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
  
  std::cout << "Inner Paths Calculation. " << std::endl;

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

  vtkSmartPointer<vtkFloatArray> Rays_Labels = vtkSmartPointer<vtkFloatArray>::New();
  Rays_Labels->SetNumberOfComponents(1);
  Rays_Labels->SetName("Rays_Labels");

for(vtkIdType ID = 0; ID < Grid_Rays->GetNumberOfPoints(); ID++)
     {
       Rays_Labels->InsertNextValue(0);
     } 

  Grid_Rays->GetLines()->InitTraversal();
  float Line_ID = 1;
  while(Grid_Rays->GetLines()->GetNextCell(idList))
    {
    for(vtkIdType pointId = 0; pointId < idList->GetNumberOfIds(); pointId++)
      {
      Rays_Labels->SetValue(idList->GetId(pointId), Line_ID);
      }    
      Line_ID+= 1;
    }
  Grid_Rays->GetPointData()->AddArray(Rays_Labels);

  // Write Results
  vtkSmartPointer<vtkPolyDataWriter> Rayspolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  Rayspolywriter->SetFileName("Grid_Rays.vtk");
  Rayspolywriter->SetInputData(Grid_Rays);
  Rayspolywriter->Write();


  //Append the meshes 
  vtkSmartPointer<vtkAppendPolyData> appendFilter1 =
    vtkSmartPointer<vtkAppendPolyData>::New();
  appendFilter1->AddInputData(ContoursPolyData);
  appendFilter1->AddInputData(Geodesicpath1->GetOutput());
  appendFilter1->AddInputData(Geodesicpath2->GetOutput());
  appendFilter1->AddInputData(Geodesicpath3->GetOutput());
  appendFilter1->AddInputData(Geodesicpath4->GetOutput());
  appendFilter1->AddInputData(Geodesicpath5->GetOutput());
  appendFilter1->AddInputData(Geodesicpath6->GetOutput());
  appendFilter1->AddInputData(Geodesicpath7->GetOutput());
  appendFilter1->AddInputData(Geodesicpath8->GetOutput());
  appendFilter1->AddInputData(GeodesicpathM1->GetOutput());
  appendFilter1->AddInputData(GeodesicpathM2->GetOutput());
  appendFilter1->AddInputData(GeodesicpathM3->GetOutput());
  appendFilter1->AddInputData(GeodesicpathM4->GetOutput());
  appendFilter1->AddInputData(GeodesicpathI1->GetOutput());
  appendFilter1->AddInputData(GeodesicpathI2->GetOutput());
  appendFilter1->Update();
  vtkPolyData* Grid = appendFilter1->GetOutput();
  std::cout << "Final Grid file has " << Grid->GetNumberOfLines() << " lines." << std::endl;

  // Write Results
  vtkSmartPointer<vtkPolyDataWriter> Gridpolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  Gridpolywriter->SetFileName("Grid.vtk");
  Gridpolywriter->SetInputData(Grid);
  Gridpolywriter->Write();

//-----------------------------------------------Extract Regions Vertices---------------------------------------------------------


//***************************************************Outer 8 Regions*************************************************************
  
 for(int Region_ID = 1; Region_ID < 9; Region_ID++)
     {
        std::cout << "Current Region ID " << Region_ID << std::endl;
   
	    vtkSmartPointer<vtkPoints> RegionPoints =
	    vtkSmartPointer<vtkPoints>::New();
	    for(vtkIdType ID = 0; ID < Grid_Rays->GetNumberOfPoints(); ID++)
	     {
	       if (Rays_Labels->GetValue(ID) == Region_ID || Rays_Labels->GetValue(ID) == (Region_ID+1 % 8))
	       {
		  double S[3];
		  Grid_Rays->GetPoint(ID,S);
		  RegionPoints->InsertNextPoint(S);
	       }
	     } 

	 for(vtkIdType ID = 0; ID < ContoursPolyData->GetNumberOfPoints(); ID++)
	     { 
	       if (Contours_Labels->GetValue(ID) == Region_ID || Contours_Labels->GetValue(ID) == Region_ID + 8 )
	       {
		  double S[3];
		  ContoursPolyData->GetPoint(ID,S);
		  RegionPoints->InsertNextPoint(S);
	       }
	     } 

	  vtkSmartPointer<vtkPolyData> polydata =
	    vtkSmartPointer<vtkPolyData>::New();
	  polydata->SetPoints(RegionPoints);
	  
         // Compute the center of mass
	  vtkSmartPointer<vtkCenterOfMass> centerOfMassFilter =
	    vtkSmartPointer<vtkCenterOfMass>::New();
	  centerOfMassFilter->SetInputData(polydata);
	  centerOfMassFilter->SetUseScalarsAsWeights(false);
	  centerOfMassFilter->Update();
	  double center[3];
	  centerOfMassFilter->GetCenter(center);

		  vtkIdType R1;
		  if (Region_ID == 1)
		  {
		  R1 = pointLocator->FindClosestPoint(S1);
		  }
		  else if (Region_ID == 2)
		  {
		  R1 = pointLocator->FindClosestPoint(S2);
		  }
		  else if (Region_ID == 3)
		  {
		  R1 = pointLocator->FindClosestPoint(S3);
		  }
		  else if (Region_ID == 4)
		  {
		  R1 = pointLocator->FindClosestPoint(S4);
		  }
		  else if (Region_ID == 5)
		  {
		  R1 = pointLocator->FindClosestPoint(S5);
		  }
		  else if (Region_ID == 6)
		  {
		  R1 = pointLocator->FindClosestPoint(S6);
		  }
		  else if (Region_ID == 7)
		  {
		  R1 = pointLocator->FindClosestPoint(S7);
		  }
		  else if (Region_ID == 8)
		  {
		  R1 = pointLocator->FindClosestPoint(S8);
		  }

		  vtkIdType R2;
		  if (Region_ID == 1)
		  {
		  R2 = pointLocator->FindClosestPoint(S2);
		  }
		  else if (Region_ID == 2)
		  {
		  R2 = pointLocator->FindClosestPoint(S3);
		  }
		  else if (Region_ID == 3)
		  {
		  R2 = pointLocator->FindClosestPoint(S4);
		  }
		  else if (Region_ID == 4)
		  {
		  R2 = pointLocator->FindClosestPoint(S5);
		  }
		  else if (Region_ID == 5)
		  {
		  R2 = pointLocator->FindClosestPoint(S6);
		  }
		  else if (Region_ID == 6)
		  {
		  R2 = pointLocator->FindClosestPoint(S7);
		  }
		  else if (Region_ID == 7)
		  {
		  R2 = pointLocator->FindClosestPoint(S8);
		  }
		  else if (Region_ID == 8)
		  {
		  R2 = pointLocator->FindClosestPoint(S1);
		  }

		  vtkIdType R3;
		  if (Region_ID == 1)
		  {
		  R3 = pointLocator->FindClosestPoint(SM1);
		  }
		  else if (Region_ID == 2)
		  {
		  R3 = pointLocator->FindClosestPoint(SM2);
		  }
		  else if (Region_ID == 3)
		  {
		  R3 = pointLocator->FindClosestPoint(SM3);
		  }
		  else if (Region_ID == 4)
		  {
		  R3 = pointLocator->FindClosestPoint(SM4);
		  }
		  else if (Region_ID == 5)
		  {
		  R3 = pointLocator->FindClosestPoint(SM5);
		  }
		  else if (Region_ID == 6)
		  {
		  R3 = pointLocator->FindClosestPoint(SM6);
		  }
		  else if (Region_ID == 7)
		  {
		  R3 = pointLocator->FindClosestPoint(SM7);
		  }
		  else if (Region_ID == 8)
		  {
		  R3 = pointLocator->FindClosestPoint(SM8);
		  }
		  vtkIdType R4;
		  if (Region_ID == 1)
		  {
		  R4 = pointLocator->FindClosestPoint(SM2);
		  }
		  else if (Region_ID == 2)
		  {
		  R4 = pointLocator->FindClosestPoint(SM3);
		  }
		  else if (Region_ID == 3)
		  {
		  R4 = pointLocator->FindClosestPoint(SM4);
		  }
		  else if (Region_ID == 4)
		  {
		  R4 = pointLocator->FindClosestPoint(SM5);
		  }
		  else if (Region_ID == 5)
		  {
		  R4 = pointLocator->FindClosestPoint(SM6);
		  }
		  else if (Region_ID == 6)
		  {
		  R4 = pointLocator->FindClosestPoint(SM7);
		  }
		  else if (Region_ID == 7)
		  {
		  R4 = pointLocator->FindClosestPoint(SM8);
		  }
		  else if (Region_ID == 8)
		  {
		  R4 = pointLocator->FindClosestPoint(SM1);
		  }
		  vtkIdType R5;
		  R5 = pointLocator->FindClosestPoint(center);
		  Result << seed->GetId(0) << "," << R1 << "," << R2 << "," << R3 << "," << R4 << ","<< R5 <<  endl;
	}


//***************************************************Middle 4 Regions*************************************************************

 for(int Region_ID = 1; Region_ID < 5; Region_ID++)
     {
        std::cout << "Current Region ID " << Region_ID << std::endl;
   
	    vtkSmartPointer<vtkPoints> RegionPoints =
	    vtkSmartPointer<vtkPoints>::New();
	    for(vtkIdType ID = 0; ID < Grid_Rays->GetNumberOfPoints(); ID++)
	     {
               int current = 8 + Region_ID;
               int next = 9 + Region_ID;
               if (next == 13)
               {
                   next = 9;
               }

	       if (Rays_Labels->GetValue(ID) == current || Rays_Labels->GetValue(ID) == next)
	       {
		  double S[3];
		  Grid_Rays->GetPoint(ID,S);
		  RegionPoints->InsertNextPoint(S);
	       }

	     } 

         
	 for(vtkIdType ID = 0; ID < ContoursPolyData->GetNumberOfPoints(); ID++)
	     { 

               
	       if (Contours_Labels->GetValue(ID) == 9 + (Region_ID-1)*2 || Contours_Labels->GetValue(ID) == 9 + (Region_ID-1)*2 + 1 || Contours_Labels->GetValue(ID) == 17 + (Region_ID-1)*2 || Contours_Labels->GetValue(ID) == 17 + (Region_ID-1)*2 + 1 )
	       {
		  double S[3];
		  ContoursPolyData->GetPoint(ID,S);
		  RegionPoints->InsertNextPoint(S);
	       }


	     } 

	  vtkSmartPointer<vtkPolyData> polydata =
	    vtkSmartPointer<vtkPolyData>::New();
	  polydata->SetPoints(RegionPoints);
	  
         // Compute the center of mass
	  vtkSmartPointer<vtkCenterOfMass> centerOfMassFilter =
	    vtkSmartPointer<vtkCenterOfMass>::New();
	  centerOfMassFilter->SetInputData(polydata);
	  centerOfMassFilter->SetUseScalarsAsWeights(false);
	  centerOfMassFilter->Update();
	  double center[3];
	  centerOfMassFilter->GetCenter(center);

		  vtkIdType R1;
		  if (Region_ID == 1)
		  {
		  R1 = pointLocator->FindClosestPoint(SM1);
		  }
		  else if (Region_ID == 2)
		  {
		  R1 = pointLocator->FindClosestPoint(SM3);
		  }
		  else if (Region_ID == 3)
		  {
		  R1 = pointLocator->FindClosestPoint(SM5);
		  }
		  else if (Region_ID == 4)
		  {
		  R1 = pointLocator->FindClosestPoint(SM7);
		  }
	
		  vtkIdType R2;	  
	          if (Region_ID == 1)
		  {
		  R2 = pointLocator->FindClosestPoint(SM3);
		  }
		  else if (Region_ID == 2)
		  {
		  R2 = pointLocator->FindClosestPoint(SM5);
		  }
		  else if (Region_ID == 3)
		  {
		  R2 = pointLocator->FindClosestPoint(SM7);
		  }
		  else if (Region_ID == 4)
		  {
		  R2 = pointLocator->FindClosestPoint(SM1);
		  }

		  vtkIdType R3;
		  if (Region_ID == 1)
		  {
		  R3 = pointLocator->FindClosestPoint(SI1);
		  }
		  else if (Region_ID == 2)
		  {
		  R3 = pointLocator->FindClosestPoint(SI3);
		  }
		  else if (Region_ID == 3)
		  {
		  R3 = pointLocator->FindClosestPoint(SI5);
		  }
		  else if (Region_ID == 4)
		  {
		  R3 = pointLocator->FindClosestPoint(SI7);
		  }
	
		  vtkIdType R4;	  
	          if (Region_ID == 1)
		  {
		  R4 = pointLocator->FindClosestPoint(SI3);
		  }
		  else if (Region_ID == 2)
		  {
		  R4 = pointLocator->FindClosestPoint(SI5);
		  }
		  else if (Region_ID == 3)
		  {
		  R4 = pointLocator->FindClosestPoint(SI7);
		  }
		  else if (Region_ID == 4)
		  {
		  R4 = pointLocator->FindClosestPoint(SI1);
                  }

		  vtkIdType R5;
		  R5 = pointLocator->FindClosestPoint(center);
		  Result << seed->GetId(0) << "," << R1 << "," << R2 << "," << R3 << "," << R4 << ","<< R5 <<  endl;
	}


//***************************************************Inner 2 Regions*************************************************************

 for(int Region_ID = 1; Region_ID < 3; Region_ID++)
     {
        std::cout << "Current Region ID " << Region_ID << std::endl;
   
	    vtkSmartPointer<vtkPoints> RegionPoints =
	    vtkSmartPointer<vtkPoints>::New();
	    for(vtkIdType ID = 0; ID < Grid_Rays->GetNumberOfPoints(); ID++)
	     {
	       if (Rays_Labels->GetValue(ID) == 13 || Rays_Labels->GetValue(ID) == 14)
	       {
		  double S[3];
		  Grid_Rays->GetPoint(ID,S);
		  RegionPoints->InsertNextPoint(S);
	       }
	     } 

         
	 for(vtkIdType ID = 0; ID < ContoursPolyData->GetNumberOfPoints(); ID++)
	     { 

              if (Region_ID == 1)
              { 
		      if (Contours_Labels->GetValue(ID) == 17 || Contours_Labels->GetValue(ID) == 18 || Contours_Labels->GetValue(ID) == 19 || Contours_Labels->GetValue(ID) == 20)
		       {
			  double S[3];
			  ContoursPolyData->GetPoint(ID,S);
			  RegionPoints->InsertNextPoint(S);
		       }
              }

              if (Region_ID == 2)
              { 
		      if (Contours_Labels->GetValue(ID) == 21 || Contours_Labels->GetValue(ID) == 22 || Contours_Labels->GetValue(ID) == 23 || Contours_Labels->GetValue(ID) == 24)
		       {
			  double S[3];
			  ContoursPolyData->GetPoint(ID,S);
			  RegionPoints->InsertNextPoint(S);
		       }
              }

	     } 

	  vtkSmartPointer<vtkPolyData> polydata =
	    vtkSmartPointer<vtkPolyData>::New();
	  polydata->SetPoints(RegionPoints);
	  
         // Compute the center of mass
	  vtkSmartPointer<vtkCenterOfMass> centerOfMassFilter =
	    vtkSmartPointer<vtkCenterOfMass>::New();
	  centerOfMassFilter->SetInputData(polydata);
	  centerOfMassFilter->SetUseScalarsAsWeights(false);
	  centerOfMassFilter->Update();
	  double center[3];
	  centerOfMassFilter->GetCenter(center);
		  vtkIdType R1;
		  if (Region_ID == 1)
		  {
		  R1 = pointLocator->FindClosestPoint(SI1);
		  }
		  else if (Region_ID == 2)
		  {
		  R1 = pointLocator->FindClosestPoint(SI5);
		  }
	
		  vtkIdType R2;	  
	          if (Region_ID == 1)
		  {
		  R2 = pointLocator->FindClosestPoint(SI3);
		  }
		  else if (Region_ID == 2)
		  {
		  R2 = pointLocator->FindClosestPoint(SI7);
		  }

		  vtkIdType R3;
		  if (Region_ID == 1)
		  {
		  R3 = pointLocator->FindClosestPoint(SI5);
		  }
		  else if (Region_ID == 2)
		  {
		  R3 = pointLocator->FindClosestPoint(SI1);
		  }
	
		  vtkIdType R4;	  
	          if (Region_ID == 1)
		  {
		  R4 = seed->GetId(0);
		  }
		  else if (Region_ID == 2)
		  {
		  R4 = seed->GetId(0);
		  }

		  vtkIdType R5;
		  R5 = pointLocator->FindClosestPoint(center);
		  Result << seed->GetId(0) << "," << R1 << "," << R2 << "," << R3 << "," << R4 << ","<< R5 <<  endl;
	}

}

  Result.close();

  //Write Surface Results
  //vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  //polywriter->SetFileName("Output.vtk");
  //polywriter->SetInputData(inputPolyData);
  //polywriter->Write();

    int stop_s=clock();
    cout << "time: " << (((float)(stop_s-start_s))/CLOCKS_PER_SEC)/60 <<" min" << endl;
  return EXIT_SUCCESS;
}

