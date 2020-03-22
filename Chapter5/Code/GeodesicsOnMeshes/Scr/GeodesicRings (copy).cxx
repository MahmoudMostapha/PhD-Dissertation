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
#include <vtkMath.h>
#include <vtkPolyDataNormals.h>
#include <stdlib.h>     /* abs */
#include <math.h>       /* cos */
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>
#include <vtkCleanPolyData.h>

#define PI 3.14159265


double *row(double array[][3], int which)
{
    double* result = new double[3];
    for (int i=0; i<3; i++)
    {
        result[i] = array[which][i];
    }
    return result;
    delete[] result;
}

int CheckDistance(double P1[3], double P2[3], double Thr)
{
    int check = 1;
    if (vtkMath::Distance2BetweenPoints(P1, P2) > Thr)
    {
    check = 0;
    }
    return check;
}

int main(int argc, char* argv[])
{

if (argc < 4)
{
std::cerr << "Usage: " << argv[0] << "SurfaceMesh.vtk OutputCSVFileName Step [StartVertex] [EndVertex]" << std::endl;
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
Result.open(argv[2]);

vtkIdType StartVertex;
vtkIdType EndVertex;
if (argc > 4)
{
StartVertex = atoi(argv[4]);
EndVertex = atoi(argv[5]);
}
else
{
StartVertex = 0;
EndVertex = inputPolyData->GetNumberOfPoints();
}


for(vtkIdType Vertex = StartVertex; Vertex < EndVertex; Vertex++)
{

  // Add the Seed
  vtkSmartPointer<vtkIdList> seed =
      vtkSmartPointer<vtkIdList>::New();
  seed->InsertNextId(Vertex);
  std::cout << "Starting from Vertex " << seed->GetId(0) << std::endl;

//-----------------------------------------------Estimate Local Orientation---------------------------------------------------------

  vtkSmartPointer<vtkFloatArray> Local_Direction = vtkSmartPointer<vtkFloatArray>::New();
  Local_Direction->SetNumberOfComponents(1);
  Local_Direction->SetName("Local_Direction");

  vtkSmartPointer<vtkFloatArray> Phi = vtkFloatArray::SafeDownCast(inputPolyData->GetPointData()->GetArray("Phi"));
  vtkSmartPointer<vtkFloatArray> Theta = vtkFloatArray::SafeDownCast(inputPolyData->GetPointData()->GetArray("Theta"));
  double lat1 = Phi->GetValue(seed->GetId(0));
  double lon1 = Theta->GetValue(seed->GetId(0));

  std::cout << "Array Red" << std::endl;

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

//-----------------------------------------------Geodesic Contours Extraction--------------------------------------------------------

  // Geodesic Filter
  vtkSmartPointer<vtkFastMarchingGeodesicDistance> Geodesic =
  vtkSmartPointer<vtkFastMarchingGeodesicDistance>::New();
  Geodesic->SetInputData(inputPolyData);
  Geodesic->SetFieldDataName("GeoDist");
  //Geodesic->SetDistanceStopCriterion(10);
  Geodesic->SetSeeds(seed);
  Geodesic->Update();
  std::cout << "Geodesic Distances Computation Done.. " << std::endl;

   vtkPolyData* outputPolyData = Geodesic->GetOutput();
   outputPolyData->GetPointData()->SetActiveScalars("GeoDist");
  std::cout << "Output surface has " << outputPolyData->GetNumberOfPoints() << " points." << std::endl;

   // Create cutter
  vtkSmartPointer<vtkContourFilter> cutter =
    vtkSmartPointer<vtkContourFilter>::New();
  cutter->SetInputData(outputPolyData);
  cutter->SetValue(0, 0.5*atof(argv[3]));
  cutter->SetValue(1, 1*atof(argv[3]));
  cutter->SetValue(2, 1.5*atof(argv[3]));
  cutter->SetValue(3, 2*atof(argv[3]));
  cutter->SetValue(4, 2.5*atof(argv[3]));
  cutter->SetValue(5, 3*atof(argv[3]));
  cutter->Update();

  //vtkSmartPointer<vtkCleanPolyData> cleaner =
  //vtkSmartPointer<vtkCleanPolyData>::New();
  //cleaner->SetInputConnection(cutter->GetOutputPort());
  //cleaner->Update();

  vtkSmartPointer<vtkPolyDataNormals> Normals =
  vtkSmartPointer<vtkPolyDataNormals>::New();
  Normals->SetInputConnection(cutter->GetOutputPort());
  Normals->Update();

  vtkSmartPointer<vtkStripper> stripper =
    vtkSmartPointer<vtkStripper>::New();
  stripper->SetInputConnection(Normals->GetOutputPort());
  //stripper->JoinContiguousSegmentsOn();
  stripper->Update();

  vtkPolyData* ContoursPolyData = stripper->GetOutput();
  std::cout << "Stripped Contours file has " << ContoursPolyData->GetNumberOfLines() << " lines." << std::endl;

  if (ContoursPolyData->GetNumberOfLines() > 6)
     {
      for (int i = 6; i < ContoursPolyData->GetNumberOfLines(); i++)
      {
      ContoursPolyData->BuildLinks();
      ContoursPolyData->DeleteCell(i);
      }
      ContoursPolyData->RemoveDeletedCells();
     }

  std::cout << "Stripped Contours file has " << ContoursPolyData->GetNumberOfLines() << " lines." << std::endl;
 
  //ContoursPolyData->GetLines()->InitTraversal();
  //vtkSmartPointer<vtkIdList> LinesidList = vtkSmartPointer<vtkIdList>::New();
  //while(ContoursPolyData->GetLines()->GetNextCell(LinesidList))
    //{
   //std::cout << "Line has " << LinesidList->GetNumberOfIds() << " points." << std::endl;
 
    //for(vtkIdType pointId = 0; pointId < LinesidList->GetNumberOfIds(); pointId++)
      //{
     //std::cout << LinesidList->GetId(pointId) << " ";
      //}
    //std::cout << std::endl;
    //}

  vtkSmartPointer<vtkKochanekSpline> spline =
  vtkSmartPointer<vtkKochanekSpline>::New();
  spline->SetDefaultTension(.5);
 
  // Subdivide Contours
  vtkSmartPointer<vtkSplineFilter> divide =
    vtkSmartPointer<vtkSplineFilter>::New();
  divide->SetInputData(ContoursPolyData);
  divide->SetSubdivideToSpecified();
  divide->SetNumberOfSubdivisions(12);
  divide->SetSpline(spline);
  divide->GetSpline()->ClosedOn(); 
  divide->Update();

  vtkPolyData* ContoursDividedPolyData = divide->GetOutput();
  std::cout << "Contours Divided file has " << ContoursDividedPolyData->GetNumberOfLines() << " lines." << std::endl;

  vtkSmartPointer<vtkFloatArray> Level = vtkFloatArray::SafeDownCast(ContoursDividedPolyData->GetPointData()->GetArray("GeoDist"));
  ContoursDividedPolyData->GetLines()->InitTraversal();
  vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();

  vtkSmartPointer<vtkFloatArray> ContoursDivided_Labels = vtkSmartPointer<vtkFloatArray>::New();
  ContoursDivided_Labels->SetNumberOfComponents(1);
  ContoursDivided_Labels->SetName("ContoursDivided_Labels");

 for(vtkIdType ID = 0; ID < ContoursDividedPolyData->GetNumberOfPoints(); ID++)
     {
       ContoursDivided_Labels->InsertNextValue(0);
     } 

 double C1[12][3];
 double C2[12][3];
 double C3[12][3];
 double C4[12][3];
 double C5[12][3];
 double C6[12][3];

 for(int Line_ID = 0;  Line_ID < ContoursDividedPolyData->GetNumberOfLines(); Line_ID++)
  {
      ContoursDividedPolyData->GetLines()->GetNextCell(idList);

      if ( abs( Level->GetValue(idList->GetId(0)) - 0.5*atof(argv[3]) ) < 0.00001 )
	 {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;                  
                   for (int i = 0; i < 12; i++)
                   {
		           double P[3];
	      		   ContoursDividedPolyData->GetPoint(idList->GetId(i),P);
                           C1[i][0] = P[0];
			   C1[i][1] = P[1];
			   C1[i][2] = P[2];
	      		   ContoursDivided_Labels->SetValue(idList->GetId(i), i+1);
                   }
         
	 }

      else if ( abs( Level->GetValue(idList->GetId(0)) - 1*atof(argv[3]) ) < 0.00001 )
	 {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;                  
                   for (int i = 0 ; i < 12 ; i++)
                   {
		           double P[3];
	      		   ContoursDividedPolyData->GetPoint(idList->GetId(i),P);
                           C2[i][0] = P[0];
			   C2[i][1] = P[1];
			   C2[i][2] = P[2];
	      		   ContoursDivided_Labels->SetValue(idList->GetId(i), i+1);
                   }
	 }

      else if ( abs( Level->GetValue(idList->GetId(0)) - 1.5*atof(argv[3]) ) < 0.00001 )
	 {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;                  
                   for (int i = 0 ; i < 12 ; i++)
                   {
		           double P[3];
	      		   ContoursDividedPolyData->GetPoint(idList->GetId(i),P);
                           C3[i][0] = P[0];
			   C3[i][1] = P[1];
			   C3[i][2] = P[2];
	      		   ContoursDivided_Labels->SetValue(idList->GetId(i), i+1);
                   }
	 }

      else if ( abs( Level->GetValue(idList->GetId(0)) - 2*atof(argv[3]) ) < 0.00001 )
	 {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;                  
                   for (int i = 0 ; i < 12 ; i++)
                   {
		           double P[3];
	      		   ContoursDividedPolyData->GetPoint(idList->GetId(i),P);
                           C4[i][0] = P[0];
			   C4[i][1] = P[1];
			   C4[i][2] = P[2];
	      		   ContoursDivided_Labels->SetValue(idList->GetId(i), i+1);
                   }
	 }

      else if ( abs( Level->GetValue(idList->GetId(0)) - 2.5*atof(argv[3]) ) < 0.00001 )
	 {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;                  
                   for (int i = 0 ; i < 12 ; i++)
                   {
		           double P[3];
	      		   ContoursDividedPolyData->GetPoint(idList->GetId(i),P);
                           C5[i][0] = P[0];
			   C5[i][1] = P[1];
			   C5[i][2] = P[2];
	      		   ContoursDivided_Labels->SetValue(idList->GetId(i), i+1);
                   }
	 }

      else if ( abs( Level->GetValue(idList->GetId(0)) - 3*atof(argv[3]) ) < 0.00001 )
	 {
      		   std::cout << Line_ID << " Current Level = " << Level->GetValue(idList->GetId(0)) << std::endl;                  
                   for (int i = 0 ; i < 12 ; i++)
                   {
		           double P[3];
	      		   ContoursDividedPolyData->GetPoint(idList->GetId(i),P);
                           C6[i][0] = P[0];
			   C6[i][1] = P[1];
			   C6[i][2] = P[2];
	      		   ContoursDivided_Labels->SetValue(idList->GetId(i), i+1);
                   }
	 }
   }

  ContoursDividedPolyData->GetPointData()->AddArray(ContoursDivided_Labels);

  // Build a locator
  vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
  pointLocator->SetDataSet(inputPolyData);
  pointLocator->BuildLocator();
  pointLocator->Update();
   
if (seed->GetId(0) <0)
{

//-----------------------------------------------Grid Visualization---------------------------------------------------------

//***************************************************Outer Four rays*************************************************************
   int check =1;
   std::cout << "Outer Paths Calculation. " << std::endl;

  // Outer 1
  vtkIdType Start1;
  Start1 = pointLocator->FindClosestPoint(row(C6,0));
  std::cout << "Starting Path from Vertex " << Start1 << std::endl;
  vtkIdType stop1;
  stop1 = pointLocator->FindClosestPoint(row(C4,0));
  vtkSmartPointer<vtkIdList> Stop1 = vtkSmartPointer<vtkIdList>::New();
  Stop1->InsertNextId(stop1);
  std::cout << "Stopping at Vertex " << Stop1->GetId(0) << std::endl;

  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath1 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  if ( CheckDistance(row(C6,0), row(C4,0), 2) == 0 )
{
  Geodesicpath1->SetBeginPointId(Start1);
  Geodesicpath1->SetSeeds(Stop1);
  Geodesicpath1->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath1-> SetInterpolationOrder(1);
  Geodesicpath1->Update();
}
  // Outer 2
  vtkIdType Start2;
  Start2 = pointLocator->FindClosestPoint(row(C6,3));
  std::cout << "Starting Path from Vertex " << Start2 << std::endl;
  vtkIdType stop2;
  stop2 = pointLocator->FindClosestPoint(row(C4,3));
  vtkSmartPointer<vtkIdList> Stop2 = vtkSmartPointer<vtkIdList>::New();
  Stop2->InsertNextId(stop2);
  std::cout << "Stopping at Vertex " << Stop2->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath2 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  if ( CheckDistance(row(C6,3), row(C4,3), 2) == 0 )
{
  Geodesicpath2->SetBeginPointId(Start2);
  Geodesicpath2->SetSeeds(Stop2);
  Geodesicpath2->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath2-> SetInterpolationOrder(1);
  Geodesicpath2->Update();
}

  // Outer 3
  vtkIdType Start3;
  Start3 = pointLocator->FindClosestPoint(row(C6,6));
  std::cout << "Starting Path from Vertex " << Start3 << std::endl;
  vtkIdType stop3;
  stop3 = pointLocator->FindClosestPoint(row(C4,6));
  vtkSmartPointer<vtkIdList> Stop3 = vtkSmartPointer<vtkIdList>::New();
  Stop3->InsertNextId(stop3);
  std::cout << "Stopping at Vertex " << Stop3->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath3 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  if ( CheckDistance(row(C6,6), row(C4,6), 2) == 0 )
{
  Geodesicpath3->SetBeginPointId(Start3);
  Geodesicpath3->SetSeeds(Stop3);
  Geodesicpath3->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath3-> SetInterpolationOrder(1);
  Geodesicpath3->Update();
}

  // Outer 4
  vtkIdType Start4;
  Start4 = pointLocator->FindClosestPoint(row(C6,9));
  std::cout << "Starting Path from Vertex " << Start4 << std::endl;
  vtkIdType stop4;
  stop4 = pointLocator->FindClosestPoint(row(C4,9));
  vtkSmartPointer<vtkIdList> Stop4 = vtkSmartPointer<vtkIdList>::New();
  Stop4->InsertNextId(stop4);
  std::cout << "Stopping at Vertex " << Stop4->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath4 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  if ( CheckDistance(row(C6,9), row(C4,9), 2) == 0 )
{
  Geodesicpath4->SetBeginPointId(Start4);
  Geodesicpath4->SetSeeds(Stop4);
  Geodesicpath4->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath4-> SetInterpolationOrder(1);
  Geodesicpath4->Update();
}

 //***************************************************Middle Two rays*************************************************************
  
  std::cout << "Middle Paths Calculation. " << std::endl;

  // Middle 1
  vtkIdType StartM1;
  StartM1 = pointLocator->FindClosestPoint(row(C4,0));
  std::cout << "Starting Path from Vertex " << StartM1 << std::endl;
  vtkIdType stopM1;
  stopM1 = pointLocator->FindClosestPoint(row(C2,0));
  vtkSmartPointer<vtkIdList> StopM1 = vtkSmartPointer<vtkIdList>::New();
  StopM1->InsertNextId(stopM1);
  std::cout << "Stopping at Vertex " << StopM1->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathM1 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  if ( CheckDistance(row(C4,0), row(C2,0), 2) == 0 )
{
  GeodesicpathM1->SetBeginPointId(StartM1);
  GeodesicpathM1->SetSeeds(StopM1);
  GeodesicpathM1->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathM1-> SetInterpolationOrder(1);
  GeodesicpathM1->Update();
}

  // Middle 2
  vtkIdType StartM2;
  StartM2 = pointLocator->FindClosestPoint(row(C4,6));
  std::cout << "Starting Path from Vertex " << StartM2 << std::endl;
  vtkIdType stopM2;
  stopM2 = pointLocator->FindClosestPoint(row(C2,6));
  vtkSmartPointer<vtkIdList> StopM2 = vtkSmartPointer<vtkIdList>::New();
  StopM2->InsertNextId(stopM2);
  std::cout << "Stopping at Vertex " << StopM2->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> GeodesicpathM2 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  if ( CheckDistance(row(C4,6), row(C2,6), 2) == 0 )
{
  GeodesicpathM2->SetBeginPointId(StartM2);
  GeodesicpathM2->SetSeeds(StopM2);
  GeodesicpathM2->SetInputConnection(0,surfacereader->GetOutputPort());
  GeodesicpathM2-> SetInterpolationOrder(1);
  GeodesicpathM2->Update();
}

   // Create cutter
  vtkSmartPointer<vtkContourFilter> cutter2 =
    vtkSmartPointer<vtkContourFilter>::New();
  cutter2->SetInputData(outputPolyData);
  cutter2->SetValue(1, 1*atof(argv[3]));
  cutter2->SetValue(2, 2*atof(argv[3]));
  cutter2->SetValue(3, 3*atof(argv[3]));
  cutter2->Update();

  vtkSmartPointer<vtkStripper> stripper2 =
    vtkSmartPointer<vtkStripper>::New();
  stripper2->SetInputConnection(cutter2->GetOutputPort());
  stripper2->Update();

  vtkPolyData* ContoursPolyData2 = stripper2->GetOutput();
  std::cout << "Contours file has " << ContoursPolyData2->GetNumberOfLines() << " lines." << std::endl;

  //Append the meshes 
  vtkSmartPointer<vtkAppendPolyData> appendFilter =
    vtkSmartPointer<vtkAppendPolyData>::New();
  appendFilter->AddInputData(ContoursPolyData2);
  appendFilter->AddInputData(Geodesicpath1->GetOutput());
  appendFilter->AddInputData(Geodesicpath2->GetOutput());
  appendFilter->AddInputData(Geodesicpath3->GetOutput());
  appendFilter->AddInputData(Geodesicpath4->GetOutput());
  appendFilter->AddInputData(GeodesicpathM1->GetOutput());
  appendFilter->AddInputData(GeodesicpathM2->GetOutput());
  appendFilter->Update();
  vtkPolyData* Grid = appendFilter->GetOutput();
  std::cout << "Final Grid file has " << Grid->GetNumberOfLines() << " lines." << std::endl;

  std::stringstream ss;
  ss << seed->GetId(0);
  std::string Prefix = ss.str();
  std::string ResultFileName = Prefix + "_" + argv[1] + "_output.vtk";
  // Write Results
  vtkSmartPointer<vtkPolyDataWriter> PolyDatapolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  PolyDatapolywriter->SetFileName(ResultFileName.c_str());
  PolyDatapolywriter->SetInputData(outputPolyData);
  PolyDatapolywriter->Write();

  ResultFileName = Prefix + "_" + argv[1] + "_ContoursPolyData.vtk";
  vtkSmartPointer<vtkPolyDataWriter> ContoursPolyDatapolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  ContoursPolyDatapolywriter->SetFileName(ResultFileName.c_str());
  ContoursPolyDatapolywriter->SetInputData(ContoursPolyData2);
  ContoursPolyDatapolywriter->Write();

  ResultFileName = Prefix + "_" + argv[1] + "_Grid.vtk";
  vtkSmartPointer<vtkPolyDataWriter> Gridpolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  Gridpolywriter->SetFileName(ResultFileName.c_str());
  Gridpolywriter->SetInputData(Grid);
  Gridpolywriter->Write();

}
//-----------------------------------------------Extract Regions Vertices---------------------------------------------------------
Result << seed->GetId(0);
//Region (1)
//Result << "," << seed->GetId(0);
for (int i = 0 ; i < 12 ; i=i+4)
{
Result << "," << pointLocator->FindClosestPoint(row(C1,i));
}
for (int i = 0 ; i < 12 ; i=i+4)
{
Result << "," << pointLocator->FindClosestPoint(row(C2,i)) ;
}
Result << endl;
Result << "," << 0 ;
Result << seed->GetId(0);
//Region (2)
for (int i = 0 ; i <= 6 ; i=i+6)
{
Result << "," << pointLocator->FindClosestPoint(row(C2,i));
}
for (int i = 2 ; i < 6 ; i=i+2)
{
Result << "," << pointLocator->FindClosestPoint(row(C3,i));
}
for (int i = 0 ; i <= 6 ; i=i+6)
{
Result << "," << pointLocator->FindClosestPoint(row(C4,i));
}
Result << "," << (Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C1,2)))+Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C1,4))))/2;
Result << endl;
Result << seed->GetId(0);
//Region (3)
for (int i = 6 ; i <= 12 ; i=i+6)
{
Result << "," << pointLocator->FindClosestPoint(row(C2,i%12));
}
for (int i = 8 ; i < 12 ; i=i+2)
{
Result << "," << pointLocator->FindClosestPoint(row(C3,i%12));
}
for (int i = 6 ; i <= 12 ; i=i+6)
{
Result << "," << pointLocator->FindClosestPoint(row(C4,i%12));
}
Result << "," << (Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C3,8)))+Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C3,10))))/2;
Result << endl;
Result << seed->GetId(0);
//Region (4)
for (int i = 0 ; i <= 3 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C4,i));
}
for (int i = 1 ; i < 3 ; i++)
{
Result << "," << pointLocator->FindClosestPoint(row(C5,i));
}
for (int i = 0 ; i <= 3 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C6,i));
}
Result << "," << (Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,1)))+Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,2))))/2;
Result << endl;
Result << seed->GetId(0);
//Region (5)
for (int i = 3 ; i <= 6 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C4,i));
}
for (int i = 4 ; i < 6 ; i++)
{
Result << "," << pointLocator->FindClosestPoint(row(C5,i));
}
for (int i = 3 ; i <= 6 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C6,i));
}
Result << "," << (Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,4)))+Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,2))))/2;
Result << endl;
Result << seed->GetId(0);
//Region (6)
for (int i = 6 ; i <= 9 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C4,i));
}
for (int i = 7 ; i < 9 ; i++)
{
Result << "," << pointLocator->FindClosestPoint(row(C5,i));
}
for (int i = 6 ; i <= 9 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C6,i));
}
Result << "," << (Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,7)))+Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,8))))/2;
Result << endl;
Result << seed->GetId(0);
//Region (7)
for (int i = 9 ; i <= 12 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C4,i%12));
}
for (int i = 10 ; i < 12 ; i++)
{
Result << "," << pointLocator->FindClosestPoint(row(C5,i%12));
}
for (int i = 9 ; i <= 12 ; i=i+3)
{
Result << "," << pointLocator->FindClosestPoint(row(C6,i%12));
}
Result << "," << (Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,10%12)))+Local_Direction->GetValue(pointLocator->FindClosestPoint(row(C5,11%12))))/2;
Result << endl;
}

Result.close();
int stop_s=clock();
cout << "time: " << (((float)(stop_s-start_s))/CLOCKS_PER_SEC)/60 <<" min" << endl;

return EXIT_SUCCESS;
}

