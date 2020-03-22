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


#include <stdlib.h>     /* abs */
#include <math.h>       /* cos */
#define PI 3.14159265


int main(int argc, char* argv[])
{
  if (argc < 4)
    {
    std::cerr << "Usage: " << argv[0] << "SurfaceMesh.vtk SphereMesh.vtk SeedID" << std::endl;
    return EXIT_FAILURE;
    }

  // Get all surface data from the file
  vtkSmartPointer<vtkPolyDataReader> surfacereader =
  vtkSmartPointer<vtkPolyDataReader>::New();
  surfacereader->SetFileName(argv[1]);
  surfacereader->Update();

  vtkPolyData* inputPolyData = surfacereader->GetOutput();
  std::cout << "Input surface has " << inputPolyData->GetNumberOfPoints() << " points." << std::endl;


  // Get all sphere data from the file
  vtkSmartPointer<vtkPolyDataReader> spherereader =
  vtkSmartPointer<vtkPolyDataReader>::New();
  spherereader->SetFileName(argv[2]);
  spherereader->Update();

  vtkPolyData* inputsphere = spherereader->GetOutput();
  std::cout << "Input sphere has " << inputsphere->GetNumberOfPoints() << " points." << std::endl;

  // Add the Seed
  vtkSmartPointer<vtkIdList> seed =
      vtkSmartPointer<vtkIdList>::New();
  seed->InsertNextId(atoi(argv[3]));
  std::cout << "Starting from Vertex " << seed->GetId(0) << std::endl;

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

  vtkSmartPointer<vtkFloatArray> Local_Direction = vtkSmartPointer<vtkFloatArray>::New();
  Local_Direction->SetNumberOfComponents(1);
  Local_Direction->SetName("Local_Direction");

  vtkSmartPointer<vtkFloatArray> Local_Direction_Ind = vtkSmartPointer<vtkFloatArray>::New();
  Local_Direction_Ind->SetNumberOfComponents(1);
  Local_Direction_Ind->SetName("Local_Direction_Ind");

  vtkSmartPointer<vtkFloatArray> Phi = vtkFloatArray::SafeDownCast(outputPolyData->GetPointData()->GetArray("Phi"));
  vtkSmartPointer<vtkFloatArray> Theta = vtkFloatArray::SafeDownCast(outputPolyData->GetPointData()->GetArray("Theta"));
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
          Local_Direction_Ind->InsertNextValue(ceil(Local_Direction->GetValue(ID)/(PI/2)));
 
     }

  outputPolyData->GetPointData()->AddArray(Local_Direction);
  inputsphere->GetPointData()->AddArray(Local_Direction);
  outputPolyData->GetPointData()->AddArray(Local_Direction_Ind);
  inputsphere->GetPointData()->AddArray(Local_Direction_Ind);

  vtkSmartPointer<vtkFloatArray> FMMDist = vtkFloatArray::SafeDownCast(outputPolyData->GetPointData()->GetArray("FMMDist"));
  vtkSmartPointer<vtkFloatArray> FMMDist_Ind = vtkSmartPointer<vtkFloatArray>::New();
  FMMDist_Ind->SetNumberOfComponents(1);
  FMMDist_Ind->SetName(" FMMDist_Ind");

  int I;
  for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
	    {
               double Dist = FMMDist->GetValue(ID);

		          if ( Dist < 3 ) 
		             {
		             I = 1;
		             } 
		          else if ( Dist < 6 ) 
		             {
		             I = 2;
		             } 
		          else if ( Dist < 9 ) 
		             {
		             I = 3;
                             }
		          else
			     {
		             I = 0;
			     } 
             FMMDist_Ind->InsertNextValue(I);

            }

  vtkSmartPointer<vtkFloatArray> Coordinates = vtkSmartPointer<vtkFloatArray>::New();
  Coordinates->SetNumberOfComponents(1);
  Coordinates->SetName("Coordinates");

  for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
     {
     if (FMMDist_Ind->GetValue(ID) == 1)
        {
           Coordinates->InsertNextValue(1);
        }
     else if ((Local_Direction_Ind->GetValue(ID) == 1 || Local_Direction_Ind->GetValue(ID) == 2) && FMMDist_Ind->GetValue(ID) == 2)
        {
           Coordinates->InsertNextValue(2);
        }
     else if ((Local_Direction_Ind->GetValue(ID) == 3 || Local_Direction_Ind->GetValue(ID) == 4) && FMMDist_Ind->GetValue(ID) == 2)
        {
           Coordinates->InsertNextValue(3);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 1 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(4);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 2 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(5);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 3 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(6);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 4 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(7);
        }
     else
        {
           Coordinates->InsertNextValue(0);
        }
    }

/*
  for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
     {
     if (Local_Direction_Ind->GetValue(ID) == 1 && FMMDist_Ind->GetValue(ID) == 1)
        {
           Coordinates->InsertNextValue(1);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 2 && FMMDist_Ind->GetValue(ID) == 1)
        {
           Coordinates->InsertNextValue(2);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 3 && FMMDist_Ind->GetValue(ID) == 1)
        {
           Coordinates->InsertNextValue(3);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 4 && FMMDist_Ind->GetValue(ID) == 1)
        {
           Coordinates->InsertNextValue(4);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 1 && FMMDist_Ind->GetValue(ID) == 2)
        {
           Coordinates->InsertNextValue(5);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 2 && FMMDist_Ind->GetValue(ID) == 2)
        {
           Coordinates->InsertNextValue(6);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 3 && FMMDist_Ind->GetValue(ID) == 2)
        {
           Coordinates->InsertNextValue(7);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 4 && FMMDist_Ind->GetValue(ID) == 2)
        {
           Coordinates->InsertNextValue(8);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 1 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(9);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 2 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(10);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 3 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(11);
        }
     else if (Local_Direction_Ind->GetValue(ID) == 4 && FMMDist_Ind->GetValue(ID) == 3)
        {
           Coordinates->InsertNextValue(12);
        }
     else
        {
           Coordinates->InsertNextValue(0);
        }
    }
*/
  outputPolyData->GetPointData()->AddArray(Coordinates);
  inputsphere->GetPointData()->AddArray(Coordinates);
/*
  std::string FileName = argv[1];
  std::string NewFileName = FileName.substr(0, FileName.size()-3);
  std::string ResultFileName = NewFileName + "Coordinates.txt";
  ofstream Result;
  Result.open (ResultFileName.c_str());
  Result << "NUMBER_OF_POINTS=" << inputPolyData->GetNumberOfPoints() << endl; 
  Result << "DIMENSION=1" << endl;
  Result << "TYPE=Scalar" << endl;
  for(vtkIdType vertex = 0; vertex < inputPolyData->GetNumberOfPoints(); vertex++)
	    {
                Result << Coordinates->GetValue(vertex) << endl;
            }
  Result.close();
*/
  // Write Surface Results
  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  polywriter->SetFileName(argv[1]);
  polywriter->SetInputData(outputPolyData);
  polywriter->Write();

  // Write Sphere Results
  vtkSmartPointer<vtkPolyDataWriter> spherewriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  spherewriter->SetFileName(argv[2]);
  spherewriter->SetInputData(inputsphere);
  spherewriter->Write();

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

  // Write Surface Results
  vtkSmartPointer<vtkPolyDataWriter> cutterwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  cutterwriter->SetFileName("Contours.vtk");
  cutterwriter->SetInputData(ContoursPolyData);
  cutterwriter->Write();

  vtkSmartPointer<vtkKochanekSpline> spline =
    vtkSmartPointer<vtkKochanekSpline>::New();
  spline->SetDefaultTension(.5);
 
  // Subdivide Contours
  vtkSmartPointer<vtkSplineFilter> divide =
    vtkSmartPointer<vtkSplineFilter>::New();
  divide->SetInputData(ContoursPolyData);
  divide->SetSubdivideToSpecified();
  divide->SetNumberOfSubdivisions(4);
  divide->SetSpline(spline);
  divide->GetSpline()->ClosedOn();
  divide->Update();

  vtkPolyData* ContoursDividedPolyData = divide->GetOutput();
  std::cout << "Contours Divided file has " << ContoursDividedPolyData->GetNumberOfLines() << " lines." << std::endl;

  // Write Surface Results
  vtkSmartPointer<vtkPolyDataWriter> dividewriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  dividewriter->SetFileName("Contours_Divided.vtk");
  dividewriter->SetInputData(ContoursDividedPolyData);
  dividewriter->Write();

  // Build a locator
  vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
  pointLocator->SetDataSet(inputPolyData);
  pointLocator->BuildLocator();
  pointLocator->Update();

  //vtkSmartPointer<vtkCellArray> lines = ContoursDividedPolyData->GetLines(); 
  ContoursDividedPolyData->GetLines()->InitTraversal();
  vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
  ContoursDividedPolyData->GetLines()->GetNextCell(idList);

  double P1[3];
  ContoursDividedPolyData->GetPoint(idList->GetId(0),P1);
  vtkIdType start1;
  start1 = pointLocator->FindClosestPoint(P1);
  std::cout << "Starting Path from Vertex " << start1 << std::endl;
  std::cout << "Stopping at Vertex " << seed->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath1 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath1->SetBeginPointId(start1);
  Geodesicpath1->SetSeeds(seed);
  Geodesicpath1->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath1-> SetInterpolationOrder(1);
  Geodesicpath1->Update();
  std::cout << "First Geodesic Path Computation Done.. " << std::endl;

  double P2[3];
  ContoursDividedPolyData->GetPoint(idList->GetId(1),P2);
  vtkIdType start2;
  start2 = pointLocator->FindClosestPoint(P2);
  std::cout << "Starting Path from Vertex " << start2 << std::endl;
  std::cout << "Stopping at Vertex " << seed->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath2 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath2->SetBeginPointId(start2);
  Geodesicpath2->SetSeeds(seed);
  Geodesicpath2->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath2-> SetInterpolationOrder(1);
  Geodesicpath2->Update();
  std::cout << "Second Geodesic Path Computation Done.. " << std::endl;

  double P3[3];
  ContoursDividedPolyData->GetPoint(idList->GetId(2),P3);
  vtkIdType start3;
  start3 = pointLocator->FindClosestPoint(P3);
  std::cout << "Starting Path from Vertex " << start3 << std::endl;
  std::cout << "Stopping at Vertex " << seed->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath3 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath3->SetBeginPointId(start3);
  Geodesicpath3->SetSeeds(seed);
  Geodesicpath3->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath3-> SetInterpolationOrder(1);
  Geodesicpath3->Update();
  std::cout << "Third Geodesic Path Computation Done.. " << std::endl;

  double P4[3];
  ContoursDividedPolyData->GetPoint(idList->GetId(3),P4);
  vtkIdType start4;
  start4 = pointLocator->FindClosestPoint(P4);
  std::cout << "Starting Path from Vertex " << start4 << std::endl;
  std::cout << "Stopping at Vertex " << seed->GetId(0) << std::endl;
  // Extract Path
  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath4 =
  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
  Geodesicpath4->SetBeginPointId(start4);
  Geodesicpath4->SetSeeds(seed);
  Geodesicpath4->SetInputConnection(0,surfacereader->GetOutputPort());
  Geodesicpath4-> SetInterpolationOrder(1);
  Geodesicpath4->Update();
  std::cout << "Fourth Geodesic Path Computation Done.. " << std::endl;

  //Append the two meshes 
  vtkSmartPointer<vtkAppendPolyData> appendFilter =
    vtkSmartPointer<vtkAppendPolyData>::New();
  appendFilter->AddInputData(ContoursPolyData);
  appendFilter->AddInputData(Geodesicpath1->GetOutput());
  appendFilter->AddInputData(Geodesicpath2->GetOutput());
  appendFilter->AddInputData(Geodesicpath3->GetOutput());
  appendFilter->AddInputData(Geodesicpath4->GetOutput());
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
  vtkSmartPointer<vtkIdList> idList2 = vtkSmartPointer<vtkIdList>::New();
  float Line_ID = 1;
  while(Grid->GetLines()->GetNextCell(idList2))
    {
    for(vtkIdType pointId = 0; pointId < idList2->GetNumberOfIds(); pointId++)
      {
      Grid_Labels->SetValue(idList2->GetId(pointId), Line_ID);
      }    
      Line_ID+= 1;
    }
  Grid->GetPointData()->AddArray(Grid_Labels);


  // Write Results
  vtkSmartPointer<vtkPolyDataWriter> Rayspolywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  Rayspolywriter->SetFileName("Grid.vtk");
  Rayspolywriter->SetInputData(appendFilter->GetOutput());
  Rayspolywriter->Write();


  return EXIT_SUCCESS;
}


/*
  vtkSmartPointer<vtkFloatArray> FMMDist_Ind = vtkSmartPointer<vtkFloatArray>::New();
  FMMDist_Ind->SetNumberOfComponents(1);
  FMMDist_Ind->SetName("GeodesicRings");

  int I;
  for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
	    {
               double Dist = FMMDist->GetValue(ID);
               if (Dist > 0)
               {
		          if ( Dist < 5 ) 
		             {
		             I = 1;
		             } 
		          else if ( Dist < 10 ) 
		             {
		             I = 3;
                             //std::cout << "ID " << ID << std::endl;
                             //std::cout << "Phi " << (Phi->GetValue(ID)*180)/PI << std::endl;
                             //std::cout << "Theta " << (Theta->GetValue(ID)*180)/PI << std::endl;
		             } 
		          else if ( Dist > 10 ) 
		             {
		             I = 0;
		             }
                }
                else
                {
                 I = Dist;
                } 
             
             FMMDist_Ind->InsertNextValue(I);

            }

*/


  //vtkSmartPointer<vtkFloatArray> FMMDist = vtkFloatArray::SafeDownCast(outputPolyData->GetPointData()->GetArray("FMMDist"));

/*
  vtkSmartPointer<vtkFloatArray> Ind = vtkSmartPointer<vtkFloatArray>::New();
  Ind->SetNumberOfComponents(1);
  Ind->SetName("Pie");

  for(vtkIdType ID = 0; ID < inputPolyData->GetNumberOfPoints(); ID++)
     {

       if (FMMDist_Ind->GetValue(ID) == 1)
          {
             if ( Local_Direction->GetValue(ID) < PI)
                {
                 Ind->InsertNextValue(1);
                }
                else
                {
                 Ind->InsertNextValue(2);
                } 
           }

       else if (FMMDist_Ind->GetValue(ID) == 3)
          {
              if ( Local_Direction->GetValue(ID) < PI)
                {
                 Ind->InsertNextValue(3);
                }
                else
                {
                 Ind->InsertNextValue(4);
                } 
           }
       else
         {
            Ind->InsertNextValue(0);
         }
     }


  double Tol = (1 * PI)/180;
  for(int i = 0; i < 4; i++)
     {

      double CurrentPhi = 0.0 + (PI/2) * i;
      std::cout << "Current Destination Phi " << (CurrentPhi*180)/PI << std::endl;

      int Max_ID = 0;
      double Max_Dist = 0.0;
      for(vtkIdType ID = 0; ID < outputPolyData->GetNumberOfPoints(); ID++)
	    {
              
		     if (FMMDist_Ind->GetValue(ID) == 3 & abs(Local_Direction->GetValue(ID) - CurrentPhi ) < Tol)
		      { 
                          if (FMMDist->GetValue(ID) > Max_Dist)
                              {
                                   Max_Dist = FMMDist->GetValue(ID);
		                   Max_ID = ID;
                              }
		      }
            }
      
          std::cout << "The " << i <<" Destination Vertex ID is " << Max_ID << std::endl;

          vtkIdType Start = seed->GetId(0);
          std::cout << "Starting Path from Vertex " << Start << std::endl;

	  // Add the Stop Seed
	  vtkSmartPointer<vtkIdList> Stop =
	      vtkSmartPointer<vtkIdList>::New();
	  Stop->InsertNextId(Max_ID);
	  std::cout << "Stopping at Vertex " << Stop->GetId(0) << std::endl;

	  // Extract Path
	  vtkSmartPointer<vtkFastMarchingGeodesicPath> Geodesicpath =
	  vtkSmartPointer<vtkFastMarchingGeodesicPath>::New();
	  Geodesicpath->SetBeginPointId(Start);
	  Geodesicpath->SetSeeds(Stop);
	  Geodesicpath->SetInputConnection(0,surfacereader->GetOutputPort());
          Geodesicpath-> SetInterpolationOrder(1);
	  Geodesicpath->Update();
	  std::cout << i << " Geodesic Path Computation Done.. " << std::endl;

	  // Write Results
	  vtkSmartPointer<vtkPolyDataWriter> polywriter = vtkSmartPointer<vtkPolyDataWriter>::New();
	  polywriter->SetFileName(argv[1]);
	  polywriter->SetInputData(Geodesicpath->GetOutput());
	  polywriter->Write();

	  // Add the Stop Seed
	  vtkSmartPointer<vtkIdList> Path =
	      vtkSmartPointer<vtkIdList>::New();
          Path = Geodesicpath->GetZerothOrderPathPointIds();

              //for(vtkIdType Idx = 0; Idx < Path->GetNumberOfIds(); Idx++)
                 //{
                   //FMMDist_Ind->SetValue(Path->GetId(Idx),5+i);
                 //}   

                 //FMMDist_Ind->SetValue(Max_ID,100);
     }


  outputPolyData->GetPointData()->AddArray(FMMDist_Ind);
  inputsphere->GetPointData()->AddArray(FMMDist_Ind);
  inputsphere->GetPointData()->AddArray(Phi);
  inputsphere->GetPointData()->AddArray(Theta);
  outputPolyData->GetPointData()->AddArray(Local_Direction);
  inputsphere->GetPointData()->AddArray(Local_Direction);
  outputPolyData->GetPointData()->AddArray(Ind);
  inputsphere->GetPointData()->AddArray(Ind);
*/


