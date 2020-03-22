#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPointSource.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkKdTreePointLocator.h>
#include <vtkDecimatePro.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataReader.h>
#include <vtkExtractEdges.h>
#include <vtkLine.h>
#include <vtkMath.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>

double ComputeAverageEdgeLength(vtkPolyData* Surface)
{

vtkSmartPointer<vtkExtractEdges> extractEdges = 
    vtkSmartPointer<vtkExtractEdges>::New();
  extractEdges->SetInputData(Surface);
  extractEdges->Update();

    double AvgEdgeLength = 0.0;
  // Traverse all of the edges
  for(vtkIdType i = 0; i < extractEdges->GetOutput()->GetNumberOfCells(); i++)
    {
    vtkSmartPointer<vtkLine> line = vtkLine::SafeDownCast(extractEdges->GetOutput()->GetCell(i)); 
    double P1[3];
    double P2[3];
    Surface->GetPoint(line->GetPointIds()->GetId(0),P1);
    Surface->GetPoint(line->GetPointIds()->GetId(1),P2);
    double squaredDistance = vtkMath::Distance2BetweenPoints(P1, P2);
    double distance = sqrt(squaredDistance);
    AvgEdgeLength+= distance;
    }
    return AvgEdgeLength/(extractEdges->GetOutput()->GetNumberOfCells());
}

int main(int argc, char* argv[])
{

if (argc < 2)
{
	std::cerr << "Usage: " << argv[0] << "HighResolutionMesh.vtk [#Neighbours] [DecimationFactor]" << std::endl;
	return EXIT_FAILURE;
}

// Get all surface data from the file
vtkSmartPointer<vtkPolyDataReader> HRsurfacereader =
	vtkSmartPointer<vtkPolyDataReader>::New();
HRsurfacereader->SetFileName(argv[1]);
HRsurfacereader->Update();
vtkPolyData* HRsurfacere = HRsurfacereader->GetOutput();
std::cout << "HR surface has " << HRsurfacere->GetNumberOfPoints() << " points." << std::endl;
std::cout << "Original_AVG_EdgeLength = " << ComputeAverageEdgeLength(HRsurfacere) << std::endl;

ofstream Result;
Result.open("Avergae_Egde_Length.csv");
Result << ComputeAverageEdgeLength(HRsurfacere) << std::endl;

int N;
if (argc == 3)
{
N = atoi(argv[2]);
}
else
{
N = 5;
}

double D;
if (argc == 4)
{
D = atof(argv[3]);
}
else
{
D = 0.5;
}


std::cout << "Before decimation" << std::endl << "------------" << std::endl;
std::cout << "There are " << HRsurfacere->GetNumberOfPoints() << " points." << std::endl;
std::cout << "There are " << HRsurfacere->GetNumberOfPolys() << " polygons." << std::endl;

//------------------------------------------------------First Level--------------------------------------------------------------------

std::cout << "Lowering Surface Resolution by " << D <<"%" << std::endl;
vtkSmartPointer<vtkDecimatePro> decimate =
vtkSmartPointer<vtkDecimatePro>::New();
decimate->SetInputData(HRsurfacere);
decimate->SetTargetReduction(D); 
decimate->Update();

vtkSmartPointer<vtkPolyData> decimated1 =
vtkSmartPointer<vtkPolyData>::New();
decimated1->ShallowCopy(decimate->GetOutput());

std::cout << "After L1 decimation" << std::endl << "------------" << std::endl;
std::cout << "There are " << decimated1->GetNumberOfPoints() << " points." << std::endl;
std::cout << "There are " << decimated1->GetNumberOfPolys() << " polygons." << std::endl;
std::cout << "L1_AVG_EdgeLength = " << ComputeAverageEdgeLength(decimated1) << std::endl;
Result << ComputeAverageEdgeLength(decimated1) << std::endl;

// Write Results
vtkSmartPointer<vtkPolyDataWriter> polywriter1 = vtkSmartPointer<vtkPolyDataWriter>::New();
polywriter1->SetFileName("Deciamted_L1.vtk");
polywriter1->SetInputData(decimated1);
polywriter1->Write();

//Create the tree
vtkSmartPointer<vtkKdTreePointLocator> pointTree1 = 
vtkSmartPointer<vtkKdTreePointLocator>::New();
pointTree1->SetDataSet(HRsurfacere);
pointTree1->BuildLocator();

//Print the Results
ofstream Result1;
Result1.open ("DeciamtionMap_L1.csv");
for(int ID = 0;  ID < decimated1->GetNumberOfPoints(); ID++)
{
double P[3];
decimated1->GetPoint(ID,P);
vtkSmartPointer<vtkIdList> result =  vtkSmartPointer<vtkIdList>::New();
pointTree1->FindClosestNPoints(N, P, result);
Result1 << result->GetId(0);
for(vtkIdType i = 1; i < N; i++)
{
Result1 << ',' << result->GetId(i);
}
Result1 << endl;
}
Result1.close();

//------------------------------------------------------Second Level------------------------------------------------------------------

std::cout << "Lowering Surface Resolution by " << D <<"%" << std::endl;
decimate->SetInputData(decimated1);
decimate->SetTargetReduction(D); 
decimate->Update();

vtkSmartPointer<vtkPolyData> decimated2 =
vtkSmartPointer<vtkPolyData>::New();
decimated2->ShallowCopy(decimate->GetOutput());

std::cout << "After L2 decimation" << std::endl << "------------" << std::endl;
std::cout << "There are " << decimated2->GetNumberOfPoints() << " points." << std::endl;
std::cout << "There are " << decimated2->GetNumberOfPolys() << " polygons." << std::endl;
std::cout << "L2_AVG_EdgeLength = " << ComputeAverageEdgeLength(decimated2) << std::endl;
Result << ComputeAverageEdgeLength(decimated2) << std::endl;

// Write Results
vtkSmartPointer<vtkPolyDataWriter> polywriter2 = vtkSmartPointer<vtkPolyDataWriter>::New();
polywriter2->SetFileName("Deciamted_L2.vtk");
polywriter2->SetInputData(decimated2);
polywriter2->Write();

//Create the tree
vtkSmartPointer<vtkKdTreePointLocator> pointTree2 = 
vtkSmartPointer<vtkKdTreePointLocator>::New();
pointTree2->SetDataSet(decimated1);
pointTree2->BuildLocator();

//Print the Results
ofstream Result2;
Result2.open ("DeciamtionMap_L2.csv");
for(int ID = 0;  ID < decimated2->GetNumberOfPoints(); ID++)
{
double P[3];
decimated2->GetPoint(ID,P);
vtkSmartPointer<vtkIdList> result =  vtkSmartPointer<vtkIdList>::New();
pointTree2->FindClosestNPoints(N, P, result);
Result2 << result->GetId(0);
for(vtkIdType i = 1; i < N; i++)
{
Result2 << ',' << result->GetId(i);
}
Result2 << endl;
}
Result2.close();

//------------------------------------------------------Third Level------------------------------------------------------------------

std::cout << "Lowering Surface Resolution by " << D <<"%" << std::endl;
decimate->SetInputData(decimated2);
decimate->SetTargetReduction(D); 
decimate->Update();

vtkSmartPointer<vtkPolyData> decimated3 =
vtkSmartPointer<vtkPolyData>::New();
decimated3->ShallowCopy(decimate->GetOutput());

std::cout << "After L3 decimation" << std::endl << "------------" << std::endl;
std::cout << "There are " << decimated3->GetNumberOfPoints() << " points." << std::endl;
std::cout << "There are " << decimated3->GetNumberOfPolys() << " polygons." << std::endl;
std::cout << "L3_AVG_EdgeLength = " << ComputeAverageEdgeLength(decimated3) << std::endl;
Result << ComputeAverageEdgeLength(decimated3) << std::endl;

// Write Results
vtkSmartPointer<vtkPolyDataWriter> polywriter3 = vtkSmartPointer<vtkPolyDataWriter>::New();
polywriter3->SetFileName("Deciamted_L3.vtk");
polywriter3->SetInputData(decimated3);
polywriter3->Write();

//Create the tree
vtkSmartPointer<vtkKdTreePointLocator> pointTree3 = 
vtkSmartPointer<vtkKdTreePointLocator>::New();
pointTree3->SetDataSet(decimated2);
pointTree3->BuildLocator();

//Print the Results
ofstream Result3;
Result3.open ("DeciamtionMap_L3.csv");
for(int ID = 0;  ID < decimated3->GetNumberOfPoints(); ID++)
{
double P[3];
decimated3->GetPoint(ID,P);
vtkSmartPointer<vtkIdList> result =  vtkSmartPointer<vtkIdList>::New();
pointTree3->FindClosestNPoints(N, P, result);
Result3 << result->GetId(0);
for(vtkIdType i = 1; i < N; i++)
{
Result3 << ',' << result->GetId(i);
}
Result3 << endl;
}
Result3.close();

//------------------------------------------------------Fourth Level------------------------------------------------------------------

std::cout << "Lowering Surface Resolution by " << D <<"%" << std::endl;
decimate->SetInputData(decimated3);
decimate->SetTargetReduction(D); 
decimate->Update();

vtkSmartPointer<vtkPolyData> decimated4 =
vtkSmartPointer<vtkPolyData>::New();
decimated4->ShallowCopy(decimate->GetOutput());

std::cout << "After L4 decimation" << std::endl << "------------" << std::endl;
std::cout << "There are " << decimated4->GetNumberOfPoints() << " points." << std::endl;
std::cout << "There are " << decimated4->GetNumberOfPolys() << " polygons." << std::endl;
std::cout << "L4_AVG_EdgeLength = " << ComputeAverageEdgeLength(decimated4) << std::endl;
Result << ComputeAverageEdgeLength(decimated4) << std::endl;

// Write Results
vtkSmartPointer<vtkPolyDataWriter> polywriter4 = vtkSmartPointer<vtkPolyDataWriter>::New();
polywriter4->SetFileName("Deciamted_L4.vtk");
polywriter4->SetInputData(decimated4);
polywriter4->Write();

//Create the tree
vtkSmartPointer<vtkKdTreePointLocator> pointTree4 = 
vtkSmartPointer<vtkKdTreePointLocator>::New();
pointTree4->SetDataSet(decimated3);
pointTree4->BuildLocator();

//Print the Results
ofstream Result4;
Result4.open ("DeciamtionMap_L4.csv");
for(int ID = 0;  ID < decimated4->GetNumberOfPoints(); ID++)
{
double P[3];
decimated4->GetPoint(ID,P);
vtkSmartPointer<vtkIdList> result =  vtkSmartPointer<vtkIdList>::New();
pointTree4->FindClosestNPoints(N, P, result);
Result4 << result->GetId(0);
for(vtkIdType i = 1; i < N; i++)
{
Result4 << ',' << result->GetId(i);
}
Result4 << endl;
}
Result4.close();

//------------------------------------------------------Fifth Level------------------------------------------------------------------

std::cout << "Lowering Surface Resolution by " << D << "%" << std::endl;
decimate->SetInputData(decimated4);
decimate->SetTargetReduction(D);
decimate->Update();

vtkSmartPointer<vtkPolyData> decimated5 =
vtkSmartPointer<vtkPolyData>::New();
decimated5->ShallowCopy(decimate->GetOutput());

std::cout << "After L5 decimation" << std::endl << "------------" << std::endl;
std::cout << "There are " << decimated5->GetNumberOfPoints() << " points." << std::endl;
std::cout << "There are " << decimated5->GetNumberOfPolys() << " polygons." << std::endl;
std::cout << "L5_AVG_EdgeLength = " << ComputeAverageEdgeLength(decimated5) << std::endl;
Result << ComputeAverageEdgeLength(decimated5) << std::endl;

// Write Results
vtkSmartPointer<vtkPolyDataWriter> polywriter5 = vtkSmartPointer<vtkPolyDataWriter>::New();
polywriter5->SetFileName("Deciamted_L5.vtk");
polywriter5->SetInputData(decimated5);
polywriter5->Write();

//Create the tree
vtkSmartPointer<vtkKdTreePointLocator> pointTree5 =
vtkSmartPointer<vtkKdTreePointLocator>::New();
pointTree5->SetDataSet(decimated4);
pointTree5->BuildLocator();

//Print the Results
ofstream Result5;
Result5.open("DeciamtionMap_L5.csv");
for (int ID = 0; ID < decimated5->GetNumberOfPoints(); ID++)
{
	double P[3];
	decimated5->GetPoint(ID, P);
	vtkSmartPointer<vtkIdList> result = vtkSmartPointer<vtkIdList>::New();
	pointTree5->FindClosestNPoints(N, P, result);
	Result5 << result->GetId(0);
	for (vtkIdType i = 1; i < N; i++)
	{
		Result5 << ',' << result->GetId(i);
	}
	Result5 << endl;
}
Result5.close();


//------------------------------------------------------sixth Level------------------------------------------------------------------

std::cout << "Lowering Surface Resolution by " << D << "%" << std::endl;
decimate->SetInputData(decimated5);
decimate->SetTargetReduction(D);
decimate->Update();

vtkSmartPointer<vtkPolyData> decimated6 =
vtkSmartPointer<vtkPolyData>::New();
decimated6->ShallowCopy(decimate->GetOutput());

std::cout << "After L6 decimation" << std::endl << "------------" << std::endl;
std::cout << "There are " << decimated6->GetNumberOfPoints() << " points." << std::endl;
std::cout << "There are " << decimated6->GetNumberOfPolys() << " polygons." << std::endl;
std::cout << "L6_AVG_EdgeLength = " << ComputeAverageEdgeLength(decimated6) << std::endl;
Result << ComputeAverageEdgeLength(decimated6) << std::endl;

// Write Results
vtkSmartPointer<vtkPolyDataWriter> polywriter6 = vtkSmartPointer<vtkPolyDataWriter>::New();
polywriter6->SetFileName("Deciamted_L6.vtk");
polywriter6->SetInputData(decimated6);
polywriter6->Write();

//Create the tree
vtkSmartPointer<vtkKdTreePointLocator> pointTree6 =
vtkSmartPointer<vtkKdTreePointLocator>::New();
pointTree6->SetDataSet(decimated5);
pointTree6->BuildLocator();

//Print the Results
ofstream Result6;
Result6.open("DeciamtionMap_L6.csv");
for (int ID = 0; ID < decimated6->GetNumberOfPoints(); ID++)
{
	double P[3];
	decimated6->GetPoint(ID, P);
	vtkSmartPointer<vtkIdList> result = vtkSmartPointer<vtkIdList>::New();
	pointTree6->FindClosestNPoints(N, P, result);
	Result6 << result->GetId(0);
	for (vtkIdType i = 1; i < N; i++)
	{
		Result6 << ',' << result->GetId(i);
	}
	Result6 << endl;
}
Result6.close();

  
  return EXIT_SUCCESS;
}
