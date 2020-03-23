//
//  kvolume.cpp
//  ktools
//
//  Created by Joowhi Lee on 9/3/15.
//
//

#include "kvolume.h"
#include "kstreamtracer.h"
#include "kgeometry.h"

#include "piOptions.h"

#include "vtkio.h"

#include <vtkPolyData.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSelectEnclosedPoints.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkExtractGrid.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkGradientFilter.h>
#include <vtkMath.h>
#include <vtkThresholdPoints.h>
#include <vtkCleanPolyData.h>
#include <vtkModifiedBSPTree.h>
#include <vtkCellLocator.h>
#include <vtkPointLocator.h>
#include <vtkGenericCell.h>
#include <vtkPolyDataNormals.h>
#include <vtkInterpolatedVelocityField.h>
#include <vtkSphereSource.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkImageData.h>
#include <vtkImageStencil.h>
#include <vtkPolyDataToImageStencil.h>
#include <vtkMetaImageWriter.h>
#include <vtkImageToStructuredGrid.h>
#include <vtkExtractGrid.h>
#include <vtkSphericalTransform.h>
#include <vtkPolyLine.h>
#include <vtkStreamTracer.h>
#include <vtkCellTreeLocator.h>


#include <ctime>
#include <algorithm>

using namespace pi;
using namespace std;
using namespace std::tr1;

static vtkIO vio;



void findNeighborPoints(vtkCell* cell, vtkIdType pid, set<vtkIdType>& nbrs) {
    for (size_t j = 0; j < cell->GetNumberOfPoints(); j++) {
        vtkIdType cellPt = cell->GetPointId(j);
        if (pid != cellPt) {
            nbrs.insert(cellPt);
        }
    }
}

void runExtractBorderline(Options& opts, StringVector& args) {
	string inputFile = args[0];
	string outputFile = args[1];
	string scalarName = opts.GetString("-scalarName", "labels");
	
	vtkIO vio;
	vtkPolyData* input = vio.readFile(inputFile);
	input->BuildCells();
	input->BuildLinks();
	
	cout << input->GetNumberOfPoints() << endl;
	cout << input->GetNumberOfLines() << endl;
	
	vtkPoints* points = input->GetPoints();
	vtkDataArray* scalar = input->GetPointData()->GetArray(scalarName.c_str());
	
	vector<pair<vtkIdType,vtkIdType> > edgeSet;
	
	for (size_t j = 0; j < input->GetNumberOfCells(); j++) {
		vtkCell* cell = input->GetCell(j);
		vtkIdType p[3]; int s[3];
		p[0] = cell->GetPointId(0);
		p[1] = cell->GetPointId(1);
		p[2] = cell->GetPointId(2);
		
		s[0] = scalar->GetTuple1(p[0]);
		s[1] = scalar->GetTuple1(p[1]);
		s[2] = scalar->GetTuple1(p[2]);
		
		if (s[0] == s[1] && s[1] == s[2] && s[2] == s[0]) {
			continue;
		}
		
		vtkIdType p1, p2;
		if (s[0] != s[1] && s[0] != s[2]) {
			p1 = p[1];
			p2 = p[2];
		} else if (s[1] != s[2] && s[1] != s[0]) {
			p1 = p[2];
			p2 = p[0];
		} else if (s[2] != s[0] && s[2] != s[1]) {
			p1 = p[0];
			p2 = p[1];
		} else {
			continue;
		}
		
		edgeSet.push_back(make_pair(p1, p2));
	}
	
	vtkPolyData* output = vtkPolyData::New();
	output->SetPoints(points);
	
	vtkCellArray* lines = vtkCellArray::New();
	for (size_t j = 0; j < edgeSet.size(); j++) {
		vtkIdList* ids = vtkIdList::New();
		ids->InsertNextId(edgeSet[j].first);
		ids->InsertNextId(edgeSet[j].second);
		lines->InsertNextCell(ids->GetNumberOfIds(), ids->GetPointer(0));
		ids->Delete();
	}
	
	output->SetLines(lines);
	output->BuildCells();
	
	vio.writeFile(outputFile, output);
	cout << "Length of Borderline: " << edgeSet.size() << endl;
}

vtkDataSet* createGridForSphereLikeObject(vtkPolyData* input, int& insideCount, int dims, bool insideOutOn) {
	// x1-x2, y1-y2, z1-z2
	double* bounds = input->GetBounds();
	
	cout << bounds[0] << "," << bounds[1] << endl;
	cout << bounds[2] << "," << bounds[3] << endl;
	cout << bounds[4] << "," << bounds[5] << endl;
	cout << "Grid Dimension: " << dims << endl;
	
	double center[3] = { 0, };
	center[0] = (bounds[0] + bounds[1])/2.0;
	center[1] = (bounds[2] + bounds[3])/2.0;
	center[2] = (bounds[4] + bounds[5])/2.0;
	
	vtkStructuredGrid* grid = vtkStructuredGrid::New();
	grid->SetDimensions(dims + 6, dims + 6, dims + 6);
	
	vtkPoints* gridPoints = vtkPoints::New();
	
	for (int k = 0; k < dims + 6; k++) {
		for (int j = 0; j < dims + 6; j++) {
			for (int i = 0; i < dims + 6; i++) {
				double x = bounds[0] + (i-3)*(bounds[1]-bounds[0])/dims;
				double y = bounds[2] + (j-3)*(bounds[3]-bounds[2])/dims;
				double z = bounds[4] + (k-3)*(bounds[5]-bounds[4])/dims;
				
				gridPoints->InsertNextPoint(x + center[0], y + center[1], z + center[2]);
			}
		}
	}
	
	grid->SetPoints(gridPoints);
	
	vtkSelectEnclosedPoints* encloser = vtkSelectEnclosedPoints::New();
	encloser->SetInputData(grid );
	encloser->SetSurfaceData(input);
	encloser->CheckSurfaceOn();
	if (insideOutOn) {
		cout << "Inside Out Mode" << endl;
		encloser->InsideOutOn();
	}
	encloser->SetTolerance(0);
	encloser->Update();
	
	vtkDataArray* selectedPoints = encloser->GetOutput()->GetPointData()->GetArray("SelectedPoints");
	for (size_t j = 0; j < selectedPoints->GetNumberOfTuples(); j++) {
		if (selectedPoints->GetTuple1(j) == 1) {
			insideCount++;
		}
	}
	return encloser->GetOutput();
}

/// perform scan conversion
/// [input-vtk] [reference-image] [output-image]
///
int runScanConversion(pi::Options& opts, pi::StringVector& args) {
	vtkIO vio;
	vtkPolyData* pd = vio.readFile(args[0]);
	
	vtkNew<vtkImageData> whiteImage;
	
	// mesh bounds
	double bounds[6];
	pd->GetBounds(bounds);
	
	// print bounds
	for (int i = 0; i < 6; i++) {
		cout << bounds[i] << ", ";
	}
	cout << endl;
	
	double dims = opts.GetStringAsReal("-dims", 100.0);
	
	double spacing[3] = { 0, };
	spacing[0] = spacing[1] = spacing[2] = std::min(bounds[1]-bounds[0], std::min(bounds[3]-bounds[2], bounds[5]-bounds[4])) / dims;
	
	whiteImage->SetSpacing(spacing);
	
	// compute dimensions
	int dim[3];
	for (int i = 0; i < 3; i++)
	{
		dim[i] = static_cast<int>(ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i])) + 6;
	}
	whiteImage->SetDimensions(dim);
	whiteImage->SetExtent(0, dim[0] + 6, 0, dim[1] + 6, 0, dim[2] + 6);
	
	double origin[3];
	origin[0] = bounds[0] + spacing[0]*-3;
	origin[1] = bounds[2] + spacing[1]*-3;
	origin[2] = bounds[4] + spacing[2]*-3;
	whiteImage->SetOrigin(origin);
	
#if VTK_MAJOR_VERSION <= 5
	whiteImage->SetScalarTypeToUnsignedChar();
	whiteImage->AllocateScalars();
#else
	whiteImage->AllocateScalars(VTK_UNSIGNED_CHAR,1);
#endif
	// fill the image with foreground voxels:
	unsigned char inval = 255;
	unsigned char outval = 0;
	vtkIdType count = whiteImage->GetNumberOfPoints();
	for (vtkIdType i = 0; i < count; ++i)
	{
		whiteImage->GetPointData()->GetScalars()->SetTuple1(i, inval);
	}
	
	// polygonal data --> image stencil:
	vtkSmartPointer<vtkPolyDataToImageStencil> pol2stenc =
	vtkSmartPointer<vtkPolyDataToImageStencil>::New();
#if VTK_MAJOR_VERSION <= 5
	pol2stenc->SetInput(pd);
#else
	pol2stenc->SetInputData(pd);
#endif
	pol2stenc->SetTolerance(0);
	pol2stenc->SetOutputOrigin(origin);
	pol2stenc->SetOutputSpacing(spacing);
	pol2stenc->SetOutputWholeExtent(whiteImage->GetExtent());
	pol2stenc->Update();
	
	// cut the corresponding white image and set the background:
	vtkSmartPointer<vtkImageStencil> imgstenc =
	vtkSmartPointer<vtkImageStencil>::New();
#if VTK_MAJOR_VERSION <= 5
	imgstenc->SetInput(whiteImage.GetPointer());
	imgstenc->SetStencil(pol2stenc->GetOutput());
#else
	imgstenc->SetBackgroundInputData(whiteImage.GetPointer());
	imgstenc->SetStencilConnection(pol2stenc->GetOutputPort());
#endif
	imgstenc->ReverseStencilOff();
	imgstenc->SetBackgroundValue(outval);
	imgstenc->Update();
	
	vtkSmartPointer<vtkMetaImageWriter> writer =
	vtkSmartPointer<vtkMetaImageWriter>::New();
	
	// must be .mhd format
	writer->SetFileName(args[1].c_str());
#if VTK_MAJOR_VERSION <= 5
	writer->SetInput(imgstenc->GetOutput());
#else
	writer->SetInputData(imgstenc->GetOutput());
#endif
	writer->Write();
	
	return EXIT_SUCCESS;
}



vtkDataSet* createGrid(vtkPolyData* osurf, vtkPolyData* isurf, const int dims, size_t& insideCountOut) {
	
	// compute the common voxel space
	struct GridCreate {
		int dim[3];
		double center[3];
		double spacing[3];
		int extent[6];
		double origin[3];
		
		GridCreate(double* bounds, const int dims) {
			double maxbound = max(bounds[1]-bounds[0], max(bounds[3]-bounds[2], bounds[5]-bounds[4]));
			
			center[0] = (bounds[1]+bounds[0])/2.0;
			center[1] = (bounds[3]+bounds[2])/2.0;
			center[2] = (bounds[5]+bounds[4])/2.0;
			
			double gridSpacing = maxbound / dims;
			spacing[0] = spacing[1] = spacing[2] = gridSpacing;
			
			dim[0] = (bounds[1]-bounds[0])/gridSpacing;
			dim[1] = (bounds[3]-bounds[2])/gridSpacing;
			dim[2] = (bounds[5]-bounds[4])/gridSpacing;
			
			extent[0] = extent[2] = extent[4] = 0;
			extent[1] = dim[0] + 6;
			extent[3] = dim[1] + 6;
			extent[5] = dim[2] + 6;
			
			origin[0] = bounds[0] + gridSpacing*-3;
			origin[1] = bounds[2] + gridSpacing*-3;
			origin[2] = bounds[4] + gridSpacing*-3;
			
			cout << "Grid Dimension: " << dims << "; Grid Spacing: " << gridSpacing << endl;
		}
		
		
		void createImage(vtkImageData* im) {
			im->SetSpacing(spacing);
			im->SetExtent(extent);
			im->SetOrigin(origin);
			im->AllocateScalars(VTK_INT, 1);
			im->GetPointData()->GetScalars()->FillComponent(0, 255);
		}
		
		vtkStructuredGrid* createStencil(vtkImageData* im, vtkPolyData* surf) {
			createImage(im);
			
			vtkNew<vtkPolyDataToImageStencil> psten;
			psten->SetInputData(surf);
			psten->SetOutputOrigin(origin);
			psten->SetOutputSpacing(spacing);
			psten->SetOutputWholeExtent(extent);
			psten->Update();
			
			vtkNew<vtkImageStencil> isten;
			isten->SetInputData(im);
			isten->SetStencilData(psten->GetOutput());
			isten->ReverseStencilOff();
			isten->SetBackgroundValue(0);
			isten->Update();
			
			vtkImageData* imgGrid = isten->GetOutput();
			imgGrid->GetPointData()->GetScalars()->SetName("SampledValue");
			
			vtkNew<vtkImageToStructuredGrid> imgToGrid;
			imgToGrid->SetInputData(imgGrid);
			imgToGrid->Update();
			
			vtkStructuredGrid* output = imgToGrid->GetOutput();
			output->GetPointData()->SetScalars(imgGrid->GetPointData()->GetScalars());
			output->Register(NULL);
			
			return output;
		}
	};
	
	GridCreate gc(osurf->GetBounds(), dims);
	
	vtkNew<vtkImageData> gim;
	vtkNew<vtkImageData> wim;
	
	vtkStructuredGrid* goim = gc.createStencil(gim.GetPointer(), osurf);
	vtkStructuredGrid* woim = gc.createStencil(wim.GetPointer(), isurf);
	
	struct BoundaryCheck {
		size_t subtract(vtkDataSet* aim, vtkDataSet* bim) {
			if (aim->GetNumberOfPoints() != bim->GetNumberOfPoints()) {
				cout << "can't process: the number of points are different!" << endl;
				return 0;
			}
			
			vtkDataArray* aarr = aim->GetPointData()->GetScalars();
			vtkDataArray* barr = bim->GetPointData()->GetScalars();
			
			size_t insideCount = 0;
			for (size_t j = 0; j < aim->GetNumberOfPoints(); j++) {
				int p = aarr->GetTuple1(j);
				int q = barr->GetTuple1(j);
				int o = 700;
				if (p == 255 && q != 255) {
					o = 1;
					insideCount ++;
				} else if (p == 255 && q == 255){
					o = 300;
				}
				aarr->SetTuple1(j, o);
			}
			return insideCount;
		}
		
		void checkSurface(vtkStructuredGrid* grid, vtkPolyData* isurf,  vtkPolyData* osurf) {
			vtkNew<vtkPointLocator> gridLoc;
			gridLoc->SetDataSet(grid);
			gridLoc->BuildLocator();

			vtkIntArray* sampledValue = vtkIntArray::SafeDownCast(grid->GetPointData()->GetScalars());
			
			const size_t nPoints = isurf->GetNumberOfPoints();
			size_t cnt = 0;
			
			for (size_t j = 0; j < nPoints; j++) {
				double p[3];
				
				isurf->GetPoint(j, p);
				vtkIdType pid = gridLoc->FindClosestPoint(p);
				
				int sample = sampledValue->GetValue(pid);
				if (sample == 300) {
					sampledValue->SetValue(pid, 1);
					cnt++;
				}
			}
			cout << "# of inside boundary correction: " << cnt << endl;
			
			cnt = 0;
			const size_t nPoints2 = osurf->GetNumberOfPoints();
			for (size_t j = 0; j < nPoints2; j++) {
				double p[3];
				
				osurf->GetPoint(j, p);
				vtkIdType pid = gridLoc->FindClosestPoint(p);
				
				int sample = sampledValue->GetValue(pid);
				if (sample == 700) {
					sampledValue->SetValue(pid, 1);
					cnt++;
				}
			}
			cout << "# of outside boundary correction: " << cnt << endl;
		}

	};
	
	BoundaryCheck bc;
	insideCountOut = bc.subtract(goim, woim);
	bc.checkSurface(goim, isurf, osurf);
	woim->Delete();
	
	return goim;
}


vtkDataSet* createGridForHumanBrainTopology(vtkPolyData* gmsurf, vtkPolyData* wmsurf, const int dims, size_t &insideCountOut) {
	// x1-x2, y1-y2, z1-z2
	double* bounds = gmsurf->GetBounds();
	
	cout << bounds[0] << "," << bounds[1] << endl;
	cout << bounds[2] << "," << bounds[3] << endl;
	cout << bounds[4] << "," << bounds[5] << endl;
	
	double maxbound = max(bounds[1]-bounds[0], max(bounds[3]-bounds[2], bounds[5]-bounds[4]));
	double center[3] = { 0, };
	center[0] = (bounds[1]+bounds[0])/2.0;
	center[1] = (bounds[3]+bounds[2])/2.0;
	center[2] = (bounds[5]+bounds[4])/2.0;
	
	double gridSpacing = maxbound / dims;
	
	cout << "Grid Dimension: " << dims << "; Grid Spacing: " << gridSpacing << endl;
	
	
	size_t xdim = (bounds[1]-bounds[0])/gridSpacing;
	size_t ydim = (bounds[3]-bounds[2])/gridSpacing;
	size_t zdim = (bounds[5]-bounds[4])/gridSpacing;
	
	vtkStructuredGrid* grid = vtkStructuredGrid::New();
	grid->SetDimensions(xdim + 6, ydim + 6, zdim + 6);
	
	vtkPoints* gridPoints = vtkPoints::New();
	gridPoints->SetNumberOfPoints((xdim+6)*(ydim+6)*(zdim+6));
	//    gridPoints->SetNumberOfPoints(101*101*101);
	
	
	size_t u = 0;
	double x =bounds[0]-3*gridSpacing, y = bounds[2]-3*gridSpacing, z = bounds[4]-3*gridSpacing;
	for (int k = 0; k < zdim+6; k++) {
		for (int j = 0; j < ydim+6; j++) {
			for (int i = 0; i < xdim+6; i++) {
				gridPoints->SetPoint(u, x + center[0], y + center[1], z + center[2]);
				x += gridSpacing;
				u++;
			}
			y += gridSpacing;
			x = bounds[0] - 3*gridSpacing;
		}
		z += gridSpacing;
		y = bounds[2] - 3*gridSpacing;
	}
	
	grid->SetPoints(gridPoints);
	cout << "Grid construction done; " << u << " points ..."<< endl;
	
	vtkSelectEnclosedPoints* encloserOut = vtkSelectEnclosedPoints::New();
	encloserOut->SetInputData(grid);
	encloserOut->SetSurfaceData(gmsurf);
	encloserOut->CheckSurfaceOn();
	encloserOut->SetTolerance(0);
	cout << "Outside surface processing ..." << endl;
	encloserOut->Update();
	
	vtkDataArray* outLabel = encloserOut->GetOutput()->GetPointData()->GetArray("SelectedPoints");
	
	vtkSelectEnclosedPoints* encloserIn = vtkSelectEnclosedPoints::New();
	encloserIn->SetInputData(grid);
	encloserIn->SetSurfaceData(wmsurf);
	encloserIn->CheckSurfaceOn();
	encloserIn->InsideOutOn();
	encloserIn->SetTolerance(0);
	cout << "Inside surface processing ..." << endl;
	encloserIn->Update();
	
	vtkDataArray* inLabel = encloserIn->GetOutput()->GetPointData()->GetArray("SelectedPoints");
	
	vtkIntArray* inOutLabel = vtkIntArray::New();
	inOutLabel->SetNumberOfComponents(1);
	inOutLabel->SetNumberOfValues(inLabel->GetNumberOfTuples());
	
	size_t insideCount = 0;
	cout << "Computing the intersection ..." << endl;
	for (size_t j = 0; j < inOutLabel->GetNumberOfTuples(); j++) {
		inOutLabel->SetValue(j, (outLabel->GetTuple1(j) == 1 && inLabel->GetTuple1(j) == 1) ? 1 : 0);
		insideCount++;
	}
	
	inOutLabel->SetName("SelectedPoints");
	grid->GetPointData()->SetScalars(inOutLabel);
	
	insideCountOut = insideCount;
	
	return grid;
}


// create a structured grid with the size of input
// convert the grid to polydata
// create the intersection between the grid and the polydata
void runFillGrid(Options& opts, StringVector& args) {
	if (opts.GetBool("-humanBrain")) {
		string outputFile = args[2];
		
		vtkIO vio;
		vtkPolyData* osurf = vio.readFile(args[0]);
		vtkPolyData* isurf = vio.readFile(args[1]);
		
		size_t insideCountOut = 0;
		vtkDataSet* grid = createGrid(osurf, isurf, opts.GetStringAsInt("-dims", 100), insideCountOut);
		vio.writeFile(outputFile, grid);
		
		cout << "Inside Voxels: " << insideCountOut << endl;
	} else {
		cout << "Not supported yet!" << endl;
//		
//		string inputFile = args[0];
//		string outputFile = args[1];
//		
//		vtkIO vio;
//		vtkPolyData* input = vio.readFile(inputFile);
//		int insideCount = 0;
//		vtkDataSet* output = createGridForSphereLikeObject(input, insideCount, 100, false);
//		vio.writeFile(outputFile, output);
//		cout << "Inside Voxels: " << insideCount << endl;
	}
	
}


vtkIntArray* selectBoundaryCells(vtkDataSet* data, string scalarName) {
	const size_t nCells = data->GetNumberOfCells();
	vtkDataArray* array = data->GetPointData()->GetArray(scalarName.c_str());
	vector<int> scalarValue;
	vtkIntArray* intArray = vtkIntArray::New();
	intArray->SetNumberOfComponents(1);
	intArray->SetNumberOfTuples(nCells);
	
	for (size_t j = 0; j < nCells; j++) {
		vtkCell* cell = data->GetCell(j);
		scalarValue.clear();
		for (size_t k = 0; k < cell->GetNumberOfPoints(); k++) {
			vtkIdType cellPtId = cell->GetPointId(k);
			scalarValue.push_back(array->GetTuple1(cellPtId));
		}
		
		bool borderCell = false;
		if (scalarValue.size() > 0) {
			int value = scalarValue[0];
			for (size_t j = 1; j < scalarValue.size(); j++) {
				if (scalarValue[j] != value) {
					borderCell = true;
				}
			}
		}
		
		if (borderCell) {
			intArray->SetValue(j, 1);
		}
	}
	
	intArray->SetName("BorderCells");
	data->GetCellData()->AddArray(intArray);
	
	return intArray;
}


// mark the boundary points (2: interior boundary, 3: exterior boundary)
vtkDataArray* selectBoundaryPoints(vtkDataSet* data, std::string scalarName) {
	
	vtkDataArray* interiorMarker = data->GetPointData()->GetArray(scalarName.c_str());
	if (interiorMarker == NULL) {
		cout << "Can't find scalar values: " << scalarName << endl;
		return NULL;
	}
	
	const size_t npts = data->GetNumberOfPoints();
	
	vtkIntArray* boundaryMarker = vtkIntArray::New();
	boundaryMarker->SetNumberOfComponents(1);
	boundaryMarker->SetNumberOfTuples(npts);
	boundaryMarker->FillComponent(0, 0);
	
	
	unordered_set<vtkIdType> nbrs;
	vtkNew<vtkIdList> cellIds;
	
	for (size_t j = 0; j < npts; j++) {
		int jInteriorMark = interiorMarker->GetTuple1(j);
		
		// iterate over neighbor cells and find neighbor points
		nbrs.clear();
		cellIds->Reset();
		data->GetPointCells(j, cellIds.GetPointer());
		for (size_t k = 0; k < cellIds->GetNumberOfIds(); k++) {
			vtkCell* cell = data->GetCell(cellIds->GetId(k));
			for (size_t l = 0; l < cell->GetNumberOfEdges(); l++) {
				vtkCell* edge = cell->GetEdge(l);
				vtkIdType s = edge->GetPointId(0);
				vtkIdType e = edge->GetPointId(1);
				if (s == j) {
					nbrs.insert(e);
				} else if (e == j) {
					nbrs.insert(s);
				}
			}
		}
		
		// check neighbor points and find exterior points
		int surfaceInteriorExterior = jInteriorMark;
		unordered_set<vtkIdType>::iterator iter = nbrs.begin();
		for (; iter != nbrs.end(); iter++) {
			// if the neighbor is an exterior point
			vtkIdType nbrId = *iter;
			int nbrInteriorMark = interiorMarker->GetTuple1(nbrId);
			if (jInteriorMark == nbrInteriorMark) {
				continue;
			} else if (jInteriorMark == 0 && nbrInteriorMark == 1) {
				surfaceInteriorExterior = 10;
				break;
			} else if (jInteriorMark == 1 && nbrInteriorMark == 0) {
				// interior surface
				surfaceInteriorExterior = 11;
				break;
			} else {
				throw logic_error("invalid interior scalar makrer");
			}
		}
		
		boundaryMarker->SetTuple1(j, surfaceInteriorExterior);
	}
	
	boundaryMarker->SetName("BorderPoints");
	data->GetPointData()->AddArray(boundaryMarker);
	return boundaryMarker;
}

void runExtractStructuredGrid(Options& opts, StringVector& args) {
	string inputFile = args[0];
	string outputFile = args[1];
	
	vtkDataSet* inputPoly = vio.readDataFile(inputFile);
	double bounds[6];
	inputPoly->GetBounds(bounds);
	
	int extent[6];
	extent[0] = extent[2] = extent[4] = 0;
	extent[1] = (bounds[1]-bounds[0])/0.1;
	extent[3] = (bounds[3]-bounds[2])/0.1;
	extent[5] = (bounds[5]-bounds[4])/0.1;
	
	
	vtkNew<vtkExtractGrid> gridFilter;
	gridFilter->SetInputData(inputPoly);
	gridFilter->IncludeBoundaryOn();
	gridFilter->SetVOI(extent);
	gridFilter->SetSampleRate(1, 1, 1);
	gridFilter->Update();
	
	vio.writeFile(outputFile, gridFilter->GetOutput());
	
}


// Compute Laplace PDE based on the adjacency list and border
void computeLaplacePDE(vtkDataSet* data, const double low, const double high, const int nIters, const double dt, vtkPolyData* surfaceData = NULL) {
	
	if (data == NULL) {
		cout << "Data input is NULL" << endl;
		return;
	}
	
	class LaplaceGrid {
	public:
		double low;
		double high;
		double dt;
		vtkDataSet* dataSet;
		vtkPolyData* samplePoints;
		
		vector<vtkIdType> solutionDomain;
		vtkIntArray* boundaryCond;
		vtkPolyData* boundarySurface;
		
		vtkDoubleArray* solution;
		vtkDoubleArray* tmpSolution;
		vtkDataArray* laplaceGradient;
		vtkDoubleArray* laplaceGradientNormals;
		
		
		Geometry geom;
		Geometry::NeighborList nbrs;
		
		
		
		LaplaceGrid(double l, double h, double d, vtkDataSet* ds, vtkPolyData* pd= NULL): low(l), high(h), dt(d), dataSet(ds), boundarySurface(pd) {
			cout << "geometry edge extraction ... " << flush;
			geom.extractNeighbors(ds, nbrs);
			cout << " done " << endl;
			
			// check boundary points
			boundaryCond = vtkIntArray::SafeDownCast(ds->GetPointData()->GetArray("SampledValue"));
			if (boundaryCond == NULL) {
				throw runtime_error("No scalar values for BoundaryPoints");
			}
			
			initializeSolution();
		}
		
		void initializeSolution() {
			cout << "initializing solution grid ... " << flush;
			// low-value 2
			// high-value 1
			solution = vtkDoubleArray::New();
			solution->SetName("LaplacianSolution");
			solution->SetNumberOfComponents(1);
			solution->SetNumberOfTuples(boundaryCond->GetNumberOfTuples());
			solution->FillComponent(0, 0);
			
			tmpSolution = vtkDoubleArray::New();
			tmpSolution->SetName("LaplacianSolution");
			tmpSolution->SetNumberOfComponents(1);
			tmpSolution->SetNumberOfTuples(boundaryCond->GetNumberOfTuples());
			tmpSolution->FillComponent(0, 0);
			
			const size_t nPts = boundaryCond->GetNumberOfTuples();
			for (size_t j = 0; j < nPts; j++) {
				int domain = boundaryCond->GetValue(j);
				double uValue = 0;
				if (domain == 700) {
					// high
					uValue = high;
				} else if (domain == 300){
					// low
					uValue = low;
				} else if (domain == 1) {
					uValue = 0;
					solutionDomain.push_back(j);
				}
				solution->SetValue(j, uValue);
				tmpSolution->SetValue(j, uValue);
			}
			cout << "# of points: " << solutionDomain.size() << endl;
		}
		
		void computeStep() {
			const size_t nPts = solutionDomain.size();
			for (size_t j = 0; j < nPts; j++) {
				vtkIdType centerId = solutionDomain[j];
				Geometry::Neighbors& edgeMap = nbrs[centerId];
				Geometry::Neighbors::iterator iter = edgeMap.begin();
				
				double u = 0;
				double nNbrs = 0;
				for (; iter != edgeMap.end(); iter++) {
					const double du = solution->GetValue(*iter);
					u += du;
					nNbrs ++;

					//                    cout << iter->second.axisAligned << endl;
				}
				u = u / nNbrs;
				tmpSolution->SetValue(centerId, u);
			}
			
			vtkDoubleArray* swapTmp = tmpSolution;
			tmpSolution = solution;
			solution = swapTmp;
//			memcpy(solution->WritePointer(0, nPts), tmpSolution->GetVoidPointer(0), sizeof(double) * nTuples);
//			solution->DeepCopy(tmpSolution);
		}
		
		
		void computeNormals(vtkDataSet* data) {
			/*
			 vtkNew<vtkCellDerivatives> deriv;
			 deriv->SetInput(data);
			 deriv->SetVectorModeToComputeGradient();
			 deriv->Update();
			 vtkDataSet* derivOut = deriv->GetOutput();
			 derivOut->GetCellData()->SetActiveVectors("ScalarGradient");
			 vtkDataArray* scalarGradient = deriv->GetOutput()->GetCellData()->GetArray("ScalarGradient");
			 scalarGradient->SetName("LaplacianGradient");
			 */
			
			vtkNew<vtkGradientFilter> gradFilter;
			gradFilter->SetInputData(data);
			gradFilter->SetInputScalars(vtkDataSet::FIELD_ASSOCIATION_POINTS, "LaplacianSolution");
			gradFilter->SetResultArrayName("LaplacianGradient");
			gradFilter->Update();
			laplaceGradient = gradFilter->GetOutput()->GetPointData()->GetArray("LaplacianGradient");
			laplaceGradient->Register(NULL);
			
			laplaceGradientNormals = vtkDoubleArray::New();
			laplaceGradientNormals->SetName("LaplacianGradientNorm");
			laplaceGradientNormals->SetNumberOfComponents(3);
			laplaceGradientNormals->SetNumberOfTuples(laplaceGradient->GetNumberOfTuples());
			
			const size_t nPts = laplaceGradientNormals->GetNumberOfTuples();
			for (size_t j = 0; j < nPts; j++) {
				double* vec = laplaceGradient->GetTuple3(j);
				double norm = vtkMath::Norm(vec);
				if (norm > 1e-10) {
					laplaceGradientNormals->SetTuple3(j, vec[0]/norm, vec[1]/norm, vec[2]/norm);
				} else {
					laplaceGradientNormals->SetTuple3(j, 0, 0, 0);

				}
			}
			
			data->GetPointData()->AddArray(laplaceGradient);
			data->GetPointData()->SetVectors(laplaceGradientNormals);
		}
		
		void computeExteriorNormals(vtkPolyData* boundarySurface, const double radius = .1) {
			if (boundarySurface == NULL) {
				return;
			}
			vtkNew<vtkPolyDataNormals> normalsFilter;
			normalsFilter->SetInputData(boundarySurface);
			normalsFilter->ComputeCellNormalsOn();
			normalsFilter->ComputePointNormalsOn();
			normalsFilter->Update();
			vtkFloatArray* cellNormals = vtkFloatArray::SafeDownCast(normalsFilter->GetOutput()->GetCellData()->GetNormals());
			
			vtkNew<vtkCellLocator> cloc;
			cloc->SetDataSet(boundarySurface);
			cloc->AutomaticOn();
			cloc->BuildLocator();
			
			dataSet->GetPointData()->SetActiveScalars("SampledValue");
			
			vtkNew<vtkThresholdPoints> threshold;
			threshold->SetInputData(dataSet);
			threshold->ThresholdByUpper(250);
			threshold->Update();
			vtkDataSet* inoutBoundary = threshold->GetOutput();
			vtkIntArray* inoutBoundaryCond = vtkIntArray::SafeDownCast(inoutBoundary->GetPointData()->GetArray("SampledValue"));
			
			
			vtkNew<vtkPointLocator> ploc;
			ploc->SetDataSet(inoutBoundary);
			ploc->AutomaticOn();
			ploc->BuildLocator();
			
			const size_t nPts = dataSet->GetNumberOfPoints();
			for (size_t j = 0; j < nPts; j++) {
				int domain = boundaryCond->GetValue(j);
				if (domain == 700 || domain == 300 || domain == 0) {
					double x[3] = { 0, }, closestPoint[3] = { 0, };
					vtkIdType cellId = -1;
					int subId = 0;
					double dist2 = -1;
					dataSet->GetPoint(j, x);
					vtkNew<vtkGenericCell> closestCell;
					cloc->FindClosestPointWithinRadius(x, radius, closestPoint, cellId, subId, dist2);
					
					float cellNormal[3];
					cellNormals->GetTupleValue(cellId, cellNormal);
					cellNormal[0] = 0;
					vtkMath::Normalize(cellNormal);
					
					if (domain == 0) {
						vtkIdType xId = ploc->FindClosestPoint(x);
						domain = inoutBoundaryCond->GetValue(xId);
						assert(domain == 300 || domain == 700);
					}
					
					
					if (domain == 300) {
						laplaceGradientNormals->SetTuple3(j, -cellNormal[0], -cellNormal[1], -cellNormal[2]);
					} else {
						laplaceGradientNormals->SetTuple3(j, cellNormal[0], cellNormal[1], cellNormal[2]);
					}
				}
			}
		}
	};
	
	
	LaplaceGrid grid(low, high, dt, data, surfaceData);
	
	clock_t t1 = clock();
	
	// main iteration loop
	for (size_t i = 1; i <= nIters; i++) {
		if (i%500 == 0) {
			cout << "iteration: " << i << "\t";
			clock_t t2 = clock();
			cout << (double) (t2-t1) / CLOCKS_PER_SEC * 1000 << " ms;" << endl;
			t1 = t2;
		}
		grid.computeStep();
	}
	clock_t t2 = clock();
	cout << (double) (t2-t1) / CLOCKS_PER_SEC * 1000 << " ms;" << endl;
	
	
	// return the solution
	data->GetPointData()->AddArray(grid.solution);
	grid.computeNormals(data);
//	grid.computeExteriorNormals(surfaceData);
}



/// @brief perform a line clipping to fit within the object
bool performLineClipping(vtkPolyData* streamLines, vtkModifiedBSPTree* tree, int lineId, vtkCell* lineToClip, vtkPoints* outputPoints, vtkCellArray* outputLines, double &length) {
	
	/// - Iterate over all points in a line
	vtkIdList* ids = lineToClip->GetPointIds();
	
	
	/// - Identify a line segment included in the line
	int nIntersections = 0;
	bool foundEndpoint = false;
	std::vector<vtkIdType> idList;
	for (int j = 2; j < ids->GetNumberOfIds(); j++) {
		double p1[3], p2[3];
		streamLines->GetPoint(ids->GetId(j-1), p1);
		streamLines->GetPoint(ids->GetId(j), p2);
		
		// handle initial condition
		if (j == 2) {
			double p0[3];
			streamLines->GetPoint(ids->GetId(0), p0);
			idList.push_back(outputPoints->GetNumberOfPoints());
			outputPoints->InsertNextPoint(p0);
			
			idList.push_back(outputPoints->GetNumberOfPoints());
			outputPoints->InsertNextPoint(p1);
			
			length = sqrt(vtkMath::Distance2BetweenPoints(p0, p1));
		}
		
		int subId;
		double x[3] = {-1,-1,-1};
		double t = 0;
		
		double pcoords[3] = { -1, };
		int testLine = tree->IntersectWithLine(p1, p2, 0.01, t, x, pcoords, subId);
		if (testLine) {
			nIntersections ++;
			if (nIntersections > 0) {
				idList.push_back(outputPoints->GetNumberOfPoints());
				outputPoints->InsertNextPoint(x);
				length += sqrt(vtkMath::Distance2BetweenPoints(p1, x));
				foundEndpoint = true;
				break;
			}
		}
		//        cout << testLine << "; " << x[0] << "," << x[1] << "," << x[2] << endl;
		
		
		idList.push_back(outputPoints->GetNumberOfPoints());
		outputPoints->InsertNextPoint(p2);
		length += sqrt(vtkMath::Distance2BetweenPoints(p1, p2));
	}
	
	if (foundEndpoint) {
		outputLines->InsertNextCell(idList.size(), &idList[0]);
		return true;
	} else {
		outputLines->InsertNextCell(idList.size(), &idList[0]);
	}
	return false;
}


vtkPolyData* performStreamTracerPostProcessing(vtkPolyData* streamLines, vtkPolyData* seedPoints, vtkPolyData* destinationSurface) {
	
	const size_t nInputPoints = seedPoints->GetNumberOfPoints();
	
	// remove useless pointdata information
	streamLines->GetPointData()->Reset();
	streamLines->BuildCells();
	streamLines->BuildLinks();
	
	
	// loop over the cell and compute the length
	int nCells = streamLines->GetNumberOfCells();
	
	/// - Prepare the output as a scalar array
	//    vtkDataArray* streamLineLength = streamLines->GetCellData()->GetScalars("Length");
	
	/// - Prepare the output for the input points
	vtkDoubleArray* streamLineLengthPerPoint = vtkDoubleArray::New();
	streamLineLengthPerPoint->SetNumberOfTuples(nInputPoints);
	streamLineLengthPerPoint->SetName("Length");
	streamLineLengthPerPoint->SetNumberOfComponents(1);
	streamLineLengthPerPoint->FillComponent(0, 0);
	
	vtkIntArray* lineCorrect = vtkIntArray::New();
	lineCorrect->SetName("LineOK");
	lineCorrect->SetNumberOfValues(nInputPoints);
	lineCorrect->FillComponent(0, 0);
	
	seedPoints->GetPointData()->SetScalars(streamLineLengthPerPoint);
	seedPoints->GetPointData()->AddArray(lineCorrect);
	
	cout << "Assigning length to each source vertex ..." << endl;
	vtkDataArray* seedIds = streamLines->GetCellData()->GetScalars("SeedIds");
	if (seedIds) {
		// line clipping
		vtkPoints* outputPoints = vtkPoints::New();
		vtkCellArray* outputCells = vtkCellArray::New();
		
		/// construct a tree locator
		vtkModifiedBSPTree* tree = vtkModifiedBSPTree::New();
		tree->SetDataSet(destinationSurface);
		tree->BuildLocator();
		
		vtkDoubleArray* lengthArray = vtkDoubleArray::New();
		lengthArray->SetName("Length");
		
		vtkIntArray* pointIds = vtkIntArray::New();
		pointIds->SetName("PointIds");
		
		cout << "# of cells: " << nCells << endl;

		int noLines = 0;
		for (int i = 0; i < nCells; i++) {
			int pid = seedIds->GetTuple1(i);
			double length = 0;
			if (pid > -1) {
				vtkCell* line = streamLines->GetCell(i);
				/// - Assume that a line starts from a point on the input mesh and must meet at the opposite surface of the starting point.
				bool lineAdded = performLineClipping(streamLines, tree, i, line, outputPoints, outputCells, length);
				
				if (lineAdded) {
					pointIds->InsertNextValue(pid);
					lengthArray->InsertNextValue(length);
					streamLineLengthPerPoint->SetValue(pid, length);
					lineCorrect->SetValue(pid, 1);
				} else {
					pointIds->InsertNextValue(pid);
					lengthArray->InsertNextValue(length);
					streamLineLengthPerPoint->SetValue(pid, length);
					lineCorrect->SetValue(pid, 2);
					noLines++;
				}
			}
		}
		
		cout << "# of clipping failure: " << noLines << endl;
		
		vtkPolyData* outputStreamLines = vtkPolyData::New();
		outputStreamLines->SetPoints(outputPoints);
		outputStreamLines->SetLines(outputCells);
		outputStreamLines->GetCellData()->AddArray(pointIds);
		outputStreamLines->GetCellData()->AddArray(lengthArray);
	
		
		vtkCleanPolyData* cleaner = vtkCleanPolyData::New();
		cleaner->SetInputData(outputStreamLines);
		cleaner->Update();
		
		return cleaner->GetOutput();
	} else {
		cout << "Can't find SeedIds" << endl;
		return NULL;
	}
}


vtkPolyData* performStreamTracer(Options& opts, vtkDataSet* inputData, vtkPolyData* inputSeedPoints, vtkPolyData* destSurf, bool zRotate = false) {
    if (inputData == NULL || inputSeedPoints == NULL) {
        cout << "input vector field or seed points is null!" << endl;
        return NULL;
    }
    
    if (destSurf == NULL) {
        cout << "trace destination surface is null" << endl;
        return NULL;
    }
    
	// set active velocity field
	inputData->GetPointData()->SetActiveVectors("LaplacianGradientNorm");
	
	/// - Converting the input points to the image coordinate
	vtkPoints* points = inputSeedPoints->GetPoints();
	cout << "# of seed points: " << points->GetNumberOfPoints() << endl;
	const int nInputPoints = inputSeedPoints->GetNumberOfPoints();
	if (zRotate) {
		for (int i = 0; i < nInputPoints; i++) {
			double p[3];
			points->GetPoint(i, p);
			// FixMe: Do not use a specific scaling factor
			if (zRotate) {
				p[0] = -p[0];
				p[1] = -p[1];
				p[2] = p[2];
			}
			points->SetPoint(i, p);
		}
		inputSeedPoints->SetPoints(points);
	}
	
	
	/// StreamTracer should have a point-wise gradient field
	/// - Set up tracer (Use RK45, both direction, initial step 0.05, maximum propagation 500
	vtkStreamTracer* tracer = vtkStreamTracer::New();
	tracer->SetInputData(inputData);
	tracer->SetSourceData(inputSeedPoints);
	tracer->SetComputeVorticity(false);

	string traceDirection = opts.GetString("-traceDirection", "forward");
	if (traceDirection == "both") {
		tracer->SetIntegrationDirectionToBoth();
	} else if (traceDirection == "backward") {
		tracer->SetIntegrationDirectionToBackward();
		cout << "Backward Integration" << endl;
	} else if (traceDirection == "forward") {
		tracer->SetIntegrationDirectionToForward();
		cout << "Forward Integration" << endl;
	}

//    tracer->SetInterpolatorTypeToDataSetPointLocator();
	tracer->SetInterpolatorTypeToCellLocator();
	tracer->SetMaximumPropagation(5000);
	tracer->SetInitialIntegrationStep(0.01);
//    tracer->SetMaximumIntegrationStep(0.1);
    tracer->SetIntegratorTypeToRungeKutta45();
//    tracer->SetIntegratorTypeToRungeKutta2();

    cout << "Integration Direction: " << tracer->GetIntegrationDirection() << endl;
    cout << "Initial Integration Step: " << tracer->GetInitialIntegrationStep() << endl;
    cout << "Maximum Integration Step: " << tracer->GetMaximumIntegrationStep() << endl;
    cout << "Minimum Integration Step: " << tracer->GetMinimumIntegrationStep() << endl;
    cout << "Maximum Error: " << tracer->GetMaximumError() << endl;
    cout << "IntegratorType: " << tracer->GetIntegratorType() << endl;

    
    tracer->Update();

	
	vtkPolyData* streamLines = tracer->GetOutput();
//	streamLines->Print(cout);
	
	vio.writeFile("streamlines.vtp", streamLines);
	
	return performStreamTracerPostProcessing(streamLines, inputSeedPoints, destSurf);
}







vtkDataSet* sampleSurfaceScalarsForGrid(vtkDataSet* ds, vtkDataSet* pd, string scalarName) {
	if (ds == NULL || pd == NULL) {
		throw runtime_error("Input is NULL");
	}
	vtkDataArray* scalars = NULL;
	if (scalarName == "") {
		scalars = pd->GetPointData()->GetScalars();
	} else {
		scalars = pd->GetPointData()->GetArray(scalarName.c_str());
	}
	if (scalars == NULL) {
		throw logic_error("No scalar available!");
	}
	
	
	vtkIntArray* sampledValue = vtkIntArray::New();
	sampledValue->SetName("SampledValue");
	sampledValue->SetNumberOfComponents(1);
	sampledValue->SetNumberOfTuples(ds->GetNumberOfPoints());
	
	vtkNew<vtkCellLocator> cloc;
	cloc->SetDataSet(pd);
	cloc->AutomaticOn();
	cloc->BuildLocator();
	
	
	
	vtkDataArray* insideOut = ds->GetPointData()->GetArray("BorderPoints");
	
	vtkNew<vtkIntArray> closestCell;
	closestCell->SetName("ClosestCell");
	closestCell->SetNumberOfComponents(1);
	closestCell->SetNumberOfValues(insideOut->GetNumberOfTuples());
	ds->GetPointData()->AddArray(closestCell.GetPointer());
	
	vtkNew<vtkGenericCell> genCell;
	for (size_t j = 0; j < insideOut->GetNumberOfTuples(); j++) {
		int jInsideOut = insideOut->GetTuple1(j);
		if (jInsideOut < 10) {
			closestCell->SetValue(j, -1);
			sampledValue->SetValue(j, jInsideOut);
			continue;
		} else if (jInsideOut == 11) {
			sampledValue->SetValue(j, 1);
			continue;
		}
		
		double jPt[3], x[3] = {0,};
		ds->GetPoint(j, jPt);
		
		vtkIdType cellId = -1;
		int subId = -1;
		double dist2 = -1;
		
		cloc->FindClosestPoint(jPt, x, cellId, subId, dist2);
		closestCell->SetValue(j, cellId);
		
		if (cellId == -1) {
			throw runtime_error("Can't find a closest cell");
		}
		vtkCell* cell = pd->GetCell(cellId);
		int values[3] = { 0, };
		for (size_t k = 0; k < cell->GetNumberOfPoints(); k++) {
			vtkIdType kId = cell->GetPointId(k);
			int scalarValue = scalars->GetTuple1(kId);
			values[scalarValue] ++;
		}
		if (j == 7970 || j == 8076 || j == 8182 || j == 8183 || j == 8289 || j == 8396) {
			cout << values[0] << "," << values[1] << "," << values[2] << endl;
			sampledValue->SetValue(j, 300);
		} else if (values[1] == 0 && values[2] > 0) {
			sampledValue->SetValue(j, 300);
		} else if (values[2] == 0 && values[1] > 0) {
			sampledValue->SetValue(j, 700);
		} else {
			sampledValue->SetValue(j, 300);
		}
		//        cout << j << ": " <<  sampledValue->GetValue(j) << endl;
	}
	ds->GetPointData()->AddArray(sampledValue);
	
	Geometry geom;
	Geometry::EdgeList edges;
	geom.extractEdges(ds, edges);
	
	const size_t nPoints = edges.size();
	for (size_t j = 0; j < nPoints; j++) {
		Geometry::EdgeMap::iterator iter = edges[j].begin();
		for (; iter != edges[j].end(); iter++) {
			if (iter->second.u > iter->second.v) {
				continue;
			}
			vtkIdType uId = iter->second.u;
			vtkIdType vId = iter->second.v;
			int usv = insideOut->GetTuple1(uId);
			int vsv = insideOut->GetTuple1(vId);
			if ((usv == 0 && vsv == 1) || (usv == 1 && vsv == 0)) {
				throw logic_error("illegal boundary!");
			}
		}
	}
	return ds;
	
	
	
	vtkNew<vtkIdList> cells;
	
	//    const size_t nPoints = edges.size();
	for (size_t j = 0; j < nPoints; j++) {
		Geometry::EdgeMap::iterator iter = edges[j].begin();
		for (; iter != edges[j].end(); iter++) {
			double uPt[3], vPt[3];
			if (iter->second.u > iter->second.v) {
				continue;
			}
			
			vtkIdType uId = iter->second.u;
			vtkIdType vId = iter->second.v;
			
			ds->GetPoint(uId, uPt);
			ds->GetPoint(vId, vPt);
			
			const int uIn = insideOut->GetTuple1(uId);
			const int vIn = insideOut->GetTuple1(vId);
			if (uIn == vIn) {
				continue;
			}
			
			vtkIdType insidePt = uIn > 0 ? iter->second.u : iter->second.v;
			vtkIdType outsidePt = uIn > 0 ? iter->second.v : iter->second.u;
			
			cells->Reset();
			cloc->FindCellsAlongLine(uPt, vPt, 0, cells.GetPointer());
			if (cells->GetNumberOfIds() > 1) {
				cout << uId << "," << vId << "," << cells->GetNumberOfIds() << " ";
				for (size_t k = 0; k < cells->GetNumberOfIds(); k++) {
					cout << cells->GetId(k) << " ";
				}
				cout << endl;
			}
			for (size_t k = 0; k < cells->GetNumberOfIds(); k++) {
				vtkIdType kId = cells->GetId(k);
				vtkCell* cell = pd->GetCell(kId);
				int values[3] = { 0, };
				for (size_t l = 0; l < cell->GetNumberOfPoints(); l++) {
					int scalar = scalars->GetTuple1(cell->GetPointId(l));
					values[scalar]++;
				}
				if (values[1] == 0 || values[2] == 0) {
					int scalar = values[1] == 0 ? 2 : 1;
					sampledValue->SetValue(outsidePt, scalar);
				} else {
					sampledValue->SetValue(outsidePt, 3);
				}
			}
		}
	}
	return ds;
}


void runMeasureThicknessX(Options& opts, StringVector& args) {
	vtkIO vio;
	string inputFile = args[0];
	string outputStreamFile = args[1];
	string scalarName = opts.GetString("-scalarName", "meanLabels");
	string outputName = opts.GetString("-o", "");
	
	vtkPolyData* input = vio.readFile(inputFile);
	vtkPolyData* destSurf = NULL;
	//    vtkPolyData* inputSeedPoints = vio.readFile(inputSeedFile);
	
	//    vtkNew<vtkThresholdPoints> selector;
	//    selector->SetInput(inputSeedPoints);
	//    selector->ThresholdBetween(0.5, 1.5);
	//    selector->SetInputArrayToProcess(0, 0, 0, vtkDataSet::FIELD_ASSOCIATION_POINTS, "meanLabels");
	//    selector->Update();
	//    vtkPolyData* selectedSeeds = selector->GetOutput();
	
	//    cout << selectedSeeds->GetNumberOfPoints() << endl;
	
	int insideCount = 0;
	vtkDataSet* inOutGrid = createGridForSphereLikeObject(input, insideCount);
	cout << "Grid created..." << endl;
	selectBoundaryPoints(inOutGrid, "SelectedPoints");
	cout << "Boundary identified..." << endl;
	vtkDataSet* boundaryCondGrid = sampleSurfaceScalarsForGrid(inOutGrid, input, scalarName);
	cout << "Boundary condition assigned..." << endl;
	vio.writeFile("BoundaryCondGrid.vts", boundaryCondGrid);
	computeLaplacePDE(boundaryCondGrid, 0, 10000, 5000, 0.065, input);
	cout << "Laplace PDE computation done..." << endl;
	vio.writeFile("LaplaceSolution.vts", boundaryCondGrid);

	vtkPolyData* outputStream = performStreamTracer(opts, boundaryCondGrid, input, destSurf);
	cout << "RK4 integration done..." << endl;
	vio.writeFile(outputStreamFile, outputStream);
	if (outputName != "") {
		vio.writeFile(outputName, input);
	}
}

/// @brief Execute the stream tracer
void runStreamTracer(Options& opts, StringVector& args) {
	string inputVTUFile = args[0];
	string inputSeedPointsFile = args[1];
	string destSurfFile = args[2];
	
	string outputStreamFile = args[3];
	string outputPointFile = args[4];
	bool zRotate = opts.GetBool("-zrotate", false);
	
	vtkIO vio;
	vtkDataSet* inputData = vio.readDataFile(inputVTUFile);
	inputData->GetPointData()->SetActiveVectors("LaplacianGradientNorm");
	vtkPolyData* inputSeedPoints = vio.readFile(inputSeedPointsFile);
	vtkPolyData* destSurf = vio.readFile(destSurfFile);
	
	vtkPolyData* outputStream = performStreamTracer(opts, inputData, inputSeedPoints, destSurf, zRotate);
	
	vio.writeFile(outputPointFile, inputSeedPoints);
	if (outputStream) {
		vio.writeFile(outputStreamFile, outputStream);
	}
}


/// @brief Create a sphere enclosing a given object
void runEnclosingSphere(Options& opts, string inputObj, string outputObj, bool normalize, double diff = 0, int phiRes = 6, int thetaRes = 6) {
	vtkIO vio;
	vtkDataSet* dataSet = vio.readDataFile(inputObj);
	dataSet->ComputeBounds();
	double* bbox = dataSet->GetBounds();
	
	double radius = sqrt((bbox[1]-bbox[0])*(bbox[1]-bbox[0]) + (bbox[3]-bbox[2])*(bbox[3]-bbox[2]) + (bbox[5]-bbox[4])*(bbox[5]-bbox[4]))/2.0;
	
	double center[3] = {0,};
	center[0] = (bbox[1]+bbox[0])/2.0;
	center[1] = (bbox[3]+bbox[2])/2.0;
	center[2] = (bbox[5]+bbox[4])/2.0;
	
	cout << "Radius: " << radius << endl;
	
	vtkSphereSource* sphereSource = vtkSphereSource::New();
	sphereSource->SetRadius(radius-diff);
	sphereSource->SetPhiResolution(phiRes);
	sphereSource->SetThetaResolution(thetaRes);
	sphereSource->Update();
	
	vtkPolyData* outputSphere = sphereSource->GetOutput();
	
	vtkTransform* txf = vtkTransform::New();
	txf->Translate(center);
	txf->RotateX(90);
	
	vtkTransformPolyDataFilter* filter = vtkTransformPolyDataFilter::New();
	filter->SetTransform(txf);
	filter->SetInputData(outputSphere);
	filter->Update();
	
	
	
	vio.writeFile(outputObj, filter->GetOutput());
}


void runExtractSlice(Options& opts, StringVector& args) {
	vtkIO vio;
	
	vtkDataSet* ds = vio.readDataFile(args[0]);
	vtkStructuredGrid* sg = vtkStructuredGrid::SafeDownCast(ds);
	
	if (sg == NULL) {
		vtkImageData* img = vtkImageData::SafeDownCast(ds);
		if (img == NULL) {
			cout << "input must be either a structured grid or an image data" << endl;
			return;
		}
		vtkNew<vtkImageToStructuredGrid> imgconv;
		imgconv->SetInputData(img);
		imgconv->Update();
		sg = imgconv->GetOutput();
		sg->Register(NULL);
	}

	int extent[6];
	sg->GetExtent(extent);

	if (opts.HasString("-voi")) {
		
	}
	if (opts.HasString("-z")) {
		int z = opts.GetStringAsInt("-z", 0);
		extent[4] = extent[5] = z;
	}
	
	if (opts.HasString("-x")) {
		int x = opts.GetStringAsInt("-x", 0);
		extent[0] = extent[1] = x;
	}

	if (opts.HasString("-y")) {
		int y = opts.GetStringAsInt("-y", 0);
		extent[2] = extent[3] = y;
	}

	cout << "Extent: " << extent[0] << "," << extent[1] << ",";
	cout << extent[2] << "," << extent[3] << ",";
	cout << extent[4] << "," << extent[5] << endl;
	
	vtkNew<vtkExtractGrid> ex;
	ex->SetInputData(sg);
	ex->SetVOI(extent);
	ex->Update();
	
	vtkStructuredGrid* sgOut = ex->GetOutput();
	vio.writeFile(args[1], sgOut);
}

void runPrintTraceCorrespondence(Options& opts, string inputMeshName, string inputStreamName, string outputWarpedMeshName, vtkPolyData* srcmesh = NULL) {
	vtkIO vio;
	
	if (srcmesh == NULL) {
		srcmesh = vio.readFile(inputMeshName);
	}
	
	vtkNew<vtkPolyData> warpedMesh;
	warpedMesh->DeepCopy(srcmesh);
	
	srcmesh->ComputeBounds();
	
	double center[3];
	srcmesh->GetCenter(center);
	
	vtkDataSet* strmesh = vio.readDataFile(inputStreamName);
	
	int traceDirection = StreamTracer::FORWARD;
	string dir = opts.GetString("-traceDirection", "forward");
	if (dir == "backward") {
		traceDirection = StreamTracer::BACKWARD;
	} else if (dir == "both") {
		traceDirection = StreamTracer::BOTH;
	}
	
	
	vtkNew<vtkDoubleArray> pointArr;
	pointArr->SetName("SourcePoints");
	pointArr->SetNumberOfComponents(3);
	pointArr->SetNumberOfTuples(srcmesh->GetNumberOfPoints());
	
	vtkNew<vtkDoubleArray> sphrCoord;
	sphrCoord->SetName("SphericalCoordinates");
	sphrCoord->SetNumberOfComponents(3);
	sphrCoord->SetNumberOfTuples(srcmesh->GetNumberOfPoints());
	
	vtkNew<vtkSphericalTransform> sphTxf;
	sphTxf->Inverse();
	
	
	vtkNew<vtkDoubleArray> destPointArr;
	destPointArr->SetName("DestinationPoints");
	destPointArr->SetNumberOfComponents(3);
	destPointArr->SetNumberOfTuples(srcmesh->GetNumberOfPoints());
	
	vtkNew<vtkDoubleArray> sphereRadiusArr;
	sphereRadiusArr->SetName("SphereRadius");
	sphereRadiusArr->SetNumberOfComponents(1);
	sphereRadiusArr->SetNumberOfTuples(srcmesh->GetNumberOfPoints());
	
	vtkNew<vtkPoints> warpedPoints;
	warpedPoints->DeepCopy(srcmesh->GetPoints());
	
	vtkNew<vtkPointLocator> ploc;
	ploc->SetDataSet(srcmesh);
	ploc->SetTolerance(0);
	ploc->BuildLocator();
	
	
	
	const size_t nCells = strmesh->GetNumberOfCells();
	vtkDataArray* seedIds = strmesh->GetCellData()->GetArray("PointIds");
	
	
	for (size_t j = 0; j < nCells; j++) {
		vtkCell* cell = strmesh->GetCell(j);
		const size_t nPts = cell->GetNumberOfPoints();
		if (nPts < 2) {
			continue;
		}
		vtkIdType s = cell->GetPointId(0);
		vtkIdType e = cell->GetPointId(nPts-1);
		
		double qs[3], qe[3], pj[3], spj[3], npj[3];
		strmesh->GetPoint(s, qs);
		strmesh->GetPoint(e, qe);
		
		vtkIdType seedId = (vtkIdType) seedIds->GetTuple1(j);
		warpedPoints->SetPoint(seedId, qe);
		
		vtkMath::Subtract(qe, center, npj);
		double warpedPointNorm = vtkMath::Norm(npj);
		
		sphereRadiusArr->SetValue(j, warpedPointNorm);

//		cout << "cell " << j << ": " << nPts << ", " << s << " => " << e << endl;
//		cout << "cell " << j << ": " << qe[0] << "," << qe[1] << "," << qe[2] << endl;
		
//
//		srcmesh->GetPoint(seedId, pj);
//		pointArr->SetTupleValue(seedId, pj);
//		
//		vtkMath::Subtract(pj, sphereCenter, npj);
//		sphTxf->TransformPoint(npj, spj);
//		sphrCoord->SetTupleValue(seedId, spj);
//		
//		destPointArr->SetTupleValue(seedId, qe);

	}
	
	srcmesh->GetPointData()->AddArray(sphereRadiusArr.GetPointer());

	struct InterpolateBrokenPoints {
		InterpolateBrokenPoints(vtkPolyData* surf, vtkPoints* warpedPoints, vtkDataArray* seedIds) {
			// identify broken points
			vector<vtkIdType> brokenPoints;
			vtkIdType z = 0;
			for (size_t j = 0; j < seedIds->GetNumberOfTuples(); j++,z++) {
				vtkIdType y = seedIds->GetTuple1(j);
				while (z < y) {
					brokenPoints.push_back(z++);
				}
			}
			
			// find neighbors and compute interpolatead points
			vtkNew<vtkIdList> cellIds;
			set<vtkIdType> nbrs;
			for (size_t j = 0; j < brokenPoints.size(); j++) {
				vtkIdType pid = brokenPoints[j];
				cellIds->Reset();
				surf->GetPointCells(pid, cellIds.GetPointer());
				nbrs.clear();
				// find neighbor points
				for (size_t k = 0; k < cellIds->GetNumberOfIds(); k++) {
					vtkCell* cell = surf->GetCell(k);
					findNeighborPoints(cell, pid, nbrs);
				}
				// average neighbor points
				double p[3] = {0,}, q[3] = {0,};
				set<vtkIdType>::iterator it = nbrs.begin();
				for (; it != nbrs.end(); it++) {
					if (find(brokenPoints.begin(), brokenPoints.end(), *it) == brokenPoints.end()) {
						warpedPoints->GetPoint(*it, q);
						vtkMath::Add(p, q, p);
					} else {
						cout << "broken neighbor!! " << pid << "," << *it << endl;
					}
				}
				p[0]/=nbrs.size();
				p[1]/=nbrs.size();
				p[2]/=nbrs.size();
				warpedPoints->SetPoint(pid, p);
			}
		}

	};
	
	warpedMesh->SetPoints(warpedPoints.GetPointer());
	InterpolateBrokenPoints(warpedMesh.GetPointer(), warpedPoints.GetPointer(), seedIds);
	
	//	warpedMesh->Print(cout);
//	warpedMesh->GetPointData()->SetVectors(pointArr.GetPointer());
//	warpedMesh->GetPointData()->AddArray(sphrCoord.GetPointer());
	vio.writeFile(outputWarpedMeshName, warpedMesh.GetPointer());
	
//	if (args.size() > 3) {
//		srcmesh->GetPointData()->SetVectors(destPointArr.GetPointer());
//		srcmesh->GetPointData()->AddArray(sphrCoord.GetPointer());
//		vio.writeFile(args[3], srcmesh);
//	}
}


void runWarpMesh(Options& opts, StringVector& args) {
	vtkIO vio;
	vtkPolyData* inputMesh = vio.readFile(args[0]);
	vtkPolyData* inputStreams = vio.readFile(args[1]);
	
	const size_t nCells = inputStreams->GetNumberOfCells();
	vtkDataArray* seedArr = inputStreams->GetCellData()->GetArray("SeedIds");
	
	struct SplitCurve {
		vtkPolyData* mesh;
		vtkPolyLine* curve;
		size_t nPoints;
		vector<size_t> sumLength;
		vector<vtkPoints*> pointCollection;
		
		SplitCurve(vtkPolyData* m): mesh(m) {
			nPoints = mesh->GetNumberOfCells();
			pointCollection.resize(11);
			for (size_t j = 0; j < 11; j++) {
				pointCollection[j] = vtkPoints::New();
				pointCollection[j]->SetNumberOfPoints(nPoints);
			}
		}
		
		void split(vtkCell* cell, vtkIdType cellId) {
			vtkPolyLine* curve = vtkPolyLine::SafeDownCast(cell);
			const size_t nPoints = curve->GetNumberOfPoints();
			double sum = 0;
			vector<double> segments;
			for (size_t j = 1; j < nPoints; j++) {
				double p[3], q[3];
				mesh->GetPoint(curve->GetPointId(j-1), p);
				mesh->GetPoint(curve->GetPointId(j), q);
				double d = sqrt(vtkMath::Distance2BetweenPoints(p, q));
				sum += d;
				segments.push_back(d);
			}
			
			
			double newsegLength = sum / 10.0;
			double newsegSum = 0;
			size_t counter = 0;
			double p[3];
			for (size_t j = 0; j < segments.size(); j++) {
				newsegSum += segments[j];
				if (newsegSum >= newsegLength) {
					// find a point between j-1 and j
					mesh->GetPoint(curve->GetPointId(j+1), p);
					pointCollection[counter++]->SetPoint(cellId, p);
					newsegSum = 0;
				}
			}
			for (; counter < 11; counter++) {
				pointCollection[counter]->SetPoint(cellId, p);
			}
		}
	};
	
	
	SplitCurve sc(inputStreams);
	for (size_t j = 0; j < nCells; j++) {
		sc.split(inputStreams->GetCell(j), j);
	}

	vtkPoints* inputPoints = inputMesh->GetPoints();
	for (size_t j = 0; j < 11; j++) {
		char fname[256];
		for (size_t k = 0; k < inputPoints->GetNumberOfPoints(); k++) {
			inputPoints->SetPoint(k, sc.pointCollection[j]->GetPoint(k));
		}
		snprintf(fname, 256, "%s-%02lu.vtp", args[2].c_str(), j);
		vio.writeFile(fname, inputMesh);
	}
}

void runSphericalMapping(Options& opts, StringVector& args) {
	// runEnclosingSphere
	string inputObj = args[0];

	string outputGrid = args[1] + "_grid.vts";
	string outputField = args[1] + "_field.vts";
	string outputSphere = args[1] + "_sphere.vtp";
	string outputStream = args[1] + "_stream.vtp";
	string outputMesh = args[1] + "_warpedMesh.vtp";
	string outputObj = args[1] + "_object.vtp";
	
    string inputFieldFile = opts.GetString("-inputField");
    
    
    vtkDataSet* laplaceField = NULL;
    
    if (opts.HasString("-inputField")) {
        cout << "Reading " << inputFieldFile << flush;
        laplaceField = vio.readDataFile(inputFieldFile);
        cout << " done." << endl;
    } else {
        // create outer sphere
        int phiRes = opts.GetStringAsInt("-phi", 32);
        int thetaRes = opts.GetStringAsInt("-theta", 32);
        
        runEnclosingSphere(opts, inputObj, outputSphere, false, 0, phiRes, thetaRes);
        
        // compute grid
        StringVector fillGridArgs;
        fillGridArgs.push_back(outputSphere);
        fillGridArgs.push_back(inputObj);
        fillGridArgs.push_back(outputGrid);
        opts.SetBool("-humanBrain", true);
        runFillGrid(opts, fillGridArgs);
        
        // compute laplace map
        vtkDataSet* laplaceField = vio.readDataFile(outputGrid);
        
        const double dt = opts.GetStringAsReal("-dt", 0.125);
        const int numIter = opts.GetStringAsInt("-iter", 10000);
        
        computeLaplacePDE(laplaceField, 0, 10000, numIter, dt);
        vio.writeFile(outputField, laplaceField);
    }
	
	
	
	vtkIO vio;
	vtkPolyData* inputData = vio.readFile(inputObj);
//	vtkDataSet* laplaceField = vio.readDataFile(inputField);
    vtkPolyData* sphere = NULL;
    
    if (opts.HasString("-inputSphere")) {
        sphere = vio.readFile(opts.GetString("-inputSphere"));
    } else {
        sphere = vio.readFile(outputSphere);
    }

	if (opts.GetString("-traceDirection") == "backward") {
		vtkPolyData* streams = performStreamTracer(opts, laplaceField, sphere, inputData);
		vio.writeFile(outputStream, streams);
		runPrintTraceCorrespondence(opts, outputSphere, outputStream, outputMesh, sphere);
	} else {
		vtkPolyData* streams = performStreamTracer(opts, laplaceField, inputData, sphere);
		vio.writeFile(outputStream, streams);
		runPrintTraceCorrespondence(opts, inputObj, outputStream, outputMesh, inputData);
	}
	

	vio.writeFile(outputObj, inputData);
	
}

void runSurfaceCorrespondence(Options& opts, StringVector& args) {
	// runEnclosingSphere
	string inputObj1 = args[0];
    string inputObj2 = args[1];
    string prefix = args[2];

    if (inputObj1 == "" || inputObj2 == "") {
        cout << "-surfaceCorrespondence option needs two inputs" << endl;
    }
    if (prefix == "") {
        prefix = "surface_correspondence";
    }

	string outputGrid = prefix + "_grid.vts";
	string outputField = prefix + "_field.vts";
	string outputStream = prefix + "_stream.vtp";
	string outputMesh = prefix + "_warpedMesh.vtp";
	string outputObj = prefix + "_object.vtp";

    cout << "Output grid: " << outputGrid << endl;
    cout << "Output laplacian field: " << outputField << endl;
    cout << "Output streamlines: " << outputStream << endl;
    cout << "Output warped mesh: " << outputMesh << endl;
	
    string inputFieldFile = opts.GetString("-inputField");
    
    
    vtkDataSet* laplaceField = NULL;
    
    if (opts.HasString("-inputField")) {
        cout << "Reading " << inputFieldFile << flush;
        laplaceField = vio.readDataFile(inputFieldFile);
        cout << " done." << endl;
    } else {
        // create uniform grid for a FDM model
        StringVector fillGridArgs;
        fillGridArgs.push_back(inputObj2);
        fillGridArgs.push_back(inputObj1);
        fillGridArgs.push_back(outputGrid);
        opts.SetBool("-humanBrain", true);
        runFillGrid(opts, fillGridArgs);
        
        // compute laplace map
        laplaceField = vio.readDataFile(outputGrid);
        
        const double dt = opts.GetStringAsReal("-dt", 0.125);
        const int numIter = opts.GetStringAsInt("-iter", 10000);
        
        computeLaplacePDE(laplaceField, 0, 10000, numIter, dt);
        vio.writeFile(outputField, laplaceField);
    }
	
	
	
	vtkIO vio;
	vtkPolyData* inputData = vio.readFile(inputObj1);
    vtkPolyData* inputData2 = vio.readFile(inputObj2);

    if (inputData == NULL) {
        cout << inputObj1 << " is null" << endl;
        return;
    }
    if (inputData2 == NULL) {
        cout << inputObj2 << " is null" << endl;
        return;
    }

	if (opts.GetString("-traceDirection") == "backward") {
		vtkPolyData* streams = performStreamTracer(opts, laplaceField, inputData2, inputData);
		vio.writeFile(outputStream, streams);
		runPrintTraceCorrespondence(opts, inputObj2, outputStream, outputMesh, inputData2);
	} else {
		vtkPolyData* streams = performStreamTracer(opts, laplaceField, inputData, inputData2);
		vio.writeFile(outputStream, streams);
		runPrintTraceCorrespondence(opts, inputObj1, outputStream, outputMesh, inputData);
	}
	

	vio.writeFile(outputObj, inputData);
}


// run measureThickness option
void runMeasureThickness(Options& opts, StringVector& args) {
    // runEnclosingSphere
    string inputOuterObj = args[0];
    string inputInnerObj = args[1];
    
    string outputPrefix = args[2];
    string outputGrid = outputPrefix + "_grid.vts";
    string outputField = outputPrefix + "_field.vts";
    string outputStream = outputPrefix + "_stream.vtp";
    string outputMesh = outputPrefix + "_warpedMesh.vtp";
    string outputObj = outputPrefix + "_object.vtp";
    
    vtkDataSet* laplaceField = NULL;
    
    if (opts.HasString("-inputField")) {
        string inputFieldFile = opts.GetString("-inputField");
        cout << "Reading " << inputFieldFile << flush;
        laplaceField = vio.readDataFile(inputFieldFile);
        cout << " done." << endl;
    } else if (opts.GetBool("-humanBrain")) {
        // compute grid
        StringVector fillGridArgs;
        fillGridArgs.push_back(inputOuterObj);
        fillGridArgs.push_back(inputInnerObj);
        fillGridArgs.push_back(outputGrid);

        // run fill grid
        runFillGrid(opts, fillGridArgs);
        
        // compute laplace map
        laplaceField = vio.readDataFile(outputGrid);
        
        const double dt = opts.GetStringAsReal("-dt", 0.125);
        const int numIter = opts.GetStringAsInt("-iter", 10000);
        
        computeLaplacePDE(laplaceField, 0, 10000, numIter, dt);
        vio.writeFile(outputField, laplaceField);
    } else {
        cout << "rodent brain is not supported yet! (use -humanBrain option)" << endl;
    }
    
    if (laplaceField == NULL) {
        cout << "The input vector field is NULL" << endl;
        return;
    } else {
        laplaceField->ComputeBounds();
        double* bounds = laplaceField->GetBounds();
        
        for (size_t j = 0; j < 3; j++) {
            cout << bounds[2*j] << " - " << bounds[2*j+1] << "; ";
        }
        cout << endl;
    }
    
    
    vtkIO vio;
    vtkPolyData* innerObj = vio.readFile(inputInnerObj);
    vtkPolyData* outerObj = vio.readFile(inputOuterObj);
    
    if (opts.GetString("-traceDirection", "forward") == "backward") {
        vtkPolyData* streams = performStreamTracer(opts, laplaceField, outerObj, innerObj);
        if (streams != NULL) {
            vio.writeFile(outputStream, streams);
            runPrintTraceCorrespondence(opts, inputOuterObj, outputStream, outputMesh, outerObj);
            vio.writeFile(outputObj, outerObj);
        }
    } else {
        vtkPolyData* streams = performStreamTracer(opts, laplaceField, innerObj, outerObj);
        if (streams != NULL) {
            vio.writeFile(outputStream, streams);
            runPrintTraceCorrespondence(opts, inputInnerObj, outputStream, outputMesh, innerObj);
            vio.writeFile(outputObj, innerObj);
        }
    }
}


void runSurfaceSampling(Options& opts, StringVector& args) {
    if (!opts.HasString("-scalarName")) {
        cout << "-scalarName argument is required!" << endl;
        return;
    }
    string scalarName = opts.GetString("-scalarName");
    string sourceFile = args[0];
    string outputFile = args[1];
    
    vtkDataSet* source = vio.readDataFile(sourceFile);
    vtkDataArray* scalars = source->GetPointData()->GetArray(scalarName.c_str());
    vtkDataSet* output = vio.readDataFile(outputFile);
    output->ComputeBounds();
    
    vtkNew<vtkCellLocator
    > cellLoc;
    cellLoc->SetDataSet(source);
    cellLoc->BuildLocator();
    
    vtkNew<vtkDoubleArray> outputScalars;
    outputScalars->SetNumberOfComponents(1);
    outputScalars->SetName(scalarName.c_str());
    outputScalars->SetNumberOfValues(output->GetNumberOfPoints());
    output->GetPointData()->AddArray(outputScalars.GetPointer());
    
    vector<vtkIdType> failedPoints(1000);
    vtkNew<vtkIdList> cellsOnLines;
    
    double center[3];
    output->GetCenter(center);
    
    cout << "Processing " << output->GetNumberOfPoints() << " points ..." << endl;
    for (size_t j = 0; j < output->GetNumberOfPoints(); j++) {
        double x[3], pc[3], w[3], s = 0;
        cellsOnLines->Reset();
        
        output->GetPoint(j, x);
        cellLoc->FindCellsAlongLine(x, center, 0, cellsOnLines.GetPointer());
        
        double cp[3], dist2;
        int subId;
        

        if (cellsOnLines->GetNumberOfIds() >= 0) {
            vtkIdType closestCellId = -1;
            double minDist2 = DBL_MAX;
            for (size_t i = 0; i < cellsOnLines->GetNumberOfIds(); i++) {
                vtkCell* onCell = source->GetCell(cellsOnLines->GetId(i));
                onCell->EvaluatePosition(x, cp, subId, pc, dist2, w);
                if (minDist2 > dist2) {
                    closestCellId = cellsOnLines->GetId(i);
                }
            }
            
            vtkCell* closestCell = source->GetCell(closestCellId);
            vtkIdType nPoints = closestCell->GetNumberOfPoints();
            for (size_t k = 0; k < nPoints; k++) {
                vtkIdType p = closestCell->GetPointId(k);
                s += (w[k] * scalars->GetTuple1(p));
            }

            outputScalars->SetValue(j, s);
        } else {
            failedPoints.push_back(j);
        }
    }

    set<vtkIdType> nbrs;
    vtkNew<vtkIdList> cells;
    for (size_t j = 0; j < failedPoints.size(); j++) {
        nbrs.clear();
        cells->Reset();
        vtkIdType ptId = failedPoints[j];
        output->GetPointCells(ptId, cells.GetPointer());
        for (size_t k = 0; k < cells->GetNumberOfIds(); k++) {
            findNeighborPoints(output->GetCell(cells->GetId(k)), ptId, nbrs);
        }
        

        double nNbrs = nbrs.size();
        set<vtkIdType>::iterator iter = nbrs.begin();
        double s = 0;
        for (; iter != nbrs.end(); iter++) {
            vtkIdType nId = *iter;
            s += (outputScalars->GetTuple1(nId) / nNbrs);
        }
        cout << s << endl;
        outputScalars->SetValue(ptId, s);
    }

    cout << "# of total points: " << outputScalars->GetNumberOfTuples() << endl;
    cout << "# of interpolated points: " << failedPoints.size() << endl;
    
    vio.writeFile(args[2], output);
}


void processVolumeOptions(Options& opts) {
    opts.addOption("-scalarName", "specify the scalar name to use", SO_REQ_SEP);
	opts.addOption("-x", "specify the x-value", SO_REQ_SEP);
	opts.addOption("-y", "specify the y-value", SO_REQ_SEP);
	opts.addOption("-z", "specify the z-value", SO_REQ_SEP);
	opts.addOption("-phi", "specify the phi-value", SO_REQ_SEP);
	opts.addOption("-theta", "specify the theta-value", SO_REQ_SEP);
	opts.addOption("-steps", "specify the number of steps", SO_REQ_SEP);
	opts.addOption("-dt", "time step", SO_REQ_SEP);
	opts.addOption("-iter", "number of iterations", SO_REQ_SEP);
	opts.addOption("-voi", "volume of interests (x means the default)", "-voi=x,x,x,x,10,10", SO_REQ_CMB);
	opts.addOption("-markBorderCells", "Mark border cells of an input dataset. The border cells have 1 in BorderCells data", "-markBorderCells input-data output-data", SO_NONE);
	opts.addOption("-markBorderPoints", "Mark border points of an input dataset. The border points will be marked as 2 and its exterior neighbors will be marked as 3.", "-markBorderPoints input-data output-data", SO_NONE);
	
	opts.addOption("-extractBorderline", "Extract the borderlines between different labels", "-extractBorderline obj.vtp", SO_NONE);
	
	opts.addOption("-fillGrid", "Fill the inside of a polydata with a uniform grid (refer -humanBrain option)", "-fillGrid input.vtp output.vtp", SO_NONE);
	
	opts.addOption("-extractStructuredGrid", "Extract structured grid from a polydata ", "-extractStructuredGrid input.vtp output.vts", SO_NONE);
	
	opts.addOption("-dims", "x-y-z dimensions", "-dims 100", SO_REQ_SEP);
	
	opts.addOption("-humanBrain", "Option to generate the filled uniform grid for a human brain", "-fillGrid CSF_GM_surface.vtk GM_WM_surface.vtk output.vts -humanBrain", SO_NONE);
	
	//
	opts.addOption("-sampleSurfaceScalarsForGrid", "Sample scalar values from a poly data at each grid point by finding a cell that intersects an edge of the grid", SO_NONE);
	
	// thickness measurement
	opts.addOption("-computeLaplacePDE", "Compute the Laplace PDE over the given domain", "-computeLaplacePDE input-data input-surface output-data ", SO_NONE);
	
	// RK45 stream tracer
	opts.addOption("-traceStream", "Trace a stream line from a given point set", "-traceStream input-vtu-field inner-surface-vtk outer-surface-vrk output-lines output-points", SO_NONE);

	// RK45 stream tracer post processing (line clipping)
	opts.addOption("-traceStreamPostProcessing", "Post processing the output of the TraceStream command", "-traceStreamPostProcessing input-streamlines.vtp input-seed.vtp output-streamlines.vtp", SO_NONE);
	
	opts.addOption("-traceDirection", "Choose the direction of stream tracing (both, forward, backward)", "-traceStream ... -traceDirection (both|forward|backward)", SO_REQ_SEP);
	
	opts.addOption("-measureThickness", "Measure the thickness of the solution domain via RK45 integration", "-measureThickness input-polydata output-polydata", SO_NONE);
	
	opts.addOption("-enclosingSphere", "Create a sphere that encloses a given object", "-enclosingSphere input-vtk output-sphere-vtk", SO_NONE);
	
	opts.addOption("-scanConversion", "Convert a polygonal mesh into a binary image", "-scanConversion mesh.vtk image.mhd", SO_NONE);
	
	opts.addOption("-extractGrid", "Extract a grid at the given z-value", "-extractGrid image.vts output.vts -z 30", SO_NONE);
	
	opts.addOption("-traceCorrespondence", "Print the correspondence from the stream trace result", "-traceCorrespondence input.vtk input_stream.vtk", SO_NONE);
	
	opts.addOption("-warpMesh", "Warp a mesh using streamline correspondences", "-warpMesh input.vtp streamlines.vtp output-prefix", SO_NONE);
	
    opts.addOption("-inputField", "An input vector field data set", "-inputField laplaceField.vts", SO_REQ_SEP);
    
    opts.addOption("-inputSphere", "An input sphere for spherical mapping", "-inputSphere sphere.vtp", SO_REQ_SEP);
    
	opts.addOption("-sphericalMapping", "Construct a spherical mapping from an object", "-sphericalMapping  inputSurface.vtp prefix", SO_NONE);

	opts.addOption("-surfaceCorrespondence", "Construct a surface correspondence between two objects; prefix is used for temporary files", "-sphericalMapping  source.vtp destination.vtp prefix", SO_NONE);
	
    opts.addOption("-surfaceToSurfaceSampling", "Sample scalar values from one surface to the other", "-surfaceToSurfaceSampling source.vtp output.vtp -scalarName scalar", SO_NONE);
    
	opts.addOption("-cellInfo", "Print each cell information (type, # of points, # of edges, size(?)", "-cellInfo vtkDataSet files ...", SO_NONE);
}

void processVolumeCommands(Options& opts, StringVector& args) {
	
	string input1File, outputFile;
	
	if (opts.GetBool("-markBorderCells")) {
		input1File = args[0];
		outputFile = args[1];
		vtkDataSet* data1 = vio.readDataFile(input1File);
		string scalarName = opts.GetString("-scalarName", "SelectedPoints");
		selectBoundaryCells(data1, scalarName);
		vio.writeFile(outputFile, data1);
	} else if (opts.GetBool("-markBorderPoints")) {
		input1File = args[0];
		outputFile = args[1];
		vtkDataSet* data1 = vio.readDataFile(input1File);
		string scalarName = opts.GetString("-scalarName", "SelectedPoints");
		selectBoundaryPoints(data1, scalarName);
		vio.writeFile(outputFile, data1);
	} else if (opts.GetBool("-extractBorderline")) {
		runExtractBorderline(opts, args);
	} else if (opts.GetBool("-fillGrid")) {
		runFillGrid(opts, args);
	} else if (opts.GetBool("-sampleSurfaceScalarsForGrid")) {
		input1File = args[0];
		string input2File = args[1];
		outputFile = args[2];
		
		string scalarName = opts.GetString("-scalarName", "");
		vtkDataSet* ds = vio.readDataFile(input1File);
		vtkPolyData* pd = vio.readFile(input2File);
		vtkDataSet* outDS = sampleSurfaceScalarsForGrid(ds, pd, scalarName);
		
		vio.writeFile(outputFile, outDS);
	} else if (opts.GetBool("-computeLaplacePDE")) {
		if (args.size() == 2) {
			input1File = args[0];
			outputFile = args[1];
			
			const double dt = opts.GetStringAsReal("-dt", 0.065);
			const int numIter = opts.GetStringAsInt("-iter", 5000);
			
			vtkDataSet* data = vio.readDataFile(input1File);
			computeLaplacePDE(data, 0, 10000, numIter, dt);
			vio.writeFile(outputFile, data);
		} else if (args.size() == 3) {
			input1File = args[0];
			string input2File = args[1];
			outputFile = args[2];
			
			vtkDataSet* data = vio.readDataFile(input1File);
			vtkPolyData* surfaceData = vio.readFile(input2File);
			computeLaplacePDE(data, 0, 10000, 5000, 0.065, surfaceData);
			vio.writeFile(outputFile, data);
		}
	} else if (opts.GetBool("-traceStream")) {
		// -traceStream 312.laplaceSol.vtp 312.sliceContour.vtk 312.thicknessSol.vtp 312.streams.vtp
		runStreamTracer(opts, args);
	} else if (opts.GetBool("-traceStreamPostProcessing")) {
		vtkPolyData* inputStream = vio.readFile(args[0]);
		vtkPolyData* inputSeeds = vio.readFile(args[1]);
		vtkPolyData* destSurf = vio.readFile(args[2]);
		vtkPolyData* outputStream = performStreamTracerPostProcessing(inputStream, inputSeeds, destSurf);
		
		vio.writeFile(args[3], outputStream);
	} else if (opts.GetBool("-measureThickness")) {
		runMeasureThickness(opts, args);
	} else if (opts.GetBool("-extractStructuredGrid")) {
		runExtractStructuredGrid(opts, args);
	} else if (opts.GetBool("-enclosingSphere")) {
		if (args.size() >= 2) {
			cout << "creating an enclosing sphere..." << endl;
			runEnclosingSphere(opts, args[0], args[1], false, 0, opts.GetStringAsInt("-phi", 8), opts.GetStringAsInt("-theta", 8));
		}
	} else if (opts.GetBool("-scanConversion")) {
		runScanConversion(opts, args);
	} else if (opts.GetBool("-extractGrid")) {
		runExtractSlice(opts, args);
	} else if (opts.GetBool("-traceCorrespondence")) {
		runPrintTraceCorrespondence(opts, args[0], args[1], args[2]);
	} else if (opts.GetBool("-warpMesh")) {
		runWarpMesh(opts, args);
	} else if (opts.GetBool("-sphericalMapping")) {
		runSphericalMapping(opts, args);
    } else if (opts.GetBool("-surfaceCorrespondence")) {
        runSurfaceCorrespondence(opts, args);
    } else if (opts.GetBool("-surfaceToSurfaceSampling")) {
        runSurfaceSampling(opts, args);
	} else if (opts.GetBool("-cellInfo")) {
		struct showCellInfo {
			showCellInfo(vtkDataSet* ds) {
                set<int> cellTypes;
				for (size_t j = 0; j < ds->GetNumberOfCells(); j++) {
					vtkCell* cell = ds->GetCell(j);
                    int cellType = cell->GetCellType();
                    cellTypes.insert(cellType);
				}
                set<int>::iterator it = cellTypes.begin();
                cout << "Unique cell types: ";
                for (; it != cellTypes.end(); it++) {
                    cout << *it << " ";
                }
                cout << endl;
			}
		};
		showCellInfo(vio.readFile(args[0]));
	}

}
