#include "klaplace.h"

#include <string>
#include <vector>
#include <exception>

#include "vtkio.h"

#include <vtkIdList.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkDataSet.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCell.h>
#include <vtkGenericCell.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkThresholdPoints.h>
#include <vtkNew.h>
#include <vtkExtractGrid.h>
#include <vtkGradientFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkCleanPolyData.h>
#include <vtkMath.h>
#include <vtkCellLocator.h>
#include <vtkPointLocator.h>
#include <vtkModifiedBSPTree.h>
#include <vtkImageToStructuredGrid.h>

#include "kgeometry.h"
#include "kstreamtracer.h"
#include "kvolume.h"
#include "kvtkutils.h"

using namespace std;
using namespace pi;


static vtkIO vio;



void processGeometryOptions(Options& opts) {
	opts.addOption("-connectivity", "Print edge connectivity",  SO_NONE);
	opts.addOption("-img2vts", "Convert an image to vtkStructuredGrid", SO_NONE);
    opts.addOption("-conv", "File type conversion via vtkIO.cpp", SO_NONE);
}


void processGeometryCommands(Options& opts, StringVector& args) {
	if (opts.GetBool("-connectivity")) {
		Geometry geom;
		Geometry::NeighborList nbrs;

		vtkDataSet* ds = vio.readDataFile(args[0]);
		geom.extractNeighbors(ds, nbrs);
		
		const size_t nPoints = nbrs.size();
		for (size_t j = 0; j < nPoints; j++) {
			cout << j << "]: ";
			Geometry::Neighbors::const_iterator iter = nbrs[j].begin();
			for (size_t k = 0; iter != nbrs[j].end(); k++, iter++) {
				cout << *iter << " ";
			}
			cout << endl;
		}
		return;
	} else if (opts.GetBool("-img2vts")) {
		vtkDataSet* img = vio.readDataFile(args[0]);
		vtkNew<vtkImageToStructuredGrid> filt;
		filt->SetInputData(img);
		filt->Update();
		vio.writeFile(args[1], filt->GetOutput());
	} else if (opts.GetBool("-conv")) {
        vtkDataSet* ds = vio.readDataFile(args[0]);
        vio.writeFile(args[1], ds);
    }
}

//void processVTKUtils(pi::Options opts, pi::StringVector args) {
//    vtkIO vio;
//    string input1File, input2File, outputFile;
//    if (opts.GetBool("-sampleSurfaceScalars")) {
//        input1File = args[0];
//        input2File = args[1];
//        outputFile = args[2];
//        vtkDataSet* data1 = vio.readDataFile(input1File);
//        vtkPolyData* surf1 = vio.readFile(input2File);
//        sampleSurfaceScalar(data1, opts.GetString("-scalarName", "BorderPoints").c_str(), surf1, "labels");
//        vio.writeFile(outputFile, data1);
//        if (opts.GetString("-o") != "") {
//            vio.writeFile(opts.GetString("-o"), surf1);
//        }
//    } else if (opts.GetBool("-buildAdjacencyList")) {
//        input1File = args[0];
//        outputFile = args[1];
//
//        string scalarName = opts.GetString("-scalarName", "SampledSurfaceScalars");
//        vtkDataSet* data1 = vio.readDataFile(input1File);
//
//        vector<pair<vtkIdType, vector<vtkIdType> > > graph;
//        buildAdjacencyList(data1, scalarName, graph);
//
//        vio.writeFile(outputFile, data1);
//    } else  if (opts.GetBool("-thresholdStream")) {
//        runStreamLineThreshold(opts, args);
//    } else if (opts.GetBool("-rescaleStream")) {
//        runRescaleStream(opts, args);
//    } else if (opts.GetBool("-traceClipping")) {
//        runTraceClipping(opts, args);
//    } else if (opts.GetBool("-traceScalarCombine")) {
//        runTraceScalarCombine(opts, args);
//    } else if (opts.GetBool("-extractSurfaceBorder")) {
//        runExtractSurfaceBorder(opts, args);
//    }
//}
//

int main(int argc, char* argv[]) {
	clock_t t1 = clock();

    Options opts;
    opts.addOption("-h", "print help message", SO_NONE);

	processGeometryOptions(opts);
    processVolumeOptions(opts);
	processUtilityOptions(opts);

    StringVector args = opts.ParseOptions(argc, argv, NULL);

    if (opts.GetBool("-h")) {
        cout << "## *kmesh* Usage" << endl;
        opts.PrintUsage();
        return 0;
    }

    processVolumeCommands(opts, args);
	processGeometryCommands(opts, args);
	processUtilityCommands(opts, args);
	
	clock_t t2 = clock();
	cout << "Elapsed Time: " << (t2-t1)*(1e-3) << " ms" << endl;
	
    return 0;
}
