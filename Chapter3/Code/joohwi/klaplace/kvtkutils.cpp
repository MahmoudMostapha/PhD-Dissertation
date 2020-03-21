//
//  kvtkutils.cpp
//  ktools
//
//  Created by Joowhi Lee on 11/1/15.
//
//

#include "kvtkutils.h"

#include <vtkDataSet.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPointLocator.h>
#include <vtkNew.h>

#include "vtkio.h"
#include "csv_parser.h"

using namespace pi;
using namespace std;

vtkIO vio;

vector<StringVector> readCSV(string file) {
	vector<StringVector> rows;
	
	csv_parser csv;
	csv.init(file.c_str());
	csv.set_enclosed_char('"', ENCLOSURE_OPTIONAL);
	csv.set_field_term_char(',');
	csv.set_line_term_char('\n');
	
	for (int row_count = 0; csv.has_more_rows(); row_count++) {
		csv_row row = csv.get_row();
		StringVector sv;
		for (size_t j = 0; j < row.size(); j++) {
			sv.push_back(row[j]);
		}
		rows.push_back(sv);
	}
	
	return rows;
}


void probePoints(Options& opts, StringVector& args) {
	if (!opts.HasString("-csvfile")) {
		cout << "csv file must be specified (-csvfile)" << endl;
		return;
	}

	vtkDataSet* ds = vio.readDataFile(args[0]);
	vtkDataArray* darr = NULL;
	if (opts.HasString("-scalarName")) {
		darr = ds->GetPointData()->GetArray(opts.GetString("-scalarName").c_str());
	} else {
		darr = ds->GetPointData()->GetScalars();
	}

	vtkNew<vtkPointLocator> ploc;
	ploc->SetDataSet(ds);
	ploc->BuildLocator();

	vector<StringVector> rows = readCSV(opts.GetString("-csvfile").c_str());
	for (size_t j = 0; j < rows.size(); j++) {
		StringVector& d = rows[j];
		double p[3];
		p[0] = atof(d[0].c_str());
		p[1] = atof(d[1].c_str());
		p[2] = atof(d[2].c_str());
		
		vtkIdType pId = ploc->FindClosestPoint(p);
	}

}


void processUtilityOptions(Options& opts) {
	opts.addOption("-scalarName", "Point scalar name", "-scalarName scalarName", SO_NONE);
	opts.addOption("-csvfile", "a csv file", "-csvfile a.csv", SO_REQ_SEP);
	opts.addOption("-probePoints", "Inspect scalar values at a given points", "-probePoints mesh.vtk -csvfile points.csv -scalarName LaplacianGradientNorm", SO_NONE);
}


void processUtilityCommands(Options& opts, StringVector& args) {
	if (opts.GetBool("-probePoints")) {
		probePoints(opts, args);
	}
}