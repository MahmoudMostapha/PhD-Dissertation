//
//  kgeometry.cpp
//  ktools
//
//  Created by Joowhi Lee on 8/31/15.
//
//

#include "kgeometry.h"

#include <vtkNew.h>
#include <vtkMath.h>
#include <vtkPolyDataNormals.h>
#include <set>
#include <algorithm>

using namespace std;


size_t Geometry::extractEdges(vtkDataSet *ds, std::vector<EdgeMap> &edges) {
    size_t nEdges = 0;
    
    if (ds == NULL) {
        return nEdges;
    }
	
	vector<set<vtkIdType> > edgeList;
	edgeList.resize(ds->GetNumberOfPoints());
    edges.resize(ds->GetNumberOfPoints());
	
    const size_t nCells = ds->GetNumberOfCells();
    for (size_t j = 0; j < nCells; j++) {
        vtkCell* cell = ds->GetCell(j);
        const size_t ne = cell->GetNumberOfEdges();
        for (size_t k = 0; k < ne; k++) {
            vtkCell* edge = cell->GetEdge(k);
			vtkIdType s = edge->GetPointId(0);
			vtkIdType e = edge->GetPointId(1);
            
            double sPt[3], ePt[3];
            ds->GetPoint(s, sPt);
            ds->GetPoint(e, ePt);
            
            size_t nEq = 0;
            for (size_t j = 0; j < 3; j++) {
                if (sPt[j] == ePt[j]) {
                    nEq++;
                }
            }
            
		
			if (edgeList[e].find(s) != edgeList[e].end()) {
				edgeList[e].erase(s);
				edges[s][e] = Edge(s, e);
				edges[e][s].boundary = false;
			} else {
				edgeList[s].insert(e);
				edges[s][e] = Edge(s, e, true);
			}
            
            edges[s][e].axisAligned = nEq;
        }
    }
    
    return nEdges;
}

size_t Geometry::extractNeighbors(vtkDataSet *ds, NeighborList &nbrs) {
	const size_t nPoints = ds->GetNumberOfPoints();
	nbrs.resize(nPoints);
	
	vtkNew<vtkIdList> cellIds;
	for (size_t j = 0; j < nPoints; j++) {
		Neighbors& nbrPts = nbrs[j];
		cellIds->Reset();
		ds->GetPointCells(j, cellIds.GetPointer());
		for (size_t k = 0; k < cellIds->GetNumberOfIds(); k++) {
			vtkIdType cellId = cellIds->GetId(k);
			vtkCell* cell = ds->GetCell(cellId);
			for (size_t l = 0; l < cell->GetNumberOfEdges(); l++) {
				vtkCell* edge = cell->GetEdge(l);
				vtkIdType s = edge->GetPointId(0);
				vtkIdType e = edge->GetPointId(1);
				if (s == j) {
					if (find(nbrPts.begin(), nbrPts.end(), e) == nbrPts.end()) {
						nbrPts.push_back(e);
					}
				} else if (e == j) {
					if (find(nbrPts.begin(), nbrPts.end(), s) == nbrPts.end()) {
						nbrPts.push_back(s);
					}
				}
			}
		}
	}
	
	return nPoints;
}

double Geometry::tangentVector(const double *u, const double *v, const double *n, double *tv, vtkTransform* txf) {
    
    double uv[3];
    vtkMath::Subtract(v, u, uv);
    

    double dotProd = vtkMath::Dot(uv, n);
    
    double cross[3];
    vtkMath::Cross(uv, n, cross);
    double crossProd = vtkMath::Normalize(cross);

    double angRad = atan2(crossProd, dotProd);
    double angDeg = vtkMath::DegreesFromRadians(angRad) - 90;
    
    if (txf == NULL) {
        vtkNew<vtkTransform> tx;
        tx->RotateWXYZ(angDeg, cross);
        tx->TransformPoint(uv, tv);
        vtkMath::Add(tv, u, tv);
    } else {
        txf->RotateWXYZ(angDeg, cross);
        txf->TransformPoint(uv, tv);
        vtkMath::Add(tv, u, tv);
    }

    return angRad;
}

double Geometry::rotateVector(const double p[3], const double q[3], vtkTransform* txf, double* crossOut) {
	
    double u[3], v[3];
	memcpy(u, p, sizeof(u));
	memcpy(v, q, sizeof(v));
	
//	double pNorm = vtkMath::Normalize(u);
//	double qNorm = vtkMath::Normalize(v);
	
	double dotProd = vtkMath::Dot(u, v);
	
	double cross[3];
	vtkMath::Cross(u, v, cross);
	double crossProd = vtkMath::Normalize(cross);
	if (crossProd == 0) {
		memcpy(cross, u, sizeof(u));
	}
	
	if (crossOut != NULL) {
		for (size_t j = 0; j < 3; j++) crossOut[j] = cross[j];
	}
	
	double angRad = atan2(crossProd, dotProd);
    double angDeg = vtkMath::DegreesFromRadians(angRad);
	txf->RotateWXYZ(angDeg, cross);
	
	return angDeg;
}

double Geometry::rotatePlane(const double u1[3], const double v1[3], const double u2[3], const double v2[3], vtkTransform* txf) {
    
    double cross1[3], cross2[3];
    vtkMath::Cross(u1, v1, cross1);
    vtkMath::Cross(u2, v2, cross2);
    
    return rotateVector(cross1, cross2, txf);
}

double Geometry::normalizeToNorthPole(const double *u, const double *n, double* cross, vtkTransform* txf) {
    static const double northPole[3] = { 0, 0, 1 };
    static const double xAxis[3] = { 1, 0, 0 };
    
    vtkMath::Cross(n, northPole, cross);
    
    double crossProd = vtkMath::Norm(cross);
    double dotProd = vtkMath::Dot(n, northPole);
    
    if (crossProd == 0) {
        memcpy(cross, xAxis, sizeof(xAxis));
    }
    
    double angDeg = vtkMath::DegreesFromRadians(atan2(crossProd, dotProd));
    
    txf->PostMultiply();
    txf->Translate(-u[0], -u[1], -u[2]);
    txf->RotateWXYZ(angDeg, cross);
    
    return angDeg;
}


double Geometry::denormalizeFromNorthPole(const double u[3], const double n[3], double* cross, vtkTransform* txf) {
    static const double northPole[3] = { 0, 0, 1 };
    static const double xAxis[3] = { 1, 0, 0 };
    
    vtkMath::Cross(northPole, n, cross);
    
    double crossProd = vtkMath::Norm(cross);
    double dotProd = vtkMath::Dot(northPole, n);
    
    if (crossProd == 0) {
        memcpy(cross, xAxis, sizeof(xAxis));
    }
    
    double angDeg = vtkMath::DegreesFromRadians(atan2(crossProd, dotProd));
    
    txf->PostMultiply();
    txf->RotateWXYZ(angDeg, cross);
	txf->Translate(u[0], u[1], u[2]);
	
	
    return angDeg;
}

bool Geometry::computeContourNormals(vtkDataSet* ds, vtkDoubleArray* normals) {
	Geometry geom;
	
	EdgeList edges;
	geom.extractEdges(ds, edges);
	
	for (size_t j = 0; j < edges.size(); j++) {
		EdgeMap::const_iterator iter = edges[j].begin();
		for (; iter != edges[j].end(); iter++) {
			
		}
	}
	return false;
}

vtkPolyData* Geometry::computeSurfaceNormals(vtkPolyData *pd) {
	vtkNew<vtkPolyDataNormals> normalsFilter;
	normalsFilter->SetInputData(pd);
	normalsFilter->ComputePointNormalsOn();
	normalsFilter->Update();
	return normalsFilter->GetOutput();
}

void Geometry::print() {
	
}
