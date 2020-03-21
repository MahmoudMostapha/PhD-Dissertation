//
//  kgeometry.h
//  ktools
//
//  Created by Joowhi Lee on 8/31/15.
//
//

#ifndef __ktools__kgeometry__
#define __ktools__kgeometry__

#include <stdio.h>
#include <vtkNew.h>
#include <vtkIdList.h>
#include <vtkTransform.h>
#include <vtkIdTypeArray.h>
#include <vtkPolyData.h>
#include <vtkDataSet.h>
#include <vtkCell.h>

#include <exception>
#include <stdexcept>
#include <map>
#include <set>
#include <vector>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include <cstddef>
#include <math.h>
#include <limits>
#include <iostream>

static const double _northPole[3] = { 0, 0, 1 };

class LocalSurfaceTopology {
public:
	vtkIdType centerId;
	std::set<vtkIdType> edgeStart;
	std::map<vtkIdType,vtkIdType> edges;
	bool boundary;
	
	LocalSurfaceTopology(): centerId(-1), boundary(false) {}
	LocalSurfaceTopology(vtkIdType id): centerId(id), boundary(false) {}
	
	void setCenterId(vtkIdType id) {
		centerId = id;
	}
	
	void addEdge(vtkIdType u, vtkIdType v) {
		edges[u] = v;
		if (edgeStart.find(u) == edgeStart.end()) {
			edgeStart.insert(u);
		} else {
			edgeStart.erase(u);
		}
		if (edgeStart.find(v) == edgeStart.end()) {
			edgeStart.insert(v);
		} else {
			edgeStart.erase(v);
		}
	}
	
	vtkIdType getCenterId() {
		return centerId;
	}
	
	void getOrderedNeighbors(std::vector<vtkIdType>& boundaries) {
		vtkIdType s = -1;
		boundaries.clear();
		if (edgeStart.size() == 1) {
			boundaries.push_back(centerId);
			s = *(edgeStart.begin());
			boundaries.push_back(s);
			boundary = true;
            throw std::logic_error("can't happen!");
		} else if (edgeStart.empty()) {
			s = (edges.begin())->first;
			boundaries.push_back(s);
			boundary = false;
		} else if (edgeStart.size() == 2) {
			std::set<vtkIdType>::const_iterator iter = edgeStart.begin();
			for (; iter != edgeStart.end(); iter++) {
                if (edges.find(*iter) != edges.end()) {
                    s = *iter;
                    break;
                }
			}
//            boundaries.push_back(s);
            boundary = true;
        } else {
            throw std::logic_error("multiple segment!");
        }
		
		for (vtkIdType n = edges[s]; edges.find(n) != edges.end() && n != s; n = edges[n]) {
			boundaries.push_back(n);
		}
		if (boundary) {
			boundaries.push_back(centerId);
		} else {
			boundaries.push_back(s);
		}
	}
};



class SurfaceTopology {
public:
	vtkPolyData* dataSet;
	const size_t nPoints;
	std::tr1::unordered_map<vtkIdType, LocalSurfaceTopology> topologyMap;
	std::tr1::unordered_map<vtkIdType, std::vector<vtkIdType> > neighborIds;
	
	SurfaceTopology(vtkPolyData* ds): dataSet(ds), nPoints(ds->GetNumberOfPoints()) {
		buildTopology();
	}
	
private:
	void addEdge(vtkIdType j, vtkIdType u, vtkIdType v) {
		topologyMap[j].setCenterId(j);
		topologyMap[j].addEdge(u, v);
	}
	
	
	void buildTopology() {
		vtkNew<vtkIdList> cellIds;
		for (size_t j = 0; j < nPoints; j++) {
			dataSet->GetPointCells(j, cellIds.GetPointer());
			for (size_t k = 0; k < cellIds->GetNumberOfIds(); k++) {
				vtkCell* cell = dataSet->GetCell(cellIds->GetId(k));
				for (size_t h = 0; h < cell->GetNumberOfEdges(); h++) {
					vtkCell* edge = cell->GetEdge(h);
					vtkIdType u = edge->GetPointId(0);
					vtkIdType v = edge->GetPointId(1);
					if (u != j && v != j) {
						addEdge(j, u, v);
					}
				}
			}
		}
		
		for (size_t j = 0; j < nPoints; j++) {
			topologyMap[j].getOrderedNeighbors(neighborIds[j]);
		}
	}
};

class Geometry {
public:
    vtkNew<vtkTransform> txfm;
	
	struct Edge {
		vtkIdType u;
		vtkIdType v;
		bool boundary;
        size_t axisAligned;
		
		bool operator==(const Edge& e) const {
			return (u == e.u && v == e.v) || (u == e.v && v == e.u);
		}
		
		bool operator<(const Edge& e) const {
			return (u == e.u ? v < e.v : u < e.u);
		}
		
		bool operator<=(const Edge& e) const {
			return (u == e.u ? v <= e.v : u <= e.u);
		}
		
		Edge(): u(-1), v(-1), boundary(false) {}
		Edge(vtkIdType s, vtkIdType e): u(s), v(e), boundary(false), axisAligned(0) {}
		Edge(vtkIdType s, vtkIdType e, bool b): u(s), v(e), boundary(b), axisAligned(0) {}
		Edge(const Edge& e): u(e.u), v(e.v), boundary(e.boundary), axisAligned(0) {}
		
		void operator=(const Edge& e) {
			u = e.u;
			v = e.v;
            boundary = e.boundary;
            axisAligned = e.axisAligned;
		}
	};
	typedef std::map<vtkIdType, Edge> EdgeMap;
    typedef std::vector<EdgeMap> EdgeList;
	typedef std::vector<vtkIdType> Neighbors;
	typedef std::vector<std::vector<vtkIdType> > NeighborList;
	
    size_t extractEdges(vtkDataSet* ds, EdgeList& edges);
	size_t extractNeighbors(vtkDataSet* ds, NeighborList& nbrs);
    
    double tangentVector(const double u[3], const double v[3], const double n[3], double tv[3], vtkTransform* txf = NULL);
	
	double rotateVector(const double p[3], const double q[3], vtkTransform* txf, double* crossOut = NULL);
	
	double rotatePlane(const double u1[3], const double v1[3], const double u2[3], const double u3[3], vtkTransform* txf);
    
    double normalizeToNorthPole(const double u[3], const double n[3], double* cross, vtkTransform* txf);
    
    double denormalizeFromNorthPole(const double u[3], const double n[3], double* cross, vtkTransform* txf);
	
	
	bool computeContourNormals(vtkDataSet* ds, vtkDoubleArray* normals);
	
	vtkPolyData* computeSurfaceNormals(vtkPolyData* pd);
	
	void print();
};



#endif /* defined(__ktools__kgeometry__) */
