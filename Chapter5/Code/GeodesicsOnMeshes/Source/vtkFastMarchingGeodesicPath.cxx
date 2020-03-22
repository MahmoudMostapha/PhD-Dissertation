/*=========================================================================

  Copyright (c) Karthik Krishnan
  See Copyright.txt for details.

=========================================================================*/
#include "vtkFastMarchingGeodesicPath.h"

#include "vtkFastMarchingGeodesicDistance.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkExecutive.h"
#include "vtkIdList.h"
#include "vtkMath.h"
#include "vtkSmartPointer.h"
#include "vtkCellArray.h"
#include "vtkNew.h"

#include "GW_GeodesicMesh.h"
#include "GW_GeodesicPath.h"
#include "GW_Vertex.h"
#include "GW_Face.h"
#include "GW_Config.h"
#include <assert.h>
#include <set>

#ifdef _WIN32
// new is being defined to a new method that takes in 4 parameters.
// Go back to what its supposed to be !
#ifdef new
#undef new
#endif
#endif

//-----------------------------------------------------------------------------
vtkStandardNewMacro(vtkFastMarchingGeodesicPath);

//-----------------------------------------------------------------------------
vtkFastMarchingGeodesicPath::vtkFastMarchingGeodesicPath()
{
  this->MaximumPathPoints       = GW_INFINITE;     // no limit
  this->InterpolationOrder      = 1;               // linear
  this->BeginPointId            = -1;              // undefined
  this->Geodesic                = vtkFastMarchingGeodesicDistance::New();
  this->ZerothOrderPathPointIds = vtkIdList::New();
  this->FirstOrderPathPointIds  = vtkIdList::New();
  this->GeodesicLength          = 0;
}

//-----------------------------------------------------------------------------
vtkFastMarchingGeodesicPath::~vtkFastMarchingGeodesicPath()
{
  this->ZerothOrderPathPointIds->Delete();
  this->FirstOrderPathPointIds->Delete();
  this->Geodesic->Delete();
}

//----------------------------------------------------------------------------
int vtkFastMarchingGeodesicPath::RequestData(
  vtkInformation *           vtkNotUsed( request ),
  vtkInformationVector **    inputVector,
  vtkInformationVector *     outputVector)
{
  vtkInformation * inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  vtkPolyData *input = vtkPolyData::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPolyData *output = vtkPolyData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (!output || !input)
    {
    return 0;
    }

  vtkNew< vtkIdList > terminationIds;
  terminationIds->InsertNextId(this->BeginPointId);
  this->Geodesic->SetDestinationVertexStopCriterion(
      terminationIds.GetPointer());

  // This will re-run fast marching and compute the distance field from the
  // seeded points, if necessary (if the mesh or seeds have changed)
  this->Geodesic->Update();

  // Initialize the GW_GeodesicMesh structure
  this->ComputePath(output);

  return 1;
}

//-----------------------------------------------------------------------------
void vtkFastMarchingGeodesicPath::SetSeeds( vtkIdList *seeds )
{
  this->Geodesic->SetSeeds(seeds);
}

//-----------------------------------------------------------------------------
vtkIdList *vtkFastMarchingGeodesicPath::GetSeeds()
{
  return this->Geodesic->GetSeeds();
}

//-----------------------------------------------------------------------------
void vtkFastMarchingGeodesicPath::ComputePath(vtkPolyData *pd)
{
  this->GeodesicLength = 0;
  this->ZerothOrderPathPointIds->Initialize();
  this->FirstOrderPathPointIds->Initialize();

  vtkSmartPointer< vtkPoints > pathPoints = vtkSmartPointer< vtkPoints >::New();
  pathPoints->Initialize();

  GW::GW_GeodesicMesh *mesh = (GW::GW_GeodesicMesh *)(
                        this->Geodesic->GetGeodesicMesh());
  GW::GW_GeodesicVertex* begin =
    (GW::GW_GeodesicVertex*)(mesh->GetVertex((GW::GW_U32)this->BeginPointId));
  if (!begin)
    {
    vtkErrorMacro( << "BeginPointId was not found to lie on the mesh." );
    return;
    }

  GW::GW_GeodesicPath track;
  track.ComputePath(*begin, this->MaximumPathPoints);

  GW::T_GeodesicPointList ptList = track.GetPointList();
  GW::GW_GeodesicPoint* pt;
  float parametricPos;
  GW::GW_GeodesicVertex *endVert1, *endVert2;
  GW::GW_Vector3D endPt1, endPt2;
  double pathPt[3], lastPathPt[3];
  vtkIdType endPtId1, endPtId2, lastInsertedPtId = -1;
  vtkIdType i = 0, i0 = 0;

  const size_t nPts = ptList.size();
  pathPoints->SetNumberOfPoints(nPts);

  // With linear interpolation we return a pair of point ids (corresponding to
  // the triangle edge end points) for each path point.
  this->ZerothOrderPathPointIds->SetNumberOfIds( nPts );
  if (this->InterpolationOrder == 1)
    {
    this->FirstOrderPathPointIds->SetNumberOfIds( nPts * 2 );
    }

  for ( GW::CIT_GeodesicPointList cit = ptList.begin(), citEnd = ptList.end();
        cit != citEnd; ++cit, ++i, lastPathPt[0] = pathPt[0],
        lastPathPt[1] = pathPt[1], lastPathPt[2] = pathPt[2])
    {
    pt = *cit;

      // The parametric position of the vertex on the edge
    parametricPos = pt->GetCoord();

    // Get the end points of the edge on which the path lies.
    endVert1 = pt->GetVertex1();
    endVert2 = pt->GetVertex2();
    endPt1 = endVert1->GetPosition();
    endPt2 = endVert2->GetPosition();
    endPtId1 = endVert1->GetID();
    endPtId2 = endVert2->GetID();

    // Store the edge point ids. The ZerothOrderPointIds contain the closest
    // one. The FirstOrderPointIds contains the other one.
    if (parametricPos > 0.5)
      {

      if (lastInsertedPtId != endPtId1)
        {
        // avoid repeats
        lastInsertedPtId = endPtId1;
        this->ZerothOrderPathPointIds->SetId(i0, endPtId1);
        pathPt[0] = endPt1[0];
        pathPt[1] = endPt1[1];
        pathPt[2] = endPt1[2];
        if (this->InterpolationOrder == 0)
          {
          pathPoints->SetPoint(i0, pathPt[0], pathPt[1], pathPt[2]);
          }
        ++i0;
        }

      if (this->InterpolationOrder == 1)
        {
        this->FirstOrderPathPointIds->SetId(2*i, endPtId1);
        this->FirstOrderPathPointIds->SetId(2*i+1, endPtId2);
        }
      }
    else
      {
      if (lastInsertedPtId != endPtId2)
        {
        // avoid repeats
        lastInsertedPtId = endPtId2;
        this->ZerothOrderPathPointIds->SetId(i0, endPtId2);
        pathPt[0] = endPt2[0];
        pathPt[1] = endPt2[1];
        pathPt[2] = endPt2[2];
        if (this->InterpolationOrder == 0)
          {
          pathPoints->SetPoint(i0, pathPt[0], pathPt[1], pathPt[2]);
          }
        ++i0;
        }

      if (this->InterpolationOrder == 1)
        {
        this->FirstOrderPathPointIds->SetId(2*i, endPtId2);
        this->FirstOrderPathPointIds->SetId(2*i+1, endPtId1);
        }
      }

    if (this->InterpolationOrder == 1)
      {
      // Linearly interpolate the edge vertices based on the parametric
      // position
      pathPt[0] = parametricPos * endPt1[0] + (1-parametricPos) * endPt2[0];
      pathPt[1] = parametricPos * endPt1[1] + (1-parametricPos) * endPt2[1];
      pathPt[2] = parametricPos * endPt1[2] + (1-parametricPos) * endPt2[2];

      pathPoints->SetPoint(i, pathPt[0], pathPt[1], pathPt[2]);
      }

    // The curve length
    if (i)
      {
      this->GeodesicLength += sqrt(
        vtkMath::Distance2BetweenPoints(lastPathPt, pathPt));
      }

    } // end loop over vertices in the gradient trace

  // Set the size to the actual size, which may be less than the track.npts
  // because we avoid repeats.
  this->ZerothOrderPathPointIds->SetNumberOfIds( i0 );
  if (this->InterpolationOrder == 0)
    {
    pathPoints->SetNumberOfPoints(i0);
    }

  // Set this path on the output. Its a polyline with a single cell.
  int nUniquePoints = pathPoints->GetNumberOfPoints();
  pd->SetPoints(pathPoints);
  vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
  lines->InsertNextCell(nUniquePoints);
  for (int i = 0; i < nUniquePoints; i++)
    {
    lines->InsertCellPoint(i);
    }
  pd->SetLines(lines);
}

//----------------------------------------------------------------------------
void vtkFastMarchingGeodesicPath::SetInputConnection(
    int port, vtkAlgorithmOutput* input)
{
  this->Superclass::SetInputConnection(port, input);
  this->Geodesic->SetInputConnection(port, input);
}

//-----------------------------------------------------------------------------
void vtkFastMarchingGeodesicPath::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << this->Geodesic << "\n";
  if (this->Geodesic)
    {
    this->Geodesic->PrintSelf(os, indent.GetNextIndent());
    }
  os << indent << "BeginPointId: " << this->BeginPointId << "\n";
  os << indent << "InterpolationOrder: " << this->InterpolationOrder << "\n";
  os << indent << "GeodesicLength: " << this->GeodesicLength << "\n";
  os << indent << "MaximumPathPoints: " << this->MaximumPathPoints << "\n";
  os << indent << "ZerothOrderPathPointIds: "
     << this->ZerothOrderPathPointIds << "\n";
  os << indent << "FirstOrderPathPointIds: "
     << this->FirstOrderPathPointIds << "\n";
}
