/*=========================================================================

  Copyright (c) Karthik Krishnan
  See Copyright.txt for details.

=========================================================================*/

#include "vtkPolyDataGeodesicDistance.h"

#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkExecutive.h"
#include "vtkFloatArray.h"
#include "vtkFieldData.h"
#include "vtkPointData.h"
#include "vtkIdList.h"

vtkCxxSetObjectMacro( vtkPolyDataGeodesicDistance, Seeds, vtkIdList );

//-----------------------------------------------------------------------------
vtkPolyDataGeodesicDistance::vtkPolyDataGeodesicDistance()
{
  this->SetNumberOfInputPorts(1);
  this->FieldDataName = NULL;
  this->Seeds = NULL;
}

//-----------------------------------------------------------------------------
vtkPolyDataGeodesicDistance::~vtkPolyDataGeodesicDistance()
{
  this->SetFieldDataName(NULL);
  this->SetSeeds(NULL);
}

//-----------------------------------------------------------------------------
vtkFloatArray *vtkPolyDataGeodesicDistance
::GetGeodesicDistanceField(vtkPolyData *pd)
{
  if (this->FieldDataName == NULL)
    {
    return NULL;
    }

  vtkDataArray *arr = pd->GetPointData()->GetArray(this->FieldDataName);
  if (vtkFloatArray *farr = vtkFloatArray::SafeDownCast(arr))
    {
    // Resize the existing one
    farr->SetNumberOfValues(pd->GetNumberOfPoints());
    if (!pd->GetPointData()->GetScalars())
      {
      pd->GetPointData()->SetScalars(farr);
      }
    return farr;
    }
  else if (!arr)
    {
    // Create a new one
    vtkFloatArray *farray = vtkFloatArray::New();
    farray->SetName(this->FieldDataName);
    farray->SetNumberOfValues(pd->GetNumberOfPoints());
    pd->GetPointData()->AddArray(farray);
    farray->Delete();
    if (!pd->GetPointData()->GetScalars())
      {
      pd->GetPointData()->SetScalars(farray);
      }
    return vtkFloatArray::SafeDownCast(
      pd->GetPointData()->GetArray(this->FieldDataName));
    }
  else
    {
    vtkErrorMacro( << "A array with a different datatype already exists with the same name on this polydata" );
    }

  return NULL;
}

//-----------------------------------------------------------------------------
int vtkPolyDataGeodesicDistance::Compute()
{
  if (!this->Seeds || !this->Seeds->GetNumberOfIds())
    {
    vtkErrorMacro( << "Please supply at least one seed." );
    return 0;
    }

  return 1;
}

//----------------------------------------------------------------------------
unsigned long vtkPolyDataGeodesicDistance::GetMTime()
{
  unsigned long mTime = this->Superclass::GetMTime(), time;

  if ( this->Seeds )
    {
    time = this->Seeds->GetMTime();
    mTime = ( time > mTime ? time : mTime );
    }
  return mTime;
}

//-----------------------------------------------------------------------------
void vtkPolyDataGeodesicDistance::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  if (this->Seeds)
    {
    os << indent << "Seeds: " << this->Seeds << endl;
    this->Seeds->PrintSelf(os, indent.GetNextIndent());
    }
  os << indent << "FieldDataName: "
     << (this->FieldDataName ? this->FieldDataName : "None") << endl;
}

