#include <vtkSortDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkSmartPointer.h>
#include <vtkIdList.h>
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkVectorDot.h>

int main(int, char *[])
{

  // Add normals
  vtkSmartPointer<vtkFloatArray> normals =
    vtkSmartPointer<vtkFloatArray>::New();
  normals->SetNumberOfComponents(3);
  normals->SetName("Normals");
 
  float n0[3] = {1,0,0};
  float n1[3] = {1,0,0};
  float n2[3] = {1,0,0};
  normals->InsertNextTupleValue(n0);
  normals->InsertNextTupleValue(n1);
  normals->InsertNextTupleValue(n2);

 vtkSmartPointer<vtkIdList> idList =
      vtkSmartPointer<vtkIdList>::New();
    idList->InsertNextId(2);
    idList->InsertNextId(1);
    idList->InsertNextId(3);

  vtkSmartPointer<vtkDoubleArray> keyArray =
    vtkSmartPointer<vtkDoubleArray>::New();
  keyArray->InsertNextValue(1.1);
  keyArray->InsertNextValue(0.5);
  keyArray->InsertNextValue(3.0);

  std::cout << "Unsorted: " << idList->GetId(0) << " "
            << idList->GetId(1) << " "
            << idList->GetId(2) << std::endl;

  // Sort the array
  vtkSmartPointer<vtkSortDataArray> sortDataArray =
    vtkSmartPointer<vtkSortDataArray>::New();
  sortDataArray->Sort(keyArray, idList);

  std::cout << "sorted: " << idList->GetId(0) << " "
            << idList->GetId(1) << " "
            << idList->GetId(2) << std::endl;

  return EXIT_SUCCESS;
}
