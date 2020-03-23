#!/usr/bin/env python

import sys
import os
import subprocess
import os.path
from os import path

Data_Folder = '/proj/NIRAL/users/mahmoud/Data/EACSF_6M'

ID = int(sys.argv[1])

LH = int(sys.argv[2])

RH = int(sys.argv[3])

subprocess.call('echo starting ' + str(ID), shell=True)

os.chdir(os.path.join(Data_Folder, str(ID)))

cwd = os.getcwd()

print(cwd)

if LH:

	#LH

	os.system('../Code/CreateOuterImage/bin/CreateOuterImage ' + str(ID) + '_Seg_mask.nrrd ' + str(ID) + '_LH_GM_Dilated.nrrd 15 3')

	subprocess.call('echo starting LH', shell=True)

	os.system('../Code/CreateOuterSurface/bin/CreateOuterSurface ' +  str(ID) + '_LH_GM_Dilated.nrrd ' + str(ID) + '_LH_GM_Outer_MC.vtk 1')

	os.system('../Code/EditPolyData/build/EditPolyData ' + str(ID) + '_LH_GM_Outer_MC.vtk ' + str(ID) + '_LH_GM_Outer_MC.vtk -1 -1 1')

	subprocess.call('echo Creating LH streamlines', shell=True)

	subprocess.call('echo Establishing Surface Correspondance', shell=True)

	os.system('../Code/SolveLaplacePDE/klaplace-build/klaplace -dims 300 ' + str(ID) + '_LH_MID.vtk '  + str(ID) + '_LH_GM_Outer_MC.vtk  -surfaceCorrespondence ' + str(ID) + '_LH_Outer.corr')

	subprocess.call('Establishing Streamlines', shell=True)

	os.system('../Code/SolveLaplacePDE/klaplace-build/klaplace -traceStream ' + str(ID) +  '_LH_Outer.corr_field.vts ' + str(ID) + '_LH_MID.vtk ' + str(ID) + '_LH_GM_Outer_MC.vtk ' + str(ID) + '_LH_Outer_streamlines.vtp ' + str(ID) + '_LH_Outer_points.vtp -traceDirection forward')

	os.system('../Code/SolveLaplacePDE/klaplace-build/klaplace -conv ' + str(ID) + '_LH_Outer_streamlines.vtp ' + str(ID) + '_LH_Outer_streamlines.vtk')

	subprocess.call('echo Computing LH EACSF', shell=True)

	os.system('../Code/EstimateLocalEACSF/build/EstimateCortexStreamlinesDensity ' + str(ID) + '_LH_MID.vtk ' + str(ID) + '_LH_Outer_streamlines.vtk ' + str(ID) + '_CSF_Prop_Masked.nrrd ' + str(ID) + '_LH_GM_Dilated.nrrd ' + str(ID) + '_LH_CSF_Density.vtk ' + str(ID) + '_LH_Visitation.nrrd 0 0 0')

	os.system('/proj/NIRAL/tools/MeshMath ' + str(ID) + '_LH_GM.vtk ' + str(ID) + '_LH_GM.vtk -KWMtoPolyData ' + str(ID) + '_LH_MID.CSFDensity.txt EACSF')

	os.system('mv ' + str(ID) + '_LH_MID.CSFDensity.txt ' + str(ID) + '_LH_MID.CSFDensity_All.txt ')

	os.system('../Code/EstimateLocalEACSF/build/EstimateCortexStreamlinesDensity ' + str(ID) + '_LH_MID.vtk ' + str(ID) + '_LH_Outer_streamlines.vtk ' + str(ID) + '_CSF_Prop_Masked_EA.nrrd ' + str(ID) + '_LH_GM_Dilated.nrrd ' + str(ID) + '_LH_CSF_Density.vtk ' + str(ID) + '_LH_Visitation.nrrd 0 0 0')

	os.system('/proj/NIRAL/tools/MeshMath ' + str(ID) + '_LH_GM.vtk ' + str(ID) + '_LH_GM.vtk -KWMtoPolyData ' + str(ID) + '_LH_MID.CSFDensity.txt EACSF_EA')

	os.system('mv ' + str(ID) + '_LH_MID.CSFDensity.txt ' + str(ID) + '_LH_MID.CSFDensity_EA.txt ')

	subprocess.call('rm *.vtp', shell=True)

if RH:

	#RH

	os.system('../Code/CreateOuterImage/bin/CreateOuterImage ' + str(ID) + '_Seg_mask.nrrd ' + str(ID) + '_RH_GM_Dilated.nrrd 15 3 1')

	subprocess.call('echo starting RH', shell=True)

	os.system('../Code/CreateOuterSurface/bin/CreateOuterSurface ' +  str(ID) + '_RH_GM_Dilated.nrrd ' + str(ID) + '_RH_GM_Outer_MC.vtk 1')

	os.system('../Code/EditPolyData/build/EditPolyData ' + str(ID) + '_RH_GM_Outer_MC.vtk ' + str(ID) + '_RH_GM_Outer_MC.vtk -1 -1 1')

	subprocess.call('echo Creating RH streamlines', shell=True)

	subprocess.call('echo Establishing Surface Correspondance', shell=True)

	os.system('../Code/SolveLaplacePDE/klaplace-build/klaplace -dims 300 ' + str(ID) + '_RH_MID.vtk '  + str(ID) + '_RH_GM_Outer_MC.vtk  -surfaceCorrespondence ' + str(ID) + '_RH_Outer.corr')

	subprocess.call('Establishing Streamlines', shell=True)

	os.system('../Code/SolveLaplacePDE/klaplace-build/klaplace -traceStream ' + str(ID) +  '_RH_Outer.corr_field.vts ' + str(ID) + '_RH_MID.vtk ' + str(ID) + '_RH_GM_Outer_MC.vtk ' + str(ID) + '_RH_Outer_streamlines.vtp ' + str(ID) + '_RH_Outer_points.vtp -traceDirection forward')

	os.system('../Code/SolveLaplacePDE/klaplace-build/klaplace -conv ' + str(ID) + '_RH_Outer_streamlines.vtp ' + str(ID) + '_RH_Outer_streamlines.vtk')

	subprocess.call('echo Computing RH EACSF', shell=True)

	os.system('../Code/EstimateLocalEACSF/build/EstimateCortexStreamlinesDensity ' + str(ID) + '_RH_MID.vtk ' + str(ID) + '_RH_Outer_streamlines.vtk ' + str(ID) + '_CSF_Prop_Masked.nrrd ' + str(ID) + '_RH_GM_Dilated.nrrd ' + str(ID) + '_RH_CSF_Density.vtk ' + str(ID) + '_RH_Visitation.nrrd 0 0 0')

	os.system('/proj/NIRAL/tools/MeshMath ' + str(ID) + '_RH_GM.vtk ' + str(ID) + '_RH_GM.vtk -KWMtoPolyData ' + str(ID) + '_RH_MID.CSFDensity.txt EACSF')

	os.system('mv ' + str(ID) + '_RH_MID.CSFDensity.txt ' + str(ID) + '_RH_MID.CSFDensity_All.txt ')

	os.system('../Code/EstimateLocalEACSF/build/EstimateCortexStreamlinesDensity ' + str(ID) + '_RH_MID.vtk ' + str(ID) + '_RH_Outer_streamlines.vtk ' + str(ID) + '_CSF_Prop_Masked_EA.nrrd ' + str(ID) + '_RH_GM_Dilated.nrrd ' + str(ID) + '_RH_CSF_Density.vtk ' + str(ID) + '_RH_Visitation.nrrd 0 0 0')

	os.system('/proj/NIRAL/tools/MeshMath ' + str(ID) + '_RH_GM.vtk ' + str(ID) + '_RH_GM.vtk -KWMtoPolyData ' + str(ID) + '_RH_MID.CSFDensity.txt EACSF_EA')

	os.system('mv ' + str(ID) + '_RH_MID.CSFDensity.txt ' + str(ID) + '_RH_MID.CSFDensity_EA.txt ')

	subprocess.call('rm *.vtp', shell=True)

