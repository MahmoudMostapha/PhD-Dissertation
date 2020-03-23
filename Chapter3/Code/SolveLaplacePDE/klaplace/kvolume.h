//
//  kvolume.h
//  ktools
//
//  Created by Joowhi Lee on 9/3/15.
//
//

#ifndef __ktools__kvolume__
#define __ktools__kvolume__

#include <stdio.h>
#include "piOptions.h"

class vtkDataSet;
class vtkPolyData;

vtkDataSet* createGridForSphereLikeObject(vtkPolyData* input, int& insideCount, int dims = 100, bool insideOutOn = false);
void runExtractBorderline(pi::Options& opts, pi::StringVector& args);
void runFillGrid(pi::Options& opts, pi::StringVector& args);

void processVolumeOptions(pi::Options& opts);
void processVolumeCommands(pi::Options& opts, pi::StringVector& args);

#endif /* defined(__ktools__kvolume__) */
