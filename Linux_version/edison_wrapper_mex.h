

/* general include files */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* edison include files */
#include "msImageProcessor.h"


const kernelType DefaultKernelType = Uniform;
const unsigned int DefaultSpatialDimensionality = 2;

// const char * OutputFields = { "fimage", "labels", "modes", "regSize", "conf", "grad" };

bool CmCDisplayProgress = false; /* disable display promt */
