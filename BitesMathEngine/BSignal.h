//
//  BSignal.h
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 14/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#ifndef __BitesMathEngine__BSignal__
#define __BitesMathEngine__BSignal__

#include "BFloat32Array.h"
class BSignal
{
private:
public:
    BSignal();
    BSignal(BFloat32Array &arr, int arrb);
    BFloat32Array *arr;
    int arrb;
    
protected:
};

#endif /* defined(__BitesMathEngine__BSignal__) */
