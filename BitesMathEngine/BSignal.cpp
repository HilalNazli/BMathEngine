//
//  BSignal.cpp
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 14/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#include "BSignal.h"

BSignal::BSignal()
{
    arr = NULL;
    arrb=0;
}
BSignal::BSignal(BFloat32Array &arr, int arrb)
{
    this->arr=&arr;
    this->arrb=arrb;
}
