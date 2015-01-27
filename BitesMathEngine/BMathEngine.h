//
//  BMathEngine.h
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 09/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#ifndef __BitesMathEngine__BMathEngine__
#define __BitesMathEngine__BMathEngine__

#include <iostream>

#include "BFloat32Array.h"
#include "BComplex32Array.h"
#include "BSignal.h"

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif


class BMathEngine{
public:
    BMathEngine();
    bool add(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output);
    bool add(const BFloat32Array &input1, const float input2, BFloat32Array &output);
    bool subtract(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output);
    bool subtract(const BFloat32Array &input1, const float input2, BFloat32Array &output);
    bool multiply(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output);
    bool multiply(const BFloat32Array &input1, const float input2, BFloat32Array &output);
    bool divide(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output);
    bool divide(const BFloat32Array &input1, const float input2, BFloat32Array &output);
    bool pow(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output);
    bool pow(const BFloat32Array &input1, const float input2, BFloat32Array &output);
    bool conv(const BFloat32Array &input1, const BFloat32Array &input2,BFloat32Array &output);
    bool conv(const BSignal &input1, const BSignal &input2, BSignal &output);
    void printMessage();
    bool fft(const BFloat32Array &input, BComplex32Array &output);
    bool dft(const BFloat32Array &input, BComplex32Array &output);
    bool dft(const BComplex32Array &input, BComplex32Array &output);
    bool idft(const BFloat32Array &input, BComplex32Array &output);
    bool idft(const BComplex32Array &input, BComplex32Array &output);
    bool chirpZTransform();

    ~BMathEngine();
private:
    /* OpenCL structures */
    cl_device_id device_id; //Will be a list in the future
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue commands;
    cl_int err;
    size_t local_size, global_size;
    char *currFuncName;
    cl_mem input_buffer1, input_buffer2, output_buffer, output_buffer2;
    std::string errMessage;
    unsigned int input1Size, input2Size, outputSize;
    bool getConnected();
    bool setKernel(char funcName[]);//Which function in kernel
    void setGlobalSize(size_t globalSize);
    void setLocalSize(size_t localSize);
    bool runKernel1D(float &input1, float &input2, float &output, int count);
    bool runKernel1D(float &input1, float &input2, float &output1, float &output2, int count);
    bool runKernel1D(float &input, float &output, int count);
    bool bitReverse(const BFloat32Array &input, BFloat32Array &output);
    void deallocResources();
    
protected:


};



#endif /* defined(__BitesMathEngine__BMathEngine__) */
