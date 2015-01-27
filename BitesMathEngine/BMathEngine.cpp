//
//  BMathEngine.cpp
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 09/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#include "BMathEngine.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////

// KERNEL code

const char *KernelSource = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"#define PI 3.14159265358979323846                                      \n" \
"                                                                       \n" \
"__kernel void add(                                                     \n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input1[i] + input2[i];                              \n" \
"}                                                                      \n" \
"__kernel void multiply(                                                \n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input1[i] * input2[i];                              \n" \
"}                                                                      \n" \
"__kernel void divide(                                                  \n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input1[i] / input2[i];                              \n" \
"}                                                                      \n" \
"__kernel void subtract(                                                \n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input1[i] - input2[i];                              \n" \
"}                                                                      \n" \
"__kernel void powArr(                                                  \n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = pow(input1[i],input2[i]);                           \n" \
"}                                                                      \n" \
"__kernel void conv(                                                    \n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   __local float* local_result,                                        \n" \
"   const unsigned int count2) {                                        \n" \
"    float sum;                                                         \n" \
"                                                                       \n" \
"    int n = get_group_id(0);                                           \n" \
"    int k = get_local_id(0);                                           \n" \
"                                                                       \n" \
"    if((n-k)>=0 && (n-k)<count2){                                      \n" \
"        local_result[k] = input1[k]*input2[n-k];                       \n" \
"    }                                                                  \n" \
"                                                                       \n" \
"                                                                       \n" \
"    barrier(CLK_LOCAL_MEM_FENCE);                                      \n" \
"                                                                       \n" \
"    if(get_local_id(0) == 0) {                                         \n" \
"        sum = 0.0f;                                                    \n" \
"        for(int i=0; i<get_local_size(0); i++) {                       \n" \
"            sum += local_result[i];                                    \n" \
"        }                                                              \n" \
"        output[get_group_id(0)] = sum;                                 \n" \
"    }                                                                  \n" \
"}                                                                      \n" \
"__kernel void bitReverse(                                              \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   unsigned int position = get_global_id(0);                           \n" \
"   unsigned int target = 0;                                            \n" \
"   unsigned int counter = count-1;                                     \n" \
"   unsigned int temp;                                                  \n" \
"   while(counter){                                                     \n" \
"       counter = counter>>1;                                           \n" \
"       target = target<<1;                                             \n" \
"       temp = position&1;                                              \n" \
"       target+=temp;                                                   \n" \
"       position = position>>1;                                         \n" \
"    }                                                                  \n" \
"   position = get_global_id(0);                                        \n" \
"   output[target] = input[position];                                   \n" \
"}                                                                      \n" \
"__kernel void fft(                                                     \n" \
"   __global float* inputRe,                                            \n" \
"   __global float* inputIm,                                            \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"                                                                       \n" \
"                                                                       \n" \
"}                                                                      \n" \
"__kernel void dft(                                                     \n" \
"   __global float* inputRe,                                            \n" \
"   __global float* inputIm,                                            \n" \
"   __global float* outputRe,                                           \n" \
"   __global float* outputIm,                                           \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   unsigned int position = get_global_id(0);                           \n" \
"   float realSum = 0;                                                  \n" \
"   float imagSum = 0;                                                  \n" \
"   int i;                                                              \n" \
"   for (i = 0; i < count; i++) {                                       \n" \
"       float angle = 2 * PI * i * position / count;                    \n" \
"       realSum +=  inputRe[i] * cos(angle) + inputIm[i] * sin(angle);  \n" \
"       imagSum += -inputRe[i] * sin(angle) + inputIm[i] * cos(angle);  \n" \
"   }                                                                   \n" \
"   outputRe[position] = realSum;                                       \n" \
"   outputIm[position] = imagSum;                                       \n" \
"                                                                       \n" \
"}                                                                      \n" \
"__kernel void idft(                                                    \n" \
"   __global float* inputRe,                                            \n" \
"   __global float* inputIm,                                            \n" \
"   __global float* outputRe,                                           \n" \
"   __global float* outputIm,                                           \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   unsigned int position = get_global_id(0);                           \n" \
"   float realSum = 0;                                                  \n" \
"   float imagSum = 0;                                                  \n" \
"   int i;                                                              \n" \
"   for (i = 0; i < count; i++) {                                       \n" \
"       float angle = -2 * PI * i * position / count;                   \n" \
"       realSum +=  inputRe[i] * cos(angle) + inputIm[i] * sin(angle);  \n" \
"       imagSum += -inputRe[i] * sin(angle) + inputIm[i] * cos(angle);  \n" \
"   }                                                                   \n" \
"   outputRe[position] = realSum/count;                                 \n" \
"   outputIm[position] = imagSum/count;                                 \n" \
"                                                                       \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

BMathEngine::BMathEngine()
{
    this->global_size = 0;
    this->local_size = 0;
    /* Create device and context */
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
}
BMathEngine::~BMathEngine()
{
    clReleaseContext(context);
}

bool BMathEngine::getConnected()
{
    /* Build program */
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        errMessage = "Error: Failed to create compute program!";
        //printf("Error: Failed to create compute program!\n");
        //exit(1);
        return false;
    }
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        //GERI DON printf icin
        
        errMessage = "Error: Failed to build program executable!";
        //perror("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        //exit(1);
        return false;
    }
    
    /* Create a command queue */
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if(err < 0) {
        errMessage = "Couldn't create a command queue";
        //perror("Couldn't create a command queue");
        //exit(1);
        return false;
    }
    errMessage="";
    return true;

};



void BMathEngine::setGlobalSize(size_t globalSize)
{
    this->global_size=globalSize;
}

void BMathEngine::setLocalSize(size_t localSize)
{
    this->local_size=localSize;
}

bool BMathEngine::setKernel(char funcName[])
{
    kernel = clCreateKernel(program,funcName, &err);
    if(err < 0) {
        errMessage = "Couldn't create a kernel";
        //perror("Couldn't create a kernel");
        //exit(1);
        return false;
    }
    this->currFuncName = funcName;
    errMessage="";
    return true;
    
}

bool BMathEngine::runKernel1D(float &input1, float &input2, float &output, int count)
{
    /* Create data buffer */
   
    
    input_buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR, this->input1Size * sizeof(float), &input1, &err);
    input_buffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR, this->input2Size * sizeof(float), &input2, &err);
    output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
                                   CL_MEM_COPY_HOST_PTR, this->outputSize * sizeof(float), &output, &err);
    if(err < 0) {
        errMessage = "Couldn't create a buffer";
        return false;
    };
    
    
  
    
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
    if (strcmp(this->currFuncName,"conv")==0) {
        err |= clSetKernelArg(kernel, 3, local_size * sizeof(float), NULL);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &this->input2Size);
    }else{
        err |= clSetKernelArg(kernel, 3, sizeof(int), &this->input2Size);

    }
    if (err != CL_SUCCESS)
    {
        errMessage = "Error: Failed to set kernel arguments!";
        return false;
    }
    
    //Run Kernel
    if (local_size == 0) {
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size,
                                     NULL, 0, NULL, NULL);
    }else{
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL);
    }
    
    
        
    if(err < 0) {
        errMessage = "Couldn't enqueue the kernel";
        return false;
    }
    
    /* Read the kernel's output */
    err = clEnqueueReadBuffer(commands, output_buffer, CL_TRUE, 0,
                              sizeof(float)*count, &output, 0, NULL, NULL);
    if(err < 0) {
        errMessage = "Couldn't read the buffer";
        return false;
    }
    
    /*for (int i=0; i<count; i++) {
        cout<<i<<"    "<<*(&output+i)<<endl;
    }*/

    //Deallocation
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer1);
    clReleaseMemObject(input_buffer2);
    
    errMessage="";
    return true;
    
}
bool BMathEngine::runKernel1D(float &input1, float &input2, float &output1, float &output2, int count)
{
    /* Create data buffer */
    
    
    input_buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR, this->input1Size * sizeof(float), &input1, &err);
    input_buffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR, this->input2Size * sizeof(float), &input2, &err);
    output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
                                   CL_MEM_COPY_HOST_PTR, this->outputSize * sizeof(float), &output1, &err);
    output_buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE |
                                   CL_MEM_COPY_HOST_PTR, this->outputSize * sizeof(float), &output2, &err);
    if(err < 0) {
        errMessage = "Couldn't create a buffer";
        return false;
    }
    
    
    
    
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_buffer2);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &this->input2Size);
    if (err != CL_SUCCESS)
    {
        errMessage = "Error: Failed to set kernel arguments!";
        return false;
    }
    
    //Run Kernel
    if (local_size == 0) {
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size,
                                     NULL, 0, NULL, NULL);
    }else{
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL);
    }
    
    
    
    if(err < 0) {
        errMessage = "Couldn't enqueue the kernel";
        return false;
    }
    
    /* Read the kernel's output */
    err = clEnqueueReadBuffer(commands, output_buffer, CL_TRUE, 0,
                              sizeof(float)*count, &output1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(commands, output_buffer2, CL_TRUE, 0,
                              sizeof(float)*count, &output2, 0, NULL, NULL);
    if(err < 0) {
        errMessage = "Couldn't read the buffer";
        return false;
    }
   /* cout<<"HERE"<<endl;
    for (int i=0; i<count; i++) {
     cout<<i<<"    "<<*(&output1+i)<<endl;
     }
    for (int i=0; i<count; i++) {
     cout<<i<<"    "<<*(&output2+i)<<endl;
     }*/
    
    //Deallocation
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(output_buffer2);
    clReleaseMemObject(input_buffer1);
    clReleaseMemObject(input_buffer2);
    
    
    errMessage="";
    return true;
    
}

bool BMathEngine::runKernel1D(float &input, float &output, int count)
{
    /* Create data buffer */
    
    
    input_buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR, this->input1Size * sizeof(float), &input, &err);
    
    output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
                                   CL_MEM_COPY_HOST_PTR, this->outputSize * sizeof(float), &output, &err);
   
    if(err < 0) {
        errMessage = "Couldn't create a buffer";
        return false;
    }
    
    
    
    
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &this->input1Size);
        
    
    if (err != CL_SUCCESS)
    {
        errMessage = "Error: Failed to set kernel arguments!";
        return false;
    }
    
    //Run Kernel
    if (local_size == 0) {
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size,
                                     NULL, 0, NULL, NULL);
    }else{
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL);
    }
    
    
    
    if(err < 0) {
        errMessage = "Couldn't enqueue the kernel";
        return false;
    }
    
    /* Read the kernel's output */
    //Notice that not only output but also input is updated,
    //this is because fft function will assume that input is the real part
    //and since it is in-place, it will modify it, as it modifies the output,
    //the imaginary part.
    //BUT these are not coded yet. See kernel's fft function.
    err = clEnqueueReadBuffer(commands, output_buffer, CL_TRUE, 0,
                              sizeof(float)*count, &output, 0, NULL, NULL);
    err = clEnqueueReadBuffer(commands, input_buffer1, CL_TRUE, 0,
                              sizeof(float)*count, &input, 0, NULL, NULL);
    if(err < 0) {
        errMessage = "Couldn't read the buffer";
        return false;
    }
    
    //for (int i=0; i<count; i++) {
    //    cout<<i<<"    "<<*(&output+i)<<endl;
    //}
    
    //Deallocation
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer1);
    
    errMessage="";
    return true;
    
}



void BMathEngine::deallocResources(){
    this->input1Size=0;
    this->input2Size=0;
    this->outputSize=0;
    this->local_size=0;
    this->global_size=0;
    this->currFuncName=NULL;
    /* Deallocate resources */
    clReleaseKernel(kernel);

    clReleaseCommandQueue(commands);
    clReleaseProgram(program);
    
}


bool BMathEngine::add(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output)
{
    if (input1.getSize() != input2.getSize() || input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input2.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "add";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        temp = runKernel1D(input1.getData(), input2.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        deallocResources();
        errMessage="";
        return true;
    }
}
bool BMathEngine::add(const BFloat32Array &input1, const float input2, BFloat32Array &output)
{
    if (input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input1.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "add";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        BFloat32Array input2Arr (size);
        input2Arr.fill(input2);
        temp = runKernel1D(input1.getData(), input2Arr.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
}

bool BMathEngine::subtract(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output)
{
    if (input1.getSize() != input2.getSize() || input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input2.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }

        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "subtract";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }

        temp = runKernel1D(input1.getData(), input2.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}

bool BMathEngine::subtract(const BFloat32Array &input1, const float input2, BFloat32Array &output)
{
    if (input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input1.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "subtract";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        BFloat32Array input2Arr (size);
        input2Arr.fill(input2);
        temp = runKernel1D(input1.getData(), input2Arr.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }

        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}
bool BMathEngine::multiply(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output)
{
    if (input1.getSize() != input2.getSize() || input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input2.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "multiply";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        temp = runKernel1D(input1.getData(), input2.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}
bool BMathEngine::multiply(const BFloat32Array &input1, const float input2, BFloat32Array &output)
{
    if (input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input1.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "multiply";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        BFloat32Array input2Arr (size);
        input2Arr.fill(input2);
        temp = runKernel1D(input1.getData(), input2Arr.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}
bool BMathEngine::divide(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output)
{
    if (input1.getSize() != input2.getSize() || input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{//TODO ZERO CHECK EKLEMEK GEREKMEZ MI?
        this->input1Size=input1.getSize();
        this->input2Size=input2.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "divide";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        temp = runKernel1D(input1.getData(), input2.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}
bool BMathEngine::divide(const BFloat32Array &input1, const float input2, BFloat32Array &output)
{
    if (input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }if(input2 == 0){
        errMessage = "Can't divide by zero!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input1.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "divide";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        
        BFloat32Array input2Arr (size);
        input2Arr.fill(input2);
        temp = runKernel1D(input1.getData(), input2Arr.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}

bool BMathEngine::pow(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output)
{
    if (input1.getSize() != input2.getSize() || input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input2.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "powArr";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        temp = runKernel1D(input1.getData(), input2.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}
bool BMathEngine::pow(const BFloat32Array &input1, const float input2, BFloat32Array &output)
{
    if (input1.getSize() != output.getSize()) {
        errMessage = "Size mismatch!";
        return false;
    }else{
        this->input1Size=input1.getSize();
        this->input2Size=input1.getSize();
        this->outputSize=output.getSize();
        bool temp = getConnected();
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't get connected!";
            return false;
        }
        
        int size = input1.getSize();
        setGlobalSize(size);
        setLocalSize(0);
        char func[] = "powArr";
        temp = setKernel(func);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't set kernel!";
            return false;
        }
        
        BFloat32Array input2Arr (size);
        input2Arr.fill(input2);
        temp = runKernel1D(input1.getData(), input2Arr.getData(), output.getData(), size);
        if (temp==false) {
            cout<<errMessage<<endl;
            errMessage = "Couldn't run kernel!";
            return false;
        }
        
        deallocResources();
        errMessage="";
        return true;
    }
    errMessage="";
    return true;
}

/*
 * Output array's size doesn't have to match with anything here,
 * since user may not be able to calculate it,
 * the code reinitializes the output's array. So output can be any size. No problem.
 */
bool BMathEngine::conv(const BSignal &input1, const BSignal &input2, BSignal &output)
{
    output.arrb=input1.arrb+input2.arrb;
    this->input1Size = (*input1.arr).getSize();
    this->input2Size = (*input2.arr).getSize();
    this->outputSize = input1Size+input2Size-1;
    bool temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }
    
    setGlobalSize(outputSize*input1Size);
    setLocalSize(input1Size);
    
    char func[] = "conv";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    
 
    output.arr = new BFloat32Array(outputSize);

    temp = runKernel1D((*input1.arr).getData(), (*input2.arr).getData(), (*output.arr).getData(), outputSize);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
    
    deallocResources();

    errMessage="";
    return true;
}

bool BMathEngine::conv(const BFloat32Array &input1, const BFloat32Array &input2, BFloat32Array &output)
{
    int outSignalSize = input1.getSize() + input2.getSize() - 1;
    this->input1Size=input1.getSize();
    this->input2Size=input2.getSize();
    this->outputSize=outSignalSize;
    
    output = *new BFloat32Array(outSignalSize);
    
    bool temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }
    
    setGlobalSize(outSignalSize*input1Size);
    setLocalSize(input1Size);
  
    char func[] = "conv";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    
    temp = runKernel1D(input1.getData(), input2.getData(), output.getData(), outSignalSize);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
    deallocResources();
    
    
    
    errMessage="";
    return true;
}
void BMathEngine::printMessage()
{
    cout<<this->errMessage<<endl;
}
/*
 *Make sure that input array has a length 
 *which is a power of 2. Since this function is used
 *for the fft algorithm.
 */
bool BMathEngine::bitReverse(const BFloat32Array &input, BFloat32Array &output)
{
    this->input1Size=input.getSize();
    this->outputSize=output.getSize();
    bool temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }
    
    int size = input1Size;
    setGlobalSize(size);
    setLocalSize(0);
    char func[] = "bitReverse";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    
    temp = runKernel1D(input.getData(), output.getData(), size);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
   
    
    deallocResources();

    errMessage="";
    return true;
}


bool BMathEngine::fft(const BFloat32Array &input, BComplex32Array &output)
{
    //
    //FIRST
    //We should make sure that input's size
    //is a power of two.
    
    //SECOND
    //Reorder the input array
    //store it into the output array.
    //

    BFloat32Array outputRe(input.getSize());
    BFloat32Array outputIm(input.getSize());
    outputIm.fill(0);
   
    bool temp = bitReverse(input,outputRe);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't achieve bit reverse operation!";
        return false;
    }
    
    for (int i=0; i<8; i++) {
        output[i]=outputRe[i];
        cout<<output[i].re()<<endl;
    }
    
    //THIRD
    //Perform fft
    //
    this->input1Size=input.getSize();
    this->input2Size=0;
    this->outputSize=output.getSize();
    temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }
    
    int size = input1Size;
    setGlobalSize(size);
    setLocalSize(0);
    //WARNING: function fft is not coded properly in the kernel yet.
    char func[] = "fft";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    //Which runKernel1D should be used here is uncertain right now.
    //Since it's incomplete on the kernel side
    temp = runKernel1D(outputRe.getData(), outputIm.getData(), input1Size);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
    

    for (int i=0; i<outputSize; i++) {
        output[i].setRe(outputRe[i]);
        output[i].setIm(outputIm[i]);
    }

    deallocResources();
    errMessage="";
    return true;
}
bool BMathEngine::dft(const BFloat32Array &input, BComplex32Array &output)
{
    BFloat32Array outputRe(input.getSize());
    BFloat32Array outputIm(input.getSize());
    outputIm.fill(0);
    outputRe.fill(0);
    BFloat32Array inputIm(input.getSize());
    inputIm.fill(0);
    
    // Perform dft
    this->input1Size=input.getSize();
    this->input2Size=input.getSize();
    this->outputSize=output.getSize();
    bool temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }

    int size = input1Size;
    setGlobalSize(size);
    setLocalSize(0);
    char func[] = "dft";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    
    temp = runKernel1D(input.getData(),inputIm.getData(), outputRe.getData(), outputIm.getData(), input1Size);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
    
    
    for (int i=0; i<outputSize; i++) {
        output[i].setRe(outputRe[i]);
        output[i].setIm(outputIm[i]);
    }
    deallocResources();
    errMessage="";
    return true;
}

bool BMathEngine::dft(const BComplex32Array &input, BComplex32Array &output)
{
    BFloat32Array outputRe(input.getSize());
    BFloat32Array outputIm(input.getSize());
    outputIm.fill(0);
    outputRe.fill(0);
    BFloat32Array inputRe(input.getSize());
    BFloat32Array inputIm(input.getSize());
    for (int i=0; i<input.getSize(); i++) {
        inputRe[i]=input[i].re();
        inputIm[i]=input[i].im();
    }
    
    // Perform dft
    this->input1Size=input.getSize();
    this->input2Size=input.getSize();
    this->outputSize=output.getSize();
    bool temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }
    
    int size = input1Size;
    setGlobalSize(size);
    setLocalSize(0);
    char func[] = "dft";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    
    
    temp = runKernel1D(inputRe.getData(), inputIm.getData(), outputRe.getData(), outputIm.getData(), input1Size);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
    
    for (int i=0; i<outputSize; i++) {
        output[i].setRe(outputRe[i]);
        //cout<<outputRe[i]<<"    "<<outputIm[i]<<endl;
        output[i].setIm(outputIm[i]);
    }
    deallocResources();
    errMessage="";
    return true;
}
bool BMathEngine::idft(const BFloat32Array &input, BComplex32Array &output)
{
    
    
    BFloat32Array outputRe(input.getSize());
    BFloat32Array outputIm(input.getSize());
    outputIm.fill(0);
    outputRe.fill(0);
    BFloat32Array inputIm(input.getSize());
    inputIm.fill(0);
    
    // Perform dft
    this->input1Size=input.getSize();
    this->input2Size=input.getSize();
    this->outputSize=output.getSize();
    bool temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }
    
    int size = input1Size;
    setGlobalSize(size);
    setLocalSize(0);
    char func[] = "idft";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    
    temp = runKernel1D(input.getData(),inputIm.getData(), outputRe.getData(), outputIm.getData(), input1Size);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
    
    
    for (int i=0; i<outputSize; i++) {
        output[i].setRe(outputRe[i]);
        output[i].setIm(outputIm[i]);
    }
    deallocResources();
    errMessage="";
    return true;
}

bool BMathEngine::idft(const BComplex32Array &input, BComplex32Array &output)
{
    BFloat32Array outputRe(input.getSize());
    BFloat32Array outputIm(input.getSize());
    outputIm.fill(0);
    outputRe.fill(0);
    BFloat32Array inputRe(input.getSize());
    BFloat32Array inputIm(input.getSize());
    for (int i=0; i<input.getSize(); i++) {
        inputRe[i]=input[i].re();
        inputIm[i]=input[i].im();
    }
    
    // Perform dft
    this->input1Size=input.getSize();
    this->input2Size=input.getSize();
    this->outputSize=output.getSize();
    bool temp = getConnected();
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't get connected!";
        return false;
    }
    
    int size = input1Size;
    setGlobalSize(size);
    setLocalSize(0);
    char func[] = "idft";
    temp = setKernel(func);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't set kernel!";
        return false;
    }
    
    
    temp = runKernel1D(inputRe.getData(), inputIm.getData(), outputRe.getData(), outputIm.getData(), input1Size);
    if (temp==false) {
        cout<<errMessage<<endl;
        errMessage = "Couldn't run kernel!";
        return false;
    }
    
    for (int i=0; i<outputSize; i++) {
        output[i].setRe(outputRe[i]);
        //cout<<outputRe[i]<<"    "<<outputIm[i]<<endl;
        output[i].setIm(outputIm[i]);
    }
    deallocResources();
    errMessage="";
    return true;
}
/*
 * Chirp z-transform algorithm (1969), is a fast Fourier transform (FFT) algorithm
 * that computes the discrete Fourier transform (DFT) of arbitrary sizes 
 * (including prime sizes) by re-expressing the DFT as a convolution.
 *
 */
bool BMathEngine::chirpZTransform()
{
    
    errMessage="";
    return true;
}
