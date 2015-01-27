//
//  main.cpp
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 07/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#include <iostream>
#include "BFloat32Array.h"
#include "BMathEngine.h"
#include "BSignal.h"
#include "BComplex32Array.h"



using namespace std;

int main(int argc, const char * argv[])
{
    BMathEngine engine;

 /*
    BFloat32Array arr(7);
    arr[0]=3;
    arr[1]=11;
    arr[2]=7;
    arr[3]=0;
    arr[4]=-1;
    arr[5]=4;
    arr[6]=2;
    
    BFloat32Array arr2(7);
    arr2[0]=2;
    arr2[1]=3;
    arr2[2]=0;
    arr2[3]=-5;
    arr2[4]=2;
    arr2[5]=1;
    arr2[6]=1;
    
    //If you don't know the size, just say 0.
    BFloat32Array output(7);
    
    engine.add(arr, arr2, output);
    engine.printMessage();
    
    for (int i=0; i<output.getSize(); i++) {
        cout<<output[i]<<endl;
    }
    
  */
    
   /* BFloat32Array arr(7);
    arr[0]=3;
    arr[1]=11;
    arr[2]=7;
    arr[3]=0;
    arr[4]=-1;
    arr[5]=4;
    arr[6]=2;
    
    int arrb = -3;
    
    BSignal mySignal1 (arr, arrb);
    //cout<<(*mySignal.arr)[0]<<"    "<<arrb<<endl;
    
    
    
    BFloat32Array arr2(6);
    arr2[0]=2;
    arr2[1]=3;
    arr2[2]=0;
    arr2[3]=-5;
    arr2[4]=2;
    arr2[5]=1;
    
    int arr2b = -1;
    
    BSignal mySignal2 (arr2,arr2b);

    
    //BFloat32Array *output;
    //int *outputb;
    //becomes:
    BSignal output;
 
    
    engine.conv(mySignal1, mySignal2, output);
    
    cout<<"Signal 1:"<<endl;
    for (int i=0; i<arr.getSize(); i++) {
        cout<<mySignal1.arrb+i<<"\t\t";
    }
    cout<<endl;
    for (int i=0; i<arr.getSize(); i++) {
        cout<<(*mySignal1.arr)[i]<<"\t\t";
    }
    cout<<endl<<endl;
    cout<<"Signal 2:"<<endl;
    for (int i=0; i<arr2.getSize(); i++) {
        cout<<mySignal2.arrb+i<<"\t\t";
    }
    cout<<endl;
    for (int i=0; i<arr2.getSize(); i++) {
        cout<<(*mySignal2.arr)[i]<<"\t\t";
    }
    cout<<endl<<endl;
    cout<<"conv(Signal 1, Signal 2) = "<<endl;
    for (int i=0; i<(*output.arr).getSize(); i++) {
        cout<<output.arrb+i<<"\t\t";
    }
    cout<<endl;
    for (int i=0; i<(*output.arr).getSize(); i++) {
        cout<<(*output.arr)[i]<<"\t\t";
    }
    
    cout<<endl;
    */
    
    
    
    //engine.conv(arr, arrb, arr2, arr2b);
    //if (err==false) {
    //   engine.printMessage();
    //}
    
    /*
    BFloat32Array arr11(3);
    arr11[0]=1;
    arr11[1]=2;
    arr11[2]=3;
    
    BFloat32Array arr22(3);
    arr22[0]=1;
    arr22[1]=2;
    arr22[2]=5;
    
    BFloat32Array output2(3);
    bool err = engine.add(arr11, arr22, output2);
    if (err==false) {
        engine.printMessage();
    }
    cout<<output2[0]<<"  "<<output2[1]<<"  "<<output2[2]<<endl;
    */
    
    /* 
    BComplex cmplx(1,1);
    cout<<cmplx.im()<<"  "<<cmplx.re()<<endl;
     */
    
    
    /*
    //Test for dft and idft function of the math engine
    BComplex32Array arrForDFT(8);
    for (int i=0; i<arrForDFT.getSize(); i++) {
        arrForDFT[i]=1;
    }
    arrForDFT[0]=0;
    arrForDFT[1]=5;
    arrForDFT[2]=0;
    arrForDFT[3]=5;
    arrForDFT[4]=0;
    arrForDFT[5]=5;
    arrForDFT[6]=0;
    arrForDFT[7]=5;
   
    BComplex32Array outputForDFT(8);
    engine.dft(arrForDFT, outputForDFT);
    
    
    for (int i=0; i<outputForDFT.getSize(); i++) {
        cout<<outputForDFT[i].re()<<endl;
    }
    
    BComplex32Array output2(8);

    engine.idft(outputForDFT, output2);
    for (int i=0; i<outputForDFT.getSize(); i++) {
        cout<<output2[i].re()<<"    "<<output2[i].im()<<endl;
    }
    */

    cout<<endl;
    cout<<endl;
    cout<<endl;
    cout<<endl;
    cout<<endl;
    cout<<endl;
    
    return 0;
}

