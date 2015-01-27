//
//  BArray.h
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 07/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#ifndef __BitesMathEngine__BArray__
#define __BitesMathEngine__BArray__

#include <iostream>



template <typename T>
class BArray{
    

protected:
	BArray(int pSize)
    {
        //0 initialization erased
        mData = new T[pSize];
        mSize = pSize;
    }
    
public:

    int getSize() const
    {
        return mSize;
    }
    T& getData() const
    {
        return mData[0];
    }
 
    T& operator[](int pIndex) const
    {
        if (pIndex >= 0 && pIndex < mSize) {
            return mData[pIndex];
        }else{
            char errMsg[] = "Index out of bounds!";
            printf("%s \n", errMsg);
            exit(1);
            //throw std::out_of_range("");
            /*try
            {
                throw std::out_of_range("Index out of bounds!");
            }
            catch(std::exception& e)
            {
                std::cout<<e.what()<<std::endl;
                exit(1);
            }*/
        }
    }
    void fill(T value)//Ilerde inline assembly yapilacak
    {
        for (int i=0; i<mSize; i++) {
            mData[i]=value;
        }
    }
    ~BArray() //destructor
    {
        delete [] mData;
    }
private:
    T *mData;
    int mSize;
};

#endif /* defined(__BitesMathEngine__BArray__) */
