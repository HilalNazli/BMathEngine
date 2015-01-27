//
//  BComplex.h
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 18/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#ifndef __BitesMathEngine__BComplex__
#define __BitesMathEngine__BComplex__

class BComplex
{
private:
    float mRe;
    float mIm;
public:
    BComplex();
    BComplex(float pRe, float pIm);
    float re() const;
    float im() const;
    void setRe(float pRe);
    void setIm(float pIm);
    BComplex operator+ (const BComplex& operand) const;
	BComplex operator- (const BComplex& operand) const;
	BComplex operator* (const BComplex& operand) const;
	BComplex operator/ (const BComplex& operand) const;
    BComplex& operator= (const float rhs);
    BComplex& operator= (const BComplex &rhs);
    
    
protected:

};

#endif /* defined(__BitesMathEngine__BComplex__) */
