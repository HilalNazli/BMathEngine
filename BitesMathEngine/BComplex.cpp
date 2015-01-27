//
//  BComplex.cpp
//  BitesMathEngine
//
//  Created by Hilal Özdemir on 18/07/14.
//  Copyright (c) 2014 Hilal Özdemir. All rights reserved.
//

#include "BComplex.h"

BComplex::BComplex()
{
    mRe = 0.0f;
    mIm = 0.0f;
}
BComplex::BComplex(float pRe, float pIm)
{
    mRe = pRe;
    mIm = pIm;
}
float BComplex::re() const
{
    return mRe;
}
float BComplex::im() const
{
    return mIm;
}
void BComplex::setRe(float pRe)
{
    this->mRe=pRe;
}
void BComplex::setIm(float pIm)
{
    this->mIm=pIm;
}
BComplex BComplex::operator+ (const BComplex& operand) const
{
    return BComplex(mRe + operand.re(), mIm + operand.im());
}

BComplex BComplex::operator- (const BComplex& operand) const
{
    return BComplex(mRe - operand.re(), mIm - operand.im());
}

BComplex BComplex::operator* (const BComplex& operand) const
{
    return BComplex(mRe * operand.re() - mIm * operand.im(),
                   mRe * operand.im() + mIm * operand.re());
}

BComplex BComplex::operator/ (const BComplex& operand) const
{
    return BComplex((mRe * operand.re() + mIm * operand.im()) / operand.re() * operand.re() + operand.im() * operand.im(),
                   (mIm * operand.re() - mRe * operand.im()) / operand.re() * operand.re() + operand.im() * operand.im());
}

BComplex& BComplex::operator= (const float rhs)
{
    mRe = rhs;
    mIm = 0.0f;
    return *this;
}
BComplex& BComplex::operator= (const BComplex &rhs)
{
    mRe = rhs.re();
    mIm = rhs.im();
    return *this;
}

