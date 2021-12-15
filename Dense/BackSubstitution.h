//
// Created by d-qql on 28.01.2021.
//

#ifndef SOLECOURSE_BACKSUBSTITUTION_H
#define SOLECOURSE_BACKSUBSTITUTION_H
#include "DenseMatrix.h"
#include <vector>

template<typename T>
std::vector<T> backSubstTopTriangular(const DenseMatrix<T> &A, const std::vector<T> &b){
    //ОБРАТНЫЙ ХОД
    std::vector<T> res;
    res.resize(b.size());
    res.back() = b.back()/A(A.sizeH()-1, A.sizeW()-1);
    T sum = 0;
    for(int i = b.size()-2; i >= 0; --i){   //проход от предпоследнего элемента свободного столбца
        sum = 0;
        for(size_t j = i+1; j < b.size(); ++j){
            sum+=A(i, j)*res[j];
        }
        res[i] = (b[i] - sum)/A(i, i);
    }
    //ОКОНЧАНИЕ ОБРАТНОГО ХОДА
    return res;
}
template<typename T>
std::vector<T> backSubstTopTriangular(const DenseMatrix<T> &A, const std::vector<T> &b, size_t H, size_t W){
    //ОБРАТНЫЙ ХОД
    std::vector<T> res;
    res.resize(b.size());
    res.back() = b.back()/A(H-1, W-1);
    T sum = 0;
    for(int i = int(H)-2; i >= 0; --i){   //проход от предпоследнего элемента свободного столбца
        sum = 0;
        for(size_t j = i+1; j < W; ++j){
            sum+=A(i, j)*res[j];
        }
        res[i] = (b[i] - sum)/A(i, i);
    }
    //ОКОНЧАНИЕ ОБРАТНОГО ХОДА
    return res;
}

template<typename T>
std::vector<T> backSubstLowerTriangular(const DenseMatrix<T> &A, const std::vector<T> &b){
    //ОБРАТНЫЙ ХОД
    std::vector<T> res;
    res.resize(b.size());
    res[0] = b[0]/A(0, 0);
    T sum = 0;
    for(int i = 1; i < A.sizeH(); ++i){
        sum = 0;
        for(int j = i-1; j >= 0; --j){
            sum+=A(i, j)*res[j];
        }
        res[i] = (b[i] - sum)/A(i, i);
    }
    //ОКОНЧАНИЕ ОБРАТНОГО ХОДА
    return res;
}

#endif //SOLECOURSE_BACKSUBSTITUTION_H
