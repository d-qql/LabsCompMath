//
// Created by d-qql on 28.01.2021.
//

#ifndef SOLECOURSE_LU_H
#define SOLECOURSE_LU_H
#include "DenseMatrix.h"

template<typename T>
std::pair<DenseMatrix<T>, DenseMatrix<T>> LUdecomp(const DenseMatrix<T>& A){
    using idx_t = typename DenseMatrix<T>::idx_t;

    DenseMatrix<T> L = DenseMatrix<T>(A.sizeH(), A.sizeW());
    DenseMatrix<T> U = DenseMatrix<T>(A.sizeH(), A.sizeW());

    for(idx_t i = 0; i < L.sizeH(); ++i){
        L(i, i) = 1;
    }
    T sum = 0;
    for(idx_t i = 0; i < A.sizeH(); ++i){
        for(idx_t j = 0; j < A.sizeW(); ++j){
            sum = 0;
            if( i <= j ){
                for(idx_t k = 0; k < i; ++k){
                    sum += L(i, k) * U(k, j);
                }
                U(i, j) = A(i, j) - sum;
            } else {
                sum = 0;
                for(idx_t k = 0; k < j; ++k){
                    sum += L(i, k) * U(k, j);
                }
                L(i, j) = (A(i, j) - sum) / U(j, j);
            }
        }
    }
    return {L, U};
}

template<typename T>
std::vector<T> solveByLU(const DenseMatrix<T> &A, const std::vector<T> &b){
    std::pair<DenseMatrix<T>, DenseMatrix<T>> LU = LUdecomp(A);
    return backSubstTopTriangular(LU.second, backSubstLowerTriangular(LU.first, b));
}



#endif //SOLECOURSE_LU_H
