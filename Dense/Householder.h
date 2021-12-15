//
// Created by d-qql on 04.02.2021.
//

#ifndef SOLECOURSE_HOUSEHOLDER_H
#define SOLECOURSE_HOUSEHOLDER_H
#include "DenseMatrix.h"
#include "../utility/Consts.h"
template<typename T>
std::pair<DenseMatrix<T>, DenseMatrix<T>> HouseholderReflection(DenseMatrix<T> A){
    using idx_t = typename DenseMatrix<T>::idx_t ;
    using elm_t = typename DenseMatrix<T>::elm_t ;

    std::vector<std::vector<elm_t>> V(A.sizeW()-1);
    std::vector<elm_t> x;
    std::vector<elm_t> v;
    std::vector<elm_t> e;
    x.resize(A.sizeH());
    v.resize(A.sizeH());
    e.resize(A.sizeH());
    e[0] = static_cast<elm_t>(1);
    elm_t VbyV;
    elm_t XbyV;
    for(idx_t i = 0; i < A.sizeW() - 1; ++i){
        for(idx_t k = i; k < A.sizeH(); ++k){    //формирование вектора из подстолбца
            x[k-i] = A(k, i);
        }
        v = x + sgn(x[0]) * norm(x) * e;
        VbyV = v * v;
        for(idx_t j = i; j < A.sizeW(); ++j){ //проход по всем подстолбцам
            XbyV = 0;
            for(idx_t k = i; k < A.sizeH(); ++k){
                XbyV += A(k, j) * v[k-i];
            }
            for(idx_t k = i; k < A.sizeH(); ++k){  //прохрод по всем элементам подстолбца
                elm_t temp = A(k, j) - 2 * XbyV / VbyV * v[k-i];
                if( Tabs(temp) < tolerance<elm_t> ) A(k, j) = static_cast<elm_t>(0);
                else A(k, j) = temp;
            }
        }
        V[i].resize(A.sizeH());
        for(idx_t j = i; j < A.sizeH(); ++j){
            V[i][j] = v[j-i];
        }
        std::cout<<std::endl<<A<<std::endl;
        x.pop_back();
        v.pop_back();
        e.pop_back();
    }
    DenseMatrix<elm_t> Q(A.sizeH(), A.sizeW());
    for(int i = V.size() - 1; i >= 0; --i){        //проход по всем сохраненным векторам v
        VbyV = V[i] * V[i];

    }
    return {A, A};
}
#endif //SOLECOURSE_HOUSEHOLDER_H
