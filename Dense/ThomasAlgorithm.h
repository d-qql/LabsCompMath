//
// Created by d-qql on 28.01.2021.
//

#ifndef SOLECOURSE_THOMASALGORITHM_H
#define SOLECOURSE_THOMASALGORITHM_H

#include "DenseMatrix.h"
#include <vector>
#include "../utility/Consts.h"
template<typename T>
std::vector<T> ThomasAlgorithm(DenseMatrix<T> A, std::vector<T> B){
    using idx_t = typename DenseMatrix<T>::idx_t;

    idx_t N = B.size();
    std::vector<T> a(N);
    std::vector<T> b(N);
    std::vector<T> res(N);

    //Прямой ход метода
    T y = A(0, 0);
    try {
        if (Tabs(y) < tolerance<T>) throw "Метод ThomasAlgo не сходится.";
    } catch (char *str){
        std::cout<<str<<std::endl;
    }
    a[0] = -A(0, 1) / y;
    b[0] = B[0] / y;

    for(idx_t i = 1; i < N-1; ++i){
        try{
            y = A(i, i) + A(i, i-1) * a[i-1];
            if (Tabs(y) < tolerance<T>) throw "Метод ThomasAlgo не сходится.";
        } catch (char *str) {
            std::cout<<str<<std::endl;
        }
        a[i] = -A(i, i+1) / y;
        b[i] = (B[i] - A(i, i-1) * b[i-1]) / y;
    }

    //Обратный ход метода
    res[N-1] = (B[N-1] - A(N-1, N-2) * b[N-2]) /
            (A(N-1, N-1) + A(N-1, N-2) * a[N-2]);

    for(int i = N-2; i >= 0; --i){
        res[i] = a[i] * res[i+1] + b[i];
    }
    return res;
}
#endif //SOLECOURSE_THOMASALGORITHM_H
