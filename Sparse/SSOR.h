//
// Created by d-qql on 28.02.2021.
//

#ifndef SOLECOURSE_SSOR_H
#define SOLECOURSE_SSOR_H
#include "CSR.h"
#include "../utility/Consts.h"
#include "../utility/Overloads.h"
#include "../Chebyshev/Chebyshev.h"
template<typename T>
std::vector<T> SSOR(const CSR<T> &A, const std::vector<T> &b, const T& w){
    using idx_t = typename CSR<T>::idx_t;

    std::vector<T> r(b.size());
    std::vector<T> x(b.size());
    std::vector<T> temp_x(b.size());
    std::vector<std::pair<int, double>> plotData;
    T sum;
    r = A * x - b;
    int k = 0;

/*  Обычный SSOR
    while (norm(r) > tolerance<T>) {
        plotData.emplace_back(k, log(norm(r)));
        temp_x = x;
        for (idx_t i = 0; i < A.H; ++i) {
            sum = static_cast<T>(0);
            for (idx_t j = A.rows[i]; j < A.rows[i + 1]; ++j) {
                if (i != A.cols[j]) {
                    sum += A.values[j] * x[A.cols[j]];
                }else{ continue; }
            }
            sum *= w;
            x[i] = (w * b[i] - sum)/A(i, i) + (static_cast<T>(1) - w) * temp_x[i];
        }
        temp_x = x;
        for(int i = A.H - 1; i >= 0; --i){
            sum = static_cast<T>(0);
            for (idx_t j = A.rows[i]; j < A.rows[i + 1]; ++j) {
                if (i != A.cols[j]) {
                    sum += A.values[j] * x[A.cols[j]];
                }else{ continue; }
            }
            sum *= w;
            x[i] = (w * b[i] - sum)/A(i, i) + (static_cast<T>(1) - w) * temp_x[i];
        }
        ++k;
       // std::cout<<norm(r)<<std::endl;
        r = A * x - b;
    } */

//Ускоренный SSOR
    std::vector<T> coefs = ChebyshevCoef<T>({-0.000001, 0.000001}, 2);
    std::cout<<coefs;
    std::vector<std::vector<T>> X;
    while (norm(r) > tolerance<T>) {
        for(size_t p = 0; p < 3; ++p) {
            plotData.emplace_back(k, log(norm(r)));
            temp_x = x;
            for (idx_t i = 0; i < A.H; ++i) {
                sum = static_cast<T>(0);
                for (idx_t j = A.rows[i]; j < A.rows[i + 1]; ++j) {
                    if (i != A.cols[j]) {
                        sum += A.values[j] * x[A.cols[j]];
                    } else { continue; }
                }
                sum *= w;
                x[i] = (w * b[i] - sum) / A(i, i) + (static_cast<T>(1) - w) * temp_x[i];
            }
            temp_x = x;
            for (int i = A.H - 1; i >= 0; --i) {
                sum = static_cast<T>(0);
                for (idx_t j = A.rows[i]; j < A.rows[i + 1]; ++j) {
                    if (i != A.cols[j]) {
                        sum += A.values[j] * x[A.cols[j]];
                    } else { continue; }
                }
                sum *= w;
                x[i] = (w * b[i] - sum) / A(i, i) + (static_cast<T>(1) - w) * temp_x[i];
            }
            X.emplace_back(x);

        }
        x = coefs[2]*X[0];
        for(size_t p = 1; p < 3; ++p)  x= x+ coefs[2-p] * X[p];
        X.clear();
        ++k;
         std::cout<<norm(r)<<std::endl;
        r = A * x - b;
    }
    std::cout<<k;
    return x;
}


#endif //SOLECOURSE_SSOR_H
