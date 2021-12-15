//
// Created by d-qql on 14.02.2021.
//

#ifndef SOLECOURSE_CHEBYSHEV_H
#define SOLECOURSE_CHEBYSHEV_H
#include "vector"
#include <ctgmath>
#include "../utility/Consts.h"

template<typename T>
unsigned long long int C(int n, int k){
    unsigned long long int res = 1;
    for(int i = k + 1; i <= n; ++i ){
        res *= i;
    }
    for(int i = 1; i <= n-k; ++i){
        res /= i;
    }
    return res;
}

template<typename T>
std::vector<T> ChebyshevRoots(std::pair<T, T> section, size_t PowOf2){
    size_t PolyOrder = std::pow(2, PowOf2);
    std::vector<T> roots(PolyOrder);
    for(size_t i = 1; i <= PolyOrder; ++i){
        roots[i-1] = (section.first + section.second) / 2 +
                (section.second - section.first) *
                static_cast<T>(cos(static_cast<double>(2 * i - 1) * M_PI_2 / PolyOrder)) / 2;
    }
    std::vector<size_t> idx;
    idx.push_back(0);
    idx.push_back(1);
    std::vector<size_t> next_idx;
    size_t curOrder;
    for(size_t i = 1; i < PowOf2; ++i){     //по всем порядкам полинома до текущего
        next_idx.resize(pow(2, double(i)+1));
        curOrder = pow(2, double (i) + 1);
        for(size_t j = 0; j < curOrder-1; j+=2){     //по всем элементам нового вектора прыгая через 1
            next_idx[j] = idx[j/2];
            next_idx[j+1] = curOrder - 1 - next_idx[j];
        }
        idx = next_idx;
    }
    std::vector<T> res(PolyOrder);
    for(size_t i = 0; i < PolyOrder; ++i){
        res[i] = roots[idx[i]];
    }
    std::cout<<roots<<res;
    return res;
}

template<typename T>
std::vector<T> AccelerationCoefs(T spectrum){

}

template<typename T>
std::vector<T> ChebyshevCoef(std::pair<T, T> sec, size_t PolyOrder){
    std::vector<T> coefs_f, coefs_s, coefs_new;

    coefs_f.emplace_back(static_cast<T>(1));

    coefs_s.emplace_back( static_cast<T>(0));
    coefs_s.emplace_back( static_cast<T>(1));

    for(size_t k = 1; k < PolyOrder; ++k){ //порядок текущего полинома, после прохода цикла получим полином порядком выше
        coefs_new.resize(k+2);  //к+2 = число кэфов в полиноме порядка к+1
        for(size_t i = 0; i < k+1; ++i){
            coefs_new[i] -= coefs_f[i];
            coefs_new[i+1] += 2 * coefs_s[i];
            if(Tabs(coefs_new[i]) < tolerance<T>) coefs_new[i] = static_cast<T>(0);
            if(Tabs(coefs_new[i+1]) < tolerance<T>) coefs_new[i+1] = static_cast<T>(0);
        }
        coefs_f = coefs_s;
        coefs_s = coefs_new;
        coefs_new.clear();
    }
    coefs_new.resize(PolyOrder+1);
    for(size_t n = 0; n <= PolyOrder; ++n){
        for(size_t k = 0; k <= n; ++k){
            coefs_new[n-k] += coefs_s[n] * C<T>(n, k)*pow(2./(sec.second-sec.first), n-k)*pow(-(sec.second+sec.first)/(sec.second-sec.first), k);
        }
    }
    for(size_t i = 0; i <=PolyOrder; ++i){  //нормировка старшего кэфа
        coefs_new[i]/=coefs_new[PolyOrder];
    }
    std::cout<<coefs_new;
    std::cout<<coefs_s;
    for(size_t i = 0; i <= PolyOrder; ++i){
        std::cout<<coefs_new[i]<<"*x^"<<i<<"+";
    }
    T sum = 0;
    for(size_t i = 0; i <=PolyOrder; ++i) sum += coefs_new[i];
    for(size_t i = 0; i <=PolyOrder; ++i) coefs_new[i]/sum;
    return coefs_new;
}
#endif //SOLECOURSE_CHEBYSHEV_H
