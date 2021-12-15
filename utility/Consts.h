//
// Created by d-qql on 26.01.2021.
//

#ifndef SOLECOURSE_CONSTS_H
#define SOLECOURSE_CONSTS_H
#include <cmath>

template<typename T>
[[maybe_unused]] const auto tolerance = static_cast<T>(1e-9);

template<typename T>
T Tabs(T value){
    if (value < T(0)) return -value;
    else return value;
}

template<typename T>
T norm(const std::vector<T>& v){
    T sum = 0;
    for(auto i: v) sum+=i*i;
    return std::sqrt(sum);
}

template<typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

#endif //SOLECOURSE_CONSTS_H
