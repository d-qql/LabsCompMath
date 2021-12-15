//
// Created by d-qql on 28.01.2021.
//

#ifndef SOLECOURSE_OVERLOADS_H
#define SOLECOURSE_OVERLOADS_H
#include <vector>
#include "../Sparse/CSR.h"
#include "../Dense/DenseMatrix.h"

template<typename T>
std::vector<T> operator*(const T &k, std::vector<T> const &vec){
    std::vector<T> result;
    result.resize(vec.size());
    for(size_t i = 0; i<result.size(); ++i){
        result[i] = k*vec[i];
    }
    return result;
}

template<typename T>
std::vector<T> operator+(const std::vector<T>& fVec, const std::vector<T>& sVec){
    std::vector<T> res;
    if(fVec.size() == sVec.size()){
        res.resize(fVec.size());
        for(size_t i = 0; i < fVec.size(); ++i){
            res[i]=fVec[i]+sVec[i];
        }
    }
    return res;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& fVec, const std::vector<T>& sVec){
    std::vector<T> res;
    if(fVec.size() == sVec.size()){
        res.resize(fVec.size());
        for(size_t i = 0; i < fVec.size(); ++i){
            res[i]=fVec[i]-sVec[i];
        }
    }
    return res;
}

template<typename T>
T operator*(const std::vector<T>& a, const std::vector<T>& b){
    auto res = static_cast<T>(0);
    for(size_t i = 0; i < a.size(); ++i){
        res += a[i]*b[i];
    }
    return res;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& b){
    os<<"( ";
    for(auto i: b){
        os<<i<<" ";
    }
    os<<")\n";
    return os;
}

template<typename T>
void printSystem(const CSR<T> &A, const std::vector<T> &b){
    for(size_t i = 0; i < A.sizeH(); ++i){
        for(size_t j = 0; j < A.sizeW(); ++j){
            std::cout<<A(i, j)<<" ";
        }
        std::cout<<b[i]<<std::endl;
    }
}

template<typename T>
void printSystem(const DenseMatrix<T> &A, const std::vector<T> &b){
    for(size_t i = 0; i < A.sizeH(); ++i){
        for(size_t j = 0; j < A.sizeW(); ++j){
            std::cout<<A(i, j)<<" ";
        }
        std::cout<<b[i]<<std::endl;
    }
}


#endif //SOLECOURSE_OVERLOADS_H
