//
// Created by d-qql on 26.01.2021.
//

#ifndef SOLECOURSE_DENSEMATRIX_H
#define SOLECOURSE_DENSEMATRIX_H

#include <iostream>
#include <vector>
#include <set>
#include "../utility/Triplet.h"

template<typename T>
class DenseMatrix {
public:

    using elm_t = T;
    using idx_t = std::size_t;

private:

    idx_t H{}, W{};
    std::vector<elm_t> matrix;

public:

    DenseMatrix() = default;
    DenseMatrix(const idx_t h, const idx_t w) : H(h), W(w) {
        matrix.resize(H * W);
    }
    DenseMatrix(const idx_t h, const idx_t w, const std::set<Triplet<elm_t>> &in): H(h), W(w) {
        matrix.resize(H * W);
        for(auto elm: in){
            matrix[elm.i * W + elm.j] = elm.value;
        }
    }
    elm_t& operator()(const idx_t& i, const idx_t& j){
        return matrix[i*W + j];
    }

    const elm_t& operator()(const idx_t& i, const idx_t& j) const{
        return matrix[i*W + j];
    }

    [[nodiscard]] idx_t sizeH() const {
        return H;
    }
    [[nodiscard]] idx_t sizeW() const {
        return W;
    }

    void swap(const idx_t& fStr, const idx_t& sStr){
        for(idx_t j = 0; j < W; ++j) std::swap(matrix[fStr*W + j], matrix[sStr*W+j]);
    }

    void __deleteLastRow(){
        --H;
        matrix.erase(matrix.end()-W, matrix.end());
    }
};
//Обработать исключение несопоставимых размеров
template<typename T>
std::vector<T> operator*(const DenseMatrix<T> &M, const std::vector<T> &vec){
    std::vector<T> res;
    res.resize(M.sizeH());
    for(size_t i = 0; i < M.sizeH(); ++i){
        for(size_t j = 0; j < M.sizeW(); ++j){
            res[i]+=M(i,j)*vec[j];
        }
    }
    return res;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const DenseMatrix<T>& M){
    for(size_t i = 0; i < M.sizeH(); ++i){
        for(size_t j = 0; j<M.sizeW(); ++j){
            os<<M(i, j)<<" ";
        }
        os<<std::endl;
    }
    return os;
}



#endif //SOLECOURSE_DENSEMATRIX_H
