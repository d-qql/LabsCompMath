//
// Created by d-qql on 27.01.2021.
//

#ifndef SOLECOURSE_RANDMATRIXGENERATOR_H
#define SOLECOURSE_RANDMATRIXGENERATOR_H

#include <set>
#include "../utility/Triplet.h"

template<typename T>
std::set<Triplet<T>> GenerateMatrix(const size_t& size, const int &a, const int &b){
    srand(time(0));
    std::set<Triplet<T>> out;
    for(size_t i = 0; i < size; ++i){
        for(size_t j = 0; j < size; ++j){
            out.insert({a + static_cast <T> (rand()) /( static_cast <T> (RAND_MAX/(b-a))), i, j});
        }
    }
    return out;
}

template<typename T>
std::set<Triplet<T>> GenerateMatrix_FilledNumber(const size_t &size, const size_t& filledNumber, const int &a, const int&b){
    srand(time(0));
    std::set<Triplet<T>> out;
    while(out.size() < filledNumber){
        out.insert({static_cast<T>(a + rand()%(b-a+1)), static_cast<size_t>(rand() % size), static_cast<size_t>(rand() % size)});
    }
    return out;
}

template<typename T>
std::set<Triplet<T>> GenerateMatrixDiagDominant(const size_t& size){
    srand(time(0));
    std::set<Triplet<T>> in;
    for(size_t i = 0; i < size; ++i){
        for(size_t j = 0; j <= i; ++j){
            if( i == j ) in.insert({1 + static_cast <double > (rand()) /( static_cast <double > (RAND_MAX/(0.6))), i, j});
            else {
                in.insert({(-1 + static_cast <double > (rand()) /( static_cast <double > (RAND_MAX/(2))))/1000, i, j});
                in.insert({(-1 + static_cast <double > (rand()) /( static_cast <double > (RAND_MAX/(2))))/1000, j, i});
            }
        }
    }
    return in;
}
#endif //SOLECOURSE_RANDMATRIXGENERATOR_H
