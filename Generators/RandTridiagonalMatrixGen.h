//
// Created by d-qql on 28.01.2021.
//

#ifndef SOLECOURSE_RANDTRIDIAGONALMATRIXGEN_H
#define SOLECOURSE_RANDTRIDIAGONALMATRIXGEN_H
#include <set>
#include "../utility/Triplet.h"

template<typename T>
std::set<Triplet<T>> GenerateTridiagonal(const size_t& size, const int &a, const int &b){
    srand(time(0));
    std::set<Triplet<T>> out;
    for(size_t i = 0; i < size-1; ++i){
        out.insert({static_cast<T>(a + rand()%(b-a+1)), i, i}); //диагональный
        out.insert({static_cast<T>(a + rand()%(b-a+1)), i, i+1}); //над диагональю
        out.insert({static_cast<T>(a + rand()%(b-a+1)), i+1, i}); //под диагональю
    }
    out.insert({static_cast<T>(a + rand()%(b-a+1)), size-1, size-1}); //последний диагональный
    return out;
}
#endif //SOLECOURSE_RANDTRIDIAGONALMATRIXGEN_H
