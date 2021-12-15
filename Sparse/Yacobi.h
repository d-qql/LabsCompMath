//
// Created by d-qql on 08.02.2021.
//

#ifndef SOLECOURSE_YACOBI_H
#define SOLECOURSE_YACOBI_H
#include "CSR.h"
#include "../utility/Consts.h"
#include "../utility/Overloads.h"


template<typename T>
std::vector<T> Yacobi(const CSR<T> &A, const std::vector<T> &b) {
    using idx_t = typename CSR<T>::idx_t;

    std::vector<T> r(b.size());
    std::vector<T> x(b.size());
    std::vector<T> temp_x(b.size());
    std::vector<std::pair<int, double>> plotData;
    T sum;
    r = A * x - b;
    int k = 0;
    while (norm(r) > tolerance<T>) {
        plotData.emplace_back(k, log(norm(r)));
        for (idx_t i = 0; i < A.H; ++i) {
            sum = static_cast<T>(0);
            for (idx_t j = A.rows[i]; j < A.rows[i + 1]; ++j) {
                if (i != A.cols[j]) {
                    sum += A.values[j] * x[A.cols[j]];
                }else{ continue; }
            }
            temp_x[i] = (b[i] - sum)/A(i, i);
        }
        x = temp_x;
        ++k;
        r = A * x - b;
    }
    std::ofstream fout;
    fout.open("../PlotData/SimpleIteration/YacobiNorm.txt", std::ios::out);
    for(auto i: plotData){
        fout<<i.first<<" "<<i.second<<"\n";
    }
    fout.close();

    return x;
}
#endif //SOLECOURSE_YACOBI_H
