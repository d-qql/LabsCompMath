//
// Created by d-qql on 28.02.2021.
//

#ifndef SOLECOURSE_GRADIENTDESCENT_H
#define SOLECOURSE_GRADIENTDESCENT_H
#include "CSR.h"
#include "../utility/Consts.h"
#include "../utility/Overloads.h"

template<typename T>
int GradientDescent(const CSR<T>& A, const std::vector<T>& b){
    std::vector<std::pair<double, double>> plotData;

    std::vector<T> r(b.size());
    std::vector<T> x(b.size());
    T alpha;
    r = b - A * x;
    T N = norm(r);
    int k = 0;
    while(N > tolerance<T>){
        alpha = r*r/(r * (A * r));
        x = x + alpha * r;
        r = b - A * x;
        N = norm(r);
        ++k;
        plotData.emplace_back(k, log(N));
    }

    std::cout<<x<<std::endl;

    std::ofstream fout;
    fout.open("../PlotData/GradientDescent/10.txt", std::ios::out);
    for(auto i: plotData){
        fout<<i.first<<" "<<i.second<<"\n";
    }
    fout.close();
    //std::cout<<k;
    return k;
}
#endif //SOLECOURSE_GRADIENTDESCENT_H
