//
// Created by d-qql on 13.03.2021.
//

#ifndef SOLECOURSE_CG_H
#define SOLECOURSE_CG_H
#include <vector>
#include "CSR.h"

template<typename T>
std::vector<T> CGmethod(const CSR<T>& A, const std::vector<T>& b){
    std::vector<T> x(b.size());
    std::vector<T> Ri = b - A * x;
    std::vector<T> Ri1(Ri.size());
    std::vector<T> Di=Ri;
    std::vector<T> Di1(Ri.size());
    T alpha = 0;
    int iteration = 0;
    std::vector<std::pair<int, T>> plotData;
    T N = norm(Ri);
    while(N > tolerance<T> ){
        //std::cout<<norm(Ri)<<std::endl;
        alpha = Ri*Di/(Di*(A*Di));
        x = x - alpha*Di;
        Ri1 = A*x-b;
        Di1 = Ri1 + (Ri1*Ri1)/(Ri*Di)*Di;
        Di = Di1;
        Ri = Ri1;
        iteration++;
        N = norm(Ri);
        plotData.emplace_back(iteration, log(N));
    }
    std::ofstream fout;
    fout.open("../PlotData/CG/10.txt", std::ios::out);
    for(auto i: plotData){
        fout<<i.first<<" "<<i.second<<"\n";
    }
    fout.close();
    return x;
}

#endif //SOLECOURSE_CG_H
