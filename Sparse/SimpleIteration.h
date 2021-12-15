//
// Created by d-qql on 30.01.2021.
//

#ifndef SOLECOURSE_SIMPLEITERATION_H
#define SOLECOURSE_SIMPLEITERATION_H
#include "CSR.h"
#include "../utility/Consts.h"
#include "../utility/gnuplot-iostream.h"
template<typename T>
std::vector<T> SimpleIteration(const CSR<T> &A, const std::vector<T> &b, const T &tao){

    std::vector<T> x(A.sizeH());
    std::vector<T> r = A * x - b;
    int i = 0;
    T N = norm(r);
    std::vector<std::pair<int, double>> plotData;
    while (N > tolerance<T>) {
        x = -tao * (A * x - b) + x;
        r = A * x - b;
        plotData.emplace_back(i, log(N));
        ++i;
        N = norm(r);
        std::cout<<N<<std::endl;
    }


/*    std::ofstream fout;
    fout.open("../PlotData/SimpleIteration/IterNumbTao.txt", std::ios::app);
    fout<<tao<<" "<<i<<"\n";
    fout.close();*/

    std::ofstream fout;
    if(Tabs(tao - 1) < tolerance<T>) fout.open("../PlotData/SimpleIteration/1SimpleIterNorm.txt", std::ios::out);
    else fout.open("../PlotData/SimpleIteration/OptSimpleIterNorm.txt", std::ios::out);
    for(auto k: plotData){
        fout<<k.first<<" "<<k.second<<"\n";
    }
    fout.close();

    Gnuplot gp;
    gp<<"set xlabel 'Номер итерации' \n"
        "set ylabel 'ln(|r|))'\n"
        "set grid\n"
        "set title 'Зависимость логарифма модуля невязки от номера итерации ПМИ' font 'Helvetica Bold, 10'\n";
    gp << "plot '-' with lines title 'Iter->ln(|r|)' lc rgb 'blue'\n";
    gp.send1d(plotData);
    return x;
}

template<typename T>
std::vector<T> FastSimpleIteration(const CSR<T> &A, const std::vector<T> &b, const std::vector<T>& ChebRoots) {

    std::vector<T> x(A.sizeH());
    std::vector<T> r = A * x - b;
    int i = 0;
    T N = norm(r);
    std::vector<std::pair<int, double>> plotData;
    bool flag = true;
    while (flag) {
        for( auto root : ChebRoots) {
            x = -static_cast<T>(1)/root * (A * x - b) + x;
            r = A * x - b;
            plotData.emplace_back(i, log(N));
            ++i;
            N = norm(r);
            std::cout << N << std::endl;
            if( N < tolerance<T> ){
                flag = false;
                break;
            }
        }
    }
    std::ofstream fout;
    fout.open("../PlotData/SimpleIteration/10.txt", std::ios::out);
    for(auto k: plotData){
        fout<<k.first<<" "<<k.second<<"\n";
    }
    fout.close();
    return x;
}



#endif //SOLECOURSE_SIMPLEITERATION_H
