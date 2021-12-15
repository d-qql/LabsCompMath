//
// Created by d-qql on 12.04.2021.
//

#ifndef SOLECOURSE_BICG_H
#define SOLECOURSE_BICG_H
#include <vector>
#include "CSR.h"

template<typename T>
std::vector<T> BiCG(const CSR<T>& A, const std::vector<T>& b){
    std::vector<T> x(b.size());
    std::vector<T> r0 = b - A * x;
    std::vector<T> r1 = r0;
    std::vector<T> p0, p1;
    std::vector<T> z0, z1;
    CSR<T> At = A.transpose();
    T N = norm(r0);
    T rho;
    T prev_rho;
    T Beta;
    T Alfa;
    long long int i = 1;
    while( N > tolerance<T> ){
        rho = r1*r0;
        if (Tabs(rho) < tolerance<T>){
           // std::cout<<"FAIL!, RESTARTING...";
           // std::cout<<"\n"<<i<<"\n";
            i = 1;
           // return BiCG(A, b, x);
        }
        if( i == 1 ){
            p0 = r0;
            p1 = r1;
        }else{
            Beta = rho/prev_rho;
            p0 = r0 + Beta * p0;
            p1 = r1 + Beta * p1;
        }
        z0 = A * p0;
        z1 = At * p1;
        Alfa = rho/(p1*z0);
        x = x + Alfa * p0;
        r0 = r0 - Alfa * z0;
        r1 = r1 - Alfa * z1;
        N = norm(r0);
        ++i;
        prev_rho = rho;
    }
    return x;
}
#endif //SOLECOURSE_BICG_H
