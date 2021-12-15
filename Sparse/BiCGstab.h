//
// Created by d-qql on 27.04.2021.
//

#ifndef SOLECOURSE_BICGSTAB_H
#define SOLECOURSE_BICGSTAB_H
#include <vector>
#include "CSR.h"

template<typename T>
std::vector<T> BiCGstab(const CSR<T>& A, const std::vector<T>& b, long long int max_step){
    std::vector<T> x(b.size());
    std::vector<T> r0 = b - A * x;
    std::vector<T> r1 = r0;
    std::vector<T> p;
    std::vector<T> s;
    std::vector<T> t;
    std::vector<T> v;
    T N = norm(r0);
    T rho;
    T prev_rho;
    T Beta;
    T Alfa;
    T omega;

    for(long long int i = 1; i <= max_step; ++i){
        rho = r1*r0;
        if (Tabs(rho) < tolerance<T>){
        //    std::cout<<"FAIL!, RESTARTING...";
        //    std::cout<<"\n"<<i<<"\n";
            i = 1;
            // return BiCG(A, b, x);
        }
        if( i == 1 ){
            p = r0;
        }else{
            Beta = (rho/prev_rho)*(Alfa/omega);
            p = r0 + Beta*(p - omega * v);
        }
        v = A * p;
        Alfa = rho / (r1 * v);
        s = r0 - Alfa * v;
        if(norm(s) < tolerance<T>){
            x = x + Alfa * p;
            break;
        }
        t = A * s;
        omega = (t * s) / (t * t);
        x = x + Alfa * p + omega * s;
        r0 = s - omega * t;
        N = norm(r0);
        if(N < tolerance<T>){
            break;
        }
        if(Tabs(omega) < tolerance<T>) i = 0; //restart
        prev_rho = rho;
    }
    return x;
}
#endif //SOLECOURSE_BICGSTAB_H
