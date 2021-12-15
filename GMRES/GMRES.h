//
// Created by d-qql on 28.03.2021.
//

#ifndef SOLECOURSE_GMRES_H
#define SOLECOURSE_GMRES_H
#include "../Sparse/CSR.h"
#include "../Dense/BackSubstitution.h"
#include "../Givens/Rotations.h"
#include "../Krylov/Krylov.h"
template<typename T>
std::vector<T> GMRES(const CSR<T>& A, const std::vector<T>& b, size_t m = 4 ){
    std::vector<T> x(b.size());
    std::vector<T> r = b - A * x;

    T N = norm(r);
    int K = 0;
    std::vector<T> B;
    std::vector<T> temp_x;
    DenseMatrix<T> H(m+1, m);
    std::vector<std::vector<T>> Basis;
    Basis.push_back(1./N * r);
    DenseMatrix<T> V(r.size(), m);
    std::vector<std::pair<T, T>> rotations;
    B.resize(m+1, 0);
    B[0] = N;

    std::ofstream fout;
    fout.open("../PlotData/GMRES/Norm.txt", std::ios::out);
    fout.close();
    fout.open("../PlotData/GMRES/Norm.txt", std::ios::app);
    fout<<0<<" "<<log(N)<<"\n";
    fout.close();


        //std::cout<<x;
    for(int i = 2; i <= m+1; ++i){
        KrylovSubSpace(A, Basis, H, i-2);
        // std::cout<<V<<std::endl;
        //std::cout<<H<<std::endl;
        HessenbergRot(H, B, i-2, rotations);
        //std::cout<<H<<std::endl;
        std::cout<<"итерация: "<<i-2<<std::endl;
        //std::cout<<B<<std::endl;
        std::cout<<"НЕВЯЗКА: "<<Tabs(B[i-1])<<"\n";
        fout.open("../PlotData/GMRES/Norm.txt", std::ios::app);
        fout<<i-1<<" "<<log(Tabs(B.back()))<<"\n";
        fout.close();
        if(Tabs(B[i-1]) < tolerance<T> || i == m+1){
            // std::cout<<B;
            temp_x = backSubstTopTriangular(H, B, i-1, i-1);
            //std::vector<T> dx = V * temp_x;
            //std::cout<<"Обратный ход: "<<B-H*temp_x<<std::endl;
            // std::cout<<"Dx: "<<V*temp_x;
            for(size_t j = 0; j < i-1; ++j){
                for(size_t k = 0; k < r.size(); ++k){
                    V(k, j) = Basis[j][k];
                }
            }
            x = x + V * temp_x;
            r = b - A * x;
            N = norm(r);
            ++K;
            break;
        }
    }

    std::cout<<"НЕВЯЗКА: "<<N<<"\n";
    return x;
}
#endif //SOLECOURSE_GMRES_H
