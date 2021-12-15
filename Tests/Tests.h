//
// Created by d-qql on 31.01.2021.
//

#ifndef SOLECOURSE_TESTS_H
#define SOLECOURSE_TESTS_H

#include <vector>
#include <ctgmath>
#include "../Dense/DenseMatrix.h"
#include "../Generators/RandMatrixGenerator.h"
#include "../Generators/RandVectorGenerator.h"
#include "../Dense/LU.h"
#include "../utility/Consts.h"
#include "../Generators/RandTridiagonalMatrixGen.h"
#include "../Dense/ThomasAlgorithm.h"
#include "../Dense/GaussMethod.h"
#include "../Dense/HeadGaussMethod.h"
#include "../Sparse/CSR.h"
#include "../Sparse/SimpleIteration.h"
#include "../Sparse/Gauss-Seidel.h"

void testMultiply_Time_DifferentDim(size_t dim){     //тест скорость умножения плотной и разреженной матрицы на вектор
    std::vector<double> d(dim);
    std::vector<double> c(dim);
    std::ofstream fout;
    fout.open("../PlotData/MultiplyTime/Dense20percent.txt", std::ios::out);
    fout.close();
    fout.open("../PlotData/MultiplyTime/CSR20percent.txt", std::ios::out);
    fout.close();
    for(size_t i = 3; i<=dim; ++i) {
        std::set<Triplet<double>> in = GenerateMatrix_FilledNumber<double>(i, i*i/5, -10000, 10000);
        DenseMatrix<double> D(i, i, in);
        CSR<double> C(i, i, in);
        std::vector<double> b = GenerateVector<double>(i, -10000, 10000);
        clock_t end, start = clock();
        d = D*b;
        end = clock();
        fout.open("../PlotData/MultiplyTime/Dense20percent.txt", std::ios::app);
        fout<<i<<" "<<log(double(end - start) / CLOCKS_PER_SEC)<<"\n";
        fout.close();
        start = clock();
        c = C*b;
        end = clock();
        fout.open("../PlotData/MultiplyTime/CSR20percent.txt", std::ios::app);
        fout<<i<<" "<<log(double(end - start) / CLOCKS_PER_SEC)<<"\n";
        fout.close();
        std::cout<<d<<c<<std::endl;
        std::cout<<i<<"\n";
    }
    Gnuplot gp;
    gp<<"set xlabel 'Размерность' \n"
        "set ylabel 'Время умножения'\n"
        "set grid\n"
        "set title 'Зависимость времени умножения матрицы на плотный вектор от размерности, 20% заполнение' font 'Helvetica Bold, 10'\n";
    gp << "plot '../PlotData/MultiplyTime/Dense20percent.txt' with lines title 'Dense' lc rgb 'blue',"
          "     '../PlotData/MultiplyTime/CSR20percent.txt' with lines title 'CSR' lc rgb 'red'\n";
}

void compareSimpleIterationMetods_Norm(size_t dim){
    std::set<Triplet<double>> in;
    for(size_t i = 0; i < dim; ++i){
        for(size_t j = 0; j < dim; ++j){
            if( i == j ) in.insert({1 + static_cast <double > (rand()) /( static_cast <double > (RAND_MAX/(0.6))), i, j});
            else in.insert({(-1 + static_cast <double > (rand()) /( static_cast <double > (RAND_MAX/(2))))/10, i, j});
        }
    }
    CSR<double> A = CSR<double>(dim, dim, in);
    std::vector<double> b = GenerateVector<double>(dim, -1, 1);
    printSystem(A, b);
    std::cout<<SimpleIteration(A, b, 1.);
    std::cout<<GaussSeidel(A, b);
    Gnuplot gp;
    gp<<"set xlabel 'Номер итерации' \n"
        "set ylabel 'Логарифм невязки'\n"
        "set grid\n"
        "set title 'Зависимость логарифма нормы невязки от номера итерации' font 'Helvetica Bold, 10'\n";
    gp << "plot '../PlotData/SimpleIteration/SimpleIterNorm.txt' with lines title 'ПМИ' lc rgb 'blue',"
          "     '../PlotData/SimpleIteration/GaussSeidelNorm.txt' with lines title 'Гаусс-Зейдель' lc rgb 'red'\n";
}

void testMultiply_Time(size_t dim){     //тест скорость умножения плотной и разреженной матрицы на вектор
    std::vector<std::pair<int, double>> plotDataDense;
    std::vector<std::pair<int, double>> plotDataCSR;
    std::vector<double> d(dim);
    std::vector<double> c(dim);
    std::ofstream fout;
    fout.open("../PlotData/MultiplyTime/Dense.txt", std::ios::out);
    fout.close();
    fout.open("../PlotData/MultiplyTime/CSR.txt", std::ios::out);
    fout.close();
    for(size_t i = 1; i<=dim*dim/2; ++i) {
        std::set<Triplet<double>> in = GenerateMatrix_FilledNumber<double>(dim, i, -10000, 10000);
        DenseMatrix<double> D(dim, dim, in);
        CSR<double> C(dim, dim, in);
        std::vector<double> b = GenerateVector<double>(dim, -10000, 10000);
        clock_t end, start = clock();
        d = D*b;
        end = clock();
        fout.open("../PlotData/MultiplyTime/Dense.txt", std::ios::app);
        fout<<i<<" "<<double(end - start) / CLOCKS_PER_SEC<<"\n";
        fout.close();
        start = clock();
        c = C*b;
        end = clock();
        fout.open("../PlotData/MultiplyTime/CSR.txt", std::ios::app);
        fout<<i<<" "<<double(end - start) / CLOCKS_PER_SEC<<"\n";
        fout.close();
        std::cout<<d<<c<<std::endl;
        std::cout<<i<<"\n";
    }
    Gnuplot gp;
    gp<<"set xlabel 'Число ненулевых элементов' \n"
        "set ylabel 'Время умножения'\n"
        "set grid\n"
        "set title 'Зависимость времени умножения матрицы на плотный вектор от заполненности' font 'Helvetica Bold, 10'\n";
    gp << "plot '../PlotData/MultiplyTime/Dense.txt' with lines title 'Dense' lc rgb 'blue',"
          "     '../PlotData/MultiplyTime/Dense.txt' with lines title 'CSR' lc rgb 'red'\n";
}

void testIterNumber_Tao(){
    std::set<Triplet<double>> in;
    std::vector<double> b(300);
    for(size_t i = 1; i <= 300; ++i){
        in.insert({double(i)/30, i-1, i-1});
        b[i-1] = double(301-i)/30;
    }
    CSR<double> A = CSR<double>(300, 300, in);
    for(int i = 1; i <= 500; ++i){
        SimpleIteration(A, b, 0.19894 +  double(i) / 1000000);
    }

/*    Gnuplot gp;
    gp<<"set xlabel 'Тао' \n"
        "set ylabel 'Число итераций'\n"
        "set xrange [0.03:0.22]\n"
        "set grid\n"
        "set title 'Зависимость числа итераций от Тао для схождения с точностью 1e-12' font 'Helvetica Bold, 10'\n";
    gp << "plot '../PlotData/SimpleIteration/IterNumbTao.txt' with lines title 'Taо -> IterNumb' lc rgb 'blue'\n";*/
}

void testSimpleIteration_Speed(){
    std::set<Triplet<double>> in;
    std::vector<double> b(300);
    for(size_t i = 1; i <= 300; ++i){
        in.insert({double(i)/30, i-1, i-1});
        b[i-1] = double(301-i)/30;
    }
    CSR<double> A = CSR<double>(300, 300, in);
    //std::vector<double> tao = {0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.15, 0.17, 0.18, 0.185, 0.19, 0.20, 0.23, 0.26, 0.31, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::cout<<SimpleIteration(A, b, 0.15);
    std::cout<<SimpleIteration(A, b, 0.165);
    std::cout<<SimpleIteration(A, b, 0.18);
    std::cout<<SimpleIteration(A, b, 0.195);
    std::cout<<SimpleIteration(A, b, 0.135);

    Gnuplot gp;
    gp<<"set xlabel 'Номер итерации' \n"
        "set ylabel 'ln(|r|))'\n"
        "set grid\n"
        "set title 'Зависимость логарифма модуля невязки от номера итерации ПМИ' font 'Helvetica Bold, 10'\n";
    gp << "plot '../PlotData/SimpleIteration/SimpleIteration_Speed0.150000.txt' u 1:2 with lines title 'tao = 0.15' lc rgb 'blue', "
          "     '../PlotData/SimpleIteration/SimpleIteration_Speed0.165000.txt' u 1:2 with lines title 'tao = 0.165' lc rgb 'red', "
          "     '../PlotData/SimpleIteration/SimpleIteration_Speed0.180000.txt' u 1:2 with lines title 'tao = 0.18' lc rgb 'green', "
          "     '../PlotData/SimpleIteration/SimpleIteration_Speed0.195000.txt' u 1:2 with lines title 'tao = 0.195' lc rgb 'black', "
          "     '../PlotData/SimpleIteration/SimpleIteration_Speed0.135000.txt' u 1:2 with lines title 'tao = 0.135' lc rgb 'purple'\n";

}

std::vector<std::pair<double, double>> testLUMethod_Norm(size_t dim);

std::vector<std::pair<double, double>> testLUMethod_Time(size_t dim);

std::vector<std::pair<double, double>> testGaussMethod_Norm(size_t dim);

std::vector<std::pair<double, double>> testGaussMethod_Time(size_t dim);

std::vector<std::pair<double, double>> testHeadGaussMethod_Norm(size_t dim);

std::vector<std::pair<double, double>> testHeadGaussMethod_Time(size_t dim);

void testThomasAlgo(size_t dim);

void compareStraightMethodsNorm(size_t dim);

void compareStraightMethodsTime(size_t dim);

void compareStraightMethodsTime(size_t dim){
    Gnuplot gp;
    gp<<"set xlabel 'Matrix Dimension' \n"
        "set ylabel 'Time'\n"
        "set grid\n"
        "set title 'Зависимость времени от размера матрицы для разных методов' font 'Helvetica Bold, 10'\n";
    gp << "plot ";
    gp << "     '-' with lines title 'dim->time, pureGauss' ls 11 lc rgb 'blue',"
          "     '-' with lines title 'dim->time, headGauss' ls 11 lc rgb 'green',"
         // "     '-' with lines title 'dim->time, LU' ls 12 lc rgb 'red'"
          "\n";
    gp.send1d(testGaussMethod_Time(dim));
    gp.send1d(testHeadGaussMethod_Time(dim));
    //gp.send1d(testLUMethod_Time(dim));
}

std::vector<std::pair<double, double>> testLUMethod_Norm(size_t dim){
    //ЛОГАРИФМ МОДУЛЯ НЕВЯЗКИ ОТ РАЗМЕРА МАТРИЦЫ headGaussMethod
    std::vector<std::pair<double, double>> plotData;
    for(int i = 3; i < dim; ++i){
        DenseMatrix<double> A = DenseMatrix<double>(i, i, GenerateMatrix<double>(i, -1000, 1000));
        std::vector<double> b = GenerateVector<double>(i, -1, 1);
        std::cout<<A<<b;
        std::vector<double> res = solveByLU(A, b);
        plotData.emplace_back(i, log(norm(b-A*res)));
        std::cout<<res;
    }
/*    Gnuplot gp;
    gp<<"set xlabel 'Matrix Dimension' \n"
        "set ylabel 'ln(|r|))'\n"
        "set grid\n"
        "set title 'Зависимость логарифма модуля невязки от размера матрицы GaussMethod' font 'Helvetica Bold, 10'\n";
    gp << "plot '-' with lines title 'dim->ln(|r|)' lc rgb 'blue'\n";
    gp.send1d(plotData);*/
    return plotData;
}

std::vector<std::pair<double, double>> testLUMethod_Time(size_t dim){
    std::vector<std::pair<double, double>> plotData;
    for(int i = 3; i <= dim; ++i){
        DenseMatrix<double> A = DenseMatrix<double>(i, i, GenerateMatrix<double>(i, -1000, 1000));
        std::vector<double> b = GenerateVector<double>(i, -1, 1);
        clock_t end, start = clock();
        std::vector<double> res = solveByLU(A, b);
        end = clock();
        plotData.emplace_back(log(i), log(double(end-start)/CLOCKS_PER_SEC));
        std::cout<<res;
    }
    return plotData;
}

void testThomasAlgo(size_t dim){
    DenseMatrix<double> A = DenseMatrix<double>(dim, dim, GenerateTridiagonal<double>(dim, -200, 200));
    std::cout<<A;
    std::vector<double> b = GenerateVector<double>(dim, -1, 1);
    std::cout<<b<<ThomasAlgorithm(A, b);
}

std::vector<std::pair<double, double>> testGaussMethod_Norm(size_t dim){
    //ЛОГАРИФМ МОДУЛЯ НЕВЯЗКИ ОТ РАЗМЕРА МАТРИЦЫ headGaussMethod
    std::vector<std::pair<double, double>> plotData;
    for(int i = 3; i < dim; ++i){
        DenseMatrix<double> A = DenseMatrix<double>(i, i, GenerateMatrix<double>(i, -1000, 1000));
        std::vector<double> b = GenerateVector<double>(i, -1, 1);
        std::cout<<A<<b;
        std::vector<double> res = GaussMethod(A, b);
        plotData.emplace_back(i, log(norm(b-A*res)));
        std::cout<<res;
    }
    Gnuplot gp;
    gp<<"set xlabel 'Matrix Dimension' \n"
        "set ylabel 'ln(|r|))'\n"
        "set grid\n"
        "set title 'Зависимость логарифма модуля невязки от размера матрицы GaussMethod' font 'Helvetica Bold, 10'\n";
    gp << "plot '-' with lines title 'dim->ln(|r|)' lc rgb 'blue'\n";
    gp.send1d(plotData);
    return plotData;
}

std::vector<std::pair<double, double>> testGaussMethod_Time(size_t dim){
    //ИССЛЕДОВАНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ ОТ РАЗМЕРА МАТРИЦЫ headGaussMethod
    std::vector<std::pair<double, double>> plotData;
    for(int i = 3; i < dim; ++i){
        DenseMatrix<double> A = DenseMatrix<double>(i, i, GenerateMatrix<double>(i, -1000, 1000));
        std::vector<double> b = GenerateVector<double>(i, -1, 1);
        clock_t end, start = clock();
        std::vector<double> res = GaussMethod(A, b);
        end = clock();
        plotData.emplace_back(log(i),log(double(end-start)/CLOCKS_PER_SEC));
        std::cout<<res;
    }
    Gnuplot gp;
    gp<<"set xlabel 'ln(Matrix Dimension)' \n"
        "set ylabel 'ln(Time)'\n"
        "set grid\n"
        "set title 'Зависимость времени выполнения от размера матрицы GaussMethod' font 'Helvetica Bold, 10'\n";
    gp << "plot '-' with lines title 'ln(dim)->ln(time)' lc rgb 'blue'\n";
    gp.send1d(plotData);
    return plotData;
}

std::vector<std::pair<double, double>> testHeadGaussMethod_Norm(size_t dim){
    //ЛОГАРИФМ МОДУЛЯ НЕВЯЗКИ ОТ РАЗМЕРА МАТРИЦЫ headGaussMethod
    std::vector<std::pair<double, double>> plotData;
    for(int i = 3; i < dim; ++i){
        DenseMatrix<double> A = DenseMatrix<double>(i, i, GenerateMatrix<double>(i, -1000, 1000));
        std::vector<double> b = GenerateVector<double>(i, -1, 1);
        std::vector<double> res = headGaussMethod(A, b);
        plotData.emplace_back(i, log(norm(b-A*res)));
        std::cout<<res;
    }
    Gnuplot gp;
    gp<<"set xlabel 'Matrix Dimension' \n"
        "set ylabel 'ln(|r|))'\n"
        "set grid\n"
        "set title 'Зависимость логарифма модуля невязки от размера матрицы headGaussMethod' font 'Helvetica Bold, 10'\n";
    gp << "plot '-' with lines title 'dim->ln(|r|)' lc rgb 'blue'\n";
    gp.send1d(plotData);
    return plotData;
}

std::vector<std::pair<double, double>> testHeadGaussMethod_Time(size_t dim){
    //ИССЛЕДОВАНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ ОТ РАЗМЕРА МАТРИЦЫ headGaussMethod
    std::vector<std::pair<double, double>> plotData;
    for(int i = 3; i < dim; ++i){
        DenseMatrix<double> A = DenseMatrix<double>(i, i, GenerateMatrix<double>(i, -1000, 1000));
        std::vector<double> b = GenerateVector<double>(i, -1, 1);
        clock_t end, start = clock();
        std::vector<double> res = headGaussMethod(A, b);
        end = clock();
        plotData.emplace_back(log(i), log(double(end-start)/CLOCKS_PER_SEC));
        std::cout<<res;
    }
    Gnuplot gp;
    gp<<"set xlabel 'ln(Matrix Dimension)' \n"
        "set ylabel 'ln(Time)'\n"
        "set grid\n"
        "set title 'Зависимость времени выполнения от размера матрицы headGaussMethod' font 'Helvetica Bold, 10'\n";
    gp << "plot '-' with lines title 'ln(dim)->ln(time)' lc rgb 'blue'\n";
    gp.send1d(plotData);
    return plotData;
}

void compareStraightMethodsNorm(size_t dim){
    std::vector<std::pair<double, double>> pureGauss = testGaussMethod_Norm(dim);
    std::vector<std::pair<double, double>> headGauss = testHeadGaussMethod_Norm(dim);
    std::vector<std::pair<double, double>> LU = testLUMethod_Norm(dim);
    std::ofstream fout("pureGaussNormPlotData.txt");
    /*for(auto i: pureGauss){
        fout<<i.first<<" "<<i.second<<"\n";
    }
    fout.close();
    fout = std::ofstream("headGaussNormPlotData.txt");
    for(auto i: headGauss){
        fout<<i.first<<" "<<i.second<<"\n";
    }
    fout.close();*/
    Gnuplot gp;
    gp<<"set xlabel 'Matrix Dimension' \n"
        "set ylabel 'ln(|r|))'\n"
        "set grid\n"
        "set title 'Зависимость логарифма модуля невязки от размера матрицы разных методов' font 'Helvetica Bold, 10'\n";
    gp << "plot ";
    gp << "     '-' with lines title 'dim->ln(|r|), pureGauss' ls 11 lc rgb 'blue',"
          "     '-' with lines title 'dim->ln(|r|), headGauss' ls 11 lc rgb 'green',"
          "     '-' with lines title 'dim->ln(|r|), LU' ls 12 lc rgb 'red'\n";
    gp.send1d(pureGauss);
    gp.send1d(headGauss);
    gp.send1d(LU);
}



#endif //SOLECOURSE_TESTS_H
