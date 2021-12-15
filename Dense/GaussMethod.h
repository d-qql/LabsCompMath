//
// Created by d-qql on 26.01.2021.
//

#ifndef SOLECOURSE_GAUSSMETHOD_H
#define SOLECOURSE_GAUSSMETHOD_H

#include <vector>
#include "DenseMatrix.h"
#include "../utility/Consts.h"
#include "BackSubstitution.h"
template<typename T>
typename DenseMatrix<T>::idx_t col_nonzero(const DenseMatrix<T> &A, const typename DenseMatrix<T>::idx_t& col) {

    using idx_t = typename DenseMatrix<T>::idx_t;

    if( Tabs(A(col, col)) > tolerance<T>){
        return col;  //если диагональный не 0, то вернем индекс строки
    } else {
        for (idx_t i = col + 1; i < A.sizeH(); ++i) {     //поиск ненулевого элемента в столбце col под диагональю, если диагональный 0
            if (Tabs(A(i, col)) > tolerance<T>) {
                return i;
            }
        }
    }
    return col;
}

template <typename T>
unsigned int triangulation(DenseMatrix<T> &A, std::vector<T> &b) {

    using idx_t = typename DenseMatrix<T>::idx_t;
    using elm_t = typename DenseMatrix<T>::elm_t;

    unsigned int swapCount = 0;

    for (idx_t i = 0; i < A.sizeH()-1; ++i) {
        std::cout<<i<<"\n";
        idx_t iNonZero = col_nonzero(A, i); //индекс максимального по модулю элемента под диагональю
        if (Tabs(A(iNonZero, i)) < tolerance<T>) {
            continue;                          //пропускаем столбец, если под диагональю все нули
        } else {
            if (i != iNonZero) {
                A.swap(i, iNonZero);        //переставляем строку с максимальным по модулю элементом с текущей местами
                std::swap(b[i], b[iNonZero]); //элемент свободного столбца также переставляется
                ++swapCount;
            }

            for (idx_t k = i + 1; k < A.sizeH(); ++k) { //проход по всем строкам под текущей
                elm_t coef = (A(k, i) / A(i, i));
                for (idx_t j = i; j < A.sizeW(); ++j) { //проход по всем элементам строки
                    A(k, j) -= A(i, j) * coef;          //вычитание из строки k строки i, умноженной на коэффициент для зануления элементов столбца i
                }
                b[k] -= b[i] * coef; //вычитание свободного члена уравнения
            }
        }
    }
    return swapCount;   //возвращаем число перестановок(может потребоваться для поиска определителя)
}

template <typename T>
std::vector<T> GaussMethod(DenseMatrix<T> A,
                               std::vector<T> b) {
    triangulation(A, b); //приводим матрицу к верхнетреугольной с выбором главного элемента
    return backSubstTopTriangular(A, b);
}


#endif //SOLECOURSE_GAUSSMETHOD_H
