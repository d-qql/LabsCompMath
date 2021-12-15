//
// Created by d-qql on 29.01.2021.
//

#ifndef SOLECOURSE_CSR_H
#define SOLECOURSE_CSR_H

#include <cstddef>
#include <vector>
#include <set>
#include "../utility/Triplet.h"
#include "../utility/Consts.h"
#include <algorithm>
#include "../Generators/RandVectorGenerator.h"

template<typename T>
class CSR{
public:
    using elm_t = T;
    using idx_t = std::size_t;

    template<typename EL>
    friend std::vector<EL> GaussSeidel(const CSR<EL> &A, const std::vector<EL> &b);
    template<typename EL>
    friend std::vector<EL> Yacobi(const CSR<EL> &A, const std::vector<EL> &b);
    template<typename EL>
    friend std::vector<EL> SOR(const CSR<EL> &A, const std::vector<EL> &b, const EL& w);
    template<typename EL>
    friend std::vector<EL> SSOR(const CSR<EL> &A, const std::vector<EL> &b, const EL& w);

private:

    const idx_t H, W;
    std::vector<elm_t> values;
    std::vector<idx_t> cols;
    std::vector<idx_t> rows;
public:

    CSR(const idx_t &h, const idx_t &w, const std::set<Triplet<elm_t>> &in): H(h), W(w){
        values.resize(in.size());
        cols.resize(in.size());
        rows.resize(H+1);
        int countInRow = 0;
        int currRow = 0;
        auto it = in.begin();                                   //получаем итератор на первый ненулевой элемент из set
        for(idx_t k = 0; k < in.size(); ++k){                   //проход по всем ненулевым элементам
            while(currRow < it->i){                             //Пока номера текущих обрабатываемых строк меньше номера строки текущего элемента, подлежащего вставке:
                rows[currRow + 1] = rows[currRow] + countInRow; //вставляем суммарное количество ненулевых элементов во всех предыдущих строках + в текущей
                ++currRow;                                       //переходим к заполнению следующей строки
                countInRow = 0;                                   //обнуляем счетчик вставленных в текущую строку элементов
            }                                                    //если номер текущей строки совпал с номером строки элемента, подлежащего вставке, то:
            values[k] = it->value;              //вставляем значение элемента
            cols[k] = it->j;                    //вставлем номер столбца элемента
            ++countInRow;                       //увеличиваем счетчик вставленных в текущую строку элементов
            it = next(it);                      //переходим к следующему элементу
        }

        for( ++currRow; currRow <= H; ++currRow ) rows[currRow] = in.size();  //если в конце currRow < H => имеются нулевые строки в конце матрицы
                                                                                // (необходимо заполнить их индексацию числом ненулевых элементов всей матрицы)
    }

    CSR(const idx_t &h, const idx_t &w, const std::vector<elm_t> &values, const std::vector<idx_t> &cols, const std::vector<idx_t> &rows): H(h), W(w){
        this->values = values;
        this->cols = cols;
        this->rows = rows;
    }

    [[nodiscard]] idx_t sizeH() const {
        return H;
    }
    [[nodiscard]] idx_t sizeW() const {
        return W;
    }

    elm_t operator()(idx_t const i, idx_t const j) const{
        idx_t skip = this->rows[i];
        idx_t count = this->rows[i+1] - this->rows[i];
        for (idx_t k = skip; k < skip+count; ++k) {
            if(this->cols[k] == j) return this->values[k];
        }
        return static_cast<elm_t>(0);
    }


    std::vector<elm_t> operator*(const std::vector<elm_t> &b) const {
        std::vector<elm_t> res(this->H);
        for(idx_t i = 0; i < this->H; ++i){
            res[i] = static_cast<T>(0);
            for(idx_t j = this->rows[i]; j < this->rows[i+1]; ++j) res[i] += this->values[j] * b[this->cols[j]];
        }

        return res;
    }

    std::pair<T, T> EVcircleLocalize() const{
        auto a = static_cast<T>(0);
        auto b = static_cast<T>(0);
        T r = 0;
        T center = 0;
        for (idx_t j = this->rows[0]; j < this->rows[1]; ++j) {     //проход по 0 строке
            if( cols[j] != 0 ) r += Tabs(values[j]);                //если номер столбца не равен номеру строки, то радиус увеличивается на модуль элемента
            else center = values[j];                                //иначе это диагональный элемент => является центром круга Гершгорина
        }
        a = center - r;
        b = center + r;
        for(size_t i = 1; i < H; ++i) {                                  //проход по оставшимся строкам
            r = 0;
            center = 0;
            for (idx_t j = this->rows[i]; j < this->rows[i + 1]; ++j) {  //проход по строке
                if( cols[j] != i ) r += Tabs(values[j]);                //если номер столбца не равен номеру строки, то радиус увеличивается на модуль элемента
                else center = values[j];                                //иначе это диагональный элемент => является центром круга Гершгорина
            }
            if( center - r < a) a = center - r;
            if( center + r > b) b = center + r;
        }
        return {a, b};
    }

    elm_t spectrum(){
        std::vector<elm_t> x = GenerateVector<elm_t>(this->W, -1, 1);
        x = static_cast<elm_t>(1) / norm(x) * x;
        std::vector<elm_t> prev_x(this->W);

        auto lambda = static_cast<elm_t>(0);
        elm_t prev_lambda;

        auto err = static_cast<elm_t>(100);
        while( err > tolerance<T>){
            prev_x = x;
            x = this->operator*(x);
            prev_lambda = lambda;
            lambda = x * prev_x;
            x = static_cast<elm_t>(1) / norm(x) * x;
            err = Tabs(prev_lambda - lambda) / std::max(Tabs(prev_lambda), Tabs(lambda));
            std::cout<<err<<std::endl;
        }
        return lambda;
    }

    CSR transpose() const{
        idx_t NonZero = values.size();
        std::vector<elm_t> tVals(NonZero);
        std::vector<idx_t> tCols(NonZero);
        std::vector<idx_t> tRows(W + 1);
        for(idx_t i = 0; i < NonZero; ++i) tRows[cols[i] + 1]++;  //Посчитали число ненулевых в каждой строке транспонированной матрицы
        idx_t S = 0;
        idx_t tmp;
        for(idx_t i = 1; i <= W; ++i){          //запишем в индекс каждой из строк суммы во всех предыдущих
            tmp = tRows[i];
            tRows[i] = S;
            S = S + tmp;
        }
        idx_t j1, j2, Col, RIndex, IIndex;
        elm_t V;
        for(idx_t i = 0; i < H; ++i){
            j1 = rows[i];
            j2 = rows[i + 1];
            Col = i;   //столбец в транспонированной = строка в исходной
            for(idx_t j = j1; j < j2; ++j){
                V = values[j];      //значение
                RIndex = cols[j];   //строка транспонированной
                IIndex = tRows[RIndex + 1];
                tVals[IIndex] = V;
                tCols[IIndex] = Col;
                tRows[RIndex + 1]++;    //дополним каждую строку числом вставленных в нее элементов
            }
        }
        return CSR(W, H, tVals, tCols, tRows);
    }
    void YacobiPrecond(){
        elm_t diag_elm;
        for(idx_t i = 0; i < this->H; ++i){
            diag_elm = this->operator()(i, i);
            for(idx_t j = this->rows[i]; j < this->rows[i+1]; ++j) values[j] /= diag_elm;
        }
    }
    void print() const{
        for(auto v: values) std::cout<<v<<" ";
        std::cout<<std::endl;
        for(auto j: cols) std::cout<<j<<" ";
        std::cout<<std::endl;
        for(auto i: rows) std::cout<<i<<" ";
        std::cout<<std::endl;
    }

};
template<typename T>
std::ostream& operator<<(std::ostream& os, const CSR<T> &Matrix){
    for(size_t i = 0; i < Matrix.sizeH(); ++i){
        for(size_t j = 0; j < Matrix.sizeW(); ++j){
            os<<Matrix(i, j)<<" ";
        }
        os<<std::endl;
    }
    return os;
}

#endif //SOLECOURSE_CSR_H
