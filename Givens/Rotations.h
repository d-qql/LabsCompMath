//
// Created by d-qql on 28.03.2021.
//

#ifndef SOLECOURSE_ROTATIONS_H
#define SOLECOURSE_ROTATIONS_H

#include "../Dense/DenseMatrix.h"
#include "../utility/Consts.h"
#include "../utility/Overloads.h"
template<typename T>
void HessenbergRot(DenseMatrix<T>& H, std::vector<T>& b, size_t currSZ, std::vector<std::pair<T, T>>& rotations){
    T cos;
    T sin;
    T up;
    T down;
    T Bup;
    T Bdown;
    //rot.first = cos, .second = sin;
    for(int k = 0; k < rotations.size(); ++k) { //повернем последний столбец всеми предыдущими поворотами
        cos = rotations[k].first;
        sin = rotations[k].second;
        up = cos * H(k, currSZ) + sin * H(k + 1, currSZ);
        down = -sin * H(k, currSZ) + cos * H(k + 1, currSZ);
        if (Tabs(down) < tolerance<T>) down = 0;
        H(k, currSZ) = up;
        H(k + 1, currSZ) = down;
    }

        //посчитаем новый поворот
    cos = H(currSZ, currSZ)/sqrt(pow(H(currSZ, currSZ), 2) + pow(H(currSZ + 1, currSZ), 2));
    sin = H(currSZ + 1, currSZ)/sqrt(pow(H(currSZ, currSZ), 2) + pow(H(currSZ + 1, currSZ), 2));
    //повернем вектор новым поворотом
    Bup = cos * b[currSZ] + sin * b[currSZ+1];
    Bdown = -sin * b[currSZ] + cos * b[currSZ+1];
    b[currSZ] = Bup;
    b[currSZ+1] = Bdown;
    //повернем последний столбец новым поворотом
    up = cos * H(currSZ, currSZ) + sin * H(currSZ + 1, currSZ);
    down = -sin * H(currSZ, currSZ) + cos * H(currSZ + 1, currSZ);
    if(Tabs(down) < tolerance<T>) down = 0;
    H(currSZ, currSZ) = up;
    H(currSZ + 1, currSZ) = down;

    //запомним новый поворот
    rotations.push_back({cos, sin});

   // std::cout<<H;
}

#endif //SOLECOURSE_ROTATIONS_H
