//
// Created by d-qql on 26.01.2021.
//

#ifndef SOLECOURSE_TRIPLET_H
#define SOLECOURSE_TRIPLET_H

template<typename elm_t, typename idx_t = std::size_t>
struct Triplet{
    elm_t value;
    idx_t i;
    idx_t j;
    bool operator<(Triplet<elm_t> const &rgh) const{
        return this->i<rgh.i || (this->i == rgh.i && this->j < rgh.j);
    }
};
#endif //SOLECOURSE_TRIPLET_H
