cmake_minimum_required(VERSION 3.21)
project(Labs)

set(CMAKE_CXX_STANDARD 20)
include_directories(/usr/boost/include)
link_directories(/usr/boost/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -O3 -march=native")
add_executable(Labs main.cpp Dense/DenseMatrix.h utility/Triplet.h utility/Consts.h Dense/GaussMethod.h Dense/HeadGaussMethod.h Generators/RandMatrixGenerator.h utility/Overloads.h Dense/ThomasAlgorithm.h Generators/RandVectorGenerator.h Generators/RandTridiagonalMatrixGen.h Dense/BackSubstitution.h Dense/LU.h Sparse/CSR.h Sparse/SimpleIteration.h Tests/Tests.h Sparse/Gauss-Seidel.h Dense/Householder.h Sparse/Yacobi.h Chebyshev/Chebyshev.h Sparse/SOR.h Sparse/SSOR.h Sparse/GradientDescent.h Krylov/Krylov.h Sparse/CG.h Givens/Rotations.h GMRES/GMRES.h Sparse/BiCG.h Sparse/BiCGstab.h)
target_link_libraries(Labs -lboost_iostreams -lboost_system -lboost_filesystem)