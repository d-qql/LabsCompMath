#include <iostream>
#include <functional>
#include "Dense/DenseMatrix.h"
#include "Dense/ThomasAlgorithm.h"
#include "utility/Overloads.h"

double p(double x) {
    return (x * x - 3);
}

double q(double x) {
    return (x * x - 3) * std::cos(x);
}

double f(double x) {
    return 2 - 6 * x + 2 * x * x * x + (x * x - 3) * std::exp(x) * std::sin(x) * (1 + std::cos(x)) +
           std::cos(x) * (std::exp(x) + (x * x - 1) + x * x * x * x - 3 * x * x);
}

std::pair<DenseMatrix<double>, std::vector<double>>
generateSystem(std::vector<double> &Xn, const std::function<double(double)> &p, const std::function<double(double)> &q,
               const std::function<double(double)> &f, double a, double b, int n, double alpha1,
               double beta1,
               double alpha2, double beta2, double gamma1, double gamma2) {
    std::vector<double> rightPart(n + 1);
    double h = (b - a) / n;
    Xn.resize(n + 1);
    for (int i = 0; i < Xn.size(); ++i) {
        Xn[i] = a + i * h;
    }
    DenseMatrix<double> A(n + 1, n + 1);
    A(0, 0) = alpha1 - beta1 / h;
    A(0, 1) = beta1 / h;
    rightPart[0] = gamma1;
    double ai, bi, ci;
    for (int i = 1; i < n; ++i) {
        ai = 1. / (h * h) - p(Xn[i]) / (2 * h);
        bi = -2. / (h * h) + q(Xn[i]);
        ci = 1. / (h * h) + p(Xn[i]) / (2 * h);
        A(i, i - 1) = ai;
        A(i, i) = bi;
        A(i, i + 1) = ci;
        rightPart[i] = f(Xn[i]);
    }
    A(n, n - 1) = -beta2 / h;
    A(n, n) = alpha2 + beta2 / h;
    rightPart[n] = gamma2;
    return {A, rightPart};
}

double linearInterpolation(const std::vector<double> &mesh, const std::vector<double> &solution, double point) {
    double step = mesh[1] - mesh[0];
    int n = std::floor((point - mesh[0]) / step);
    return solution[n] + (solution[n + 1] - solution[n]) / step * (point - n * step);
}

int main() {
    std::vector<double> mesh;
    auto system = generateSystem(mesh, p, q, f, 0, M_PI, 100, 1, 0, 1, 0, 0, M_PI * M_PI);
    std::cout << mesh << ThomasAlgorithm(system.first, system.second);
    auto solution = ThomasAlgorithm(system.first, system.second);
    std::cout << linearInterpolation(mesh, solution, 0.5) << " " << linearInterpolation(mesh, solution, 1) << " " <<
              linearInterpolation(mesh, solution, 1.5) << " " << linearInterpolation(mesh, solution, 2)
              << " " << linearInterpolation(mesh, solution, 2.5) << " " << linearInterpolation(mesh, solution, 3);
    return 0;
}
