#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>

// ѕроста€ структура дл€ матричных операций, необходима€ дл€ нормального уравнени€.
struct Matrix {
    size_t rows, cols;
    std::vector<double> data;

    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& at(size_t r, size_t c) {
        return data[r * cols + c];
    }

    const double& at(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    static Matrix multiply(const Matrix& a, const Matrix& b) {
        if (a.cols != b.rows) throw std::runtime_error("Matrix dimensions mismatch for multiplication");
        Matrix result(a.rows, b.cols);
        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t j = 0; j < b.cols; ++j) {
                for (size_t k = 0; k < a.cols; ++k) {
                    result.at(i, j) += a.at(i, k) * b.at(k, j);
                }
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(j, i) = at(i, j);
            }
        }
        return result;
    }

    // јналитическое решение дл€ обратной матрицы 2x2
    Matrix inverse_2x2() const {
        if (rows != 2 || cols != 2) throw std::runtime_error("Inverse is implemented only for 2x2 matrices");
        
        double a = at(0, 0), b = at(0, 1);
        double c = at(1, 0), d = at(1, 1);

        double det = a * d - b * c;
        if (std::abs(det) < 1e-9) throw std::runtime_error("Matrix is singular, cannot find inverse");

        Matrix result(2, 2);
        result.at(0, 0) = d / det;
        result.at(0, 1) = -b / det;
        result.at(1, 0) = -c / det;
        result.at(1, 1) = a / det;
        return result;
    }
};