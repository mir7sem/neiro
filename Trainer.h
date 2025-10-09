#pragma once
#include "Matrix.h"
#include <vector>
#include <utility>

// �����-������� ��� ��������. �� ������ ���������.
// ��������� ����, ��������� ��������� ����� (���������� ���������).
class Trainer {
public:
    // ����������� �����, ������� ��������� ����� ������ � ���������� ����������� ���� [m, b]
    static std::pair<double, double> calculate_weights_normal_equation(const std::vector<std::pair<double, double>>& points) {
        if (points.size() < 2) {
            // ���������� ��������� ����� �� ����� �����, ���������� ����.
            return {0.0, 0.0};
        }

        size_t n = points.size();
        
        // 1. ������� ������� X (design matrix). ������ n x 2.
        // ������ ������ [x_i, 1]. '1' - ��� ��� ������������ �������� b.
        Matrix X(n, 2);
        // 2. ������� ������ y. ������ n x 1.
        Matrix Y(n, 1);

        for (size_t i = 0; i < n; ++i) {
            X.at(i, 0) = points[i].first;  // x_i
            X.at(i, 1) = 1.0;              // bias term
            Y.at(i, 0) = points[i].second; // y_i
        }

        try {
            // 3. ��������� �������: theta = (X^T * X)^-1 * X^T * Y
            Matrix Xt = X.transpose();              // X^T
            Matrix XtX = Matrix::multiply(Xt, X);   // X^T * X
            Matrix XtX_inv = XtX.inverse_2x2();     // (X^T * X)^-1
            Matrix XtY = Matrix::multiply(Xt, Y);   // X^T * Y
            
            Matrix theta = Matrix::multiply(XtX_inv, XtY); // theta

            double m = theta.at(0, 0);
            double b = theta.at(1, 0);

            return {m, b};
        } catch (const std::runtime_error& e) {
            std::cerr << "������ ��� ���������� �����: " << e.what() << ". ���������� ������� ����." << std::endl;
            return {0.0, 0.0};
        }
    }
};