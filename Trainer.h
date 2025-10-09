#pragma once
#include "Matrix.h"
#include <vector>
#include <utility>

// Класс-утилита для обучения. Не хранит состояние.
// Вычисляет веса, используя матричный метод (нормальное уравнение).
class Trainer {
public:
    // Статический метод, который принимает набор данных и возвращает оптимальные веса [m, b]
    static std::pair<double, double> calculate_weights_normal_equation(const std::vector<std::pair<double, double>>& points) {
        if (points.size() < 2) {
            // Невозможно построить линию по одной точке, возвращаем нули.
            return {0.0, 0.0};
        }

        size_t n = points.size();
        
        // 1. Создаем матрицу X (design matrix). Размер n x 2.
        // Каждая строка [x_i, 1]. '1' - это для коэффициента смещения b.
        Matrix X(n, 2);
        // 2. Создаем вектор y. Размер n x 1.
        Matrix Y(n, 1);

        for (size_t i = 0; i < n; ++i) {
            X.at(i, 0) = points[i].first;  // x_i
            X.at(i, 1) = 1.0;              // bias term
            Y.at(i, 0) = points[i].second; // y_i
        }

        try {
            // 3. Реализуем формулу: theta = (X^T * X)^-1 * X^T * Y
            Matrix Xt = X.transpose();              // X^T
            Matrix XtX = Matrix::multiply(Xt, X);   // X^T * X
            Matrix XtX_inv = XtX.inverse_2x2();     // (X^T * X)^-1
            Matrix XtY = Matrix::multiply(Xt, Y);   // X^T * Y
            
            Matrix theta = Matrix::multiply(XtX_inv, XtY); // theta

            double m = theta.at(0, 0);
            double b = theta.at(1, 0);

            return {m, b};
        } catch (const std::runtime_error& e) {
            std::cerr << "Ошибка при вычислении весов: " << e.what() << ". Возвращены нулевые веса." << std::endl;
            return {0.0, 0.0};
        }
    }
};