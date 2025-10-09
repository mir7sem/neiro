#pragma once
#include <utility>

// Моделирует нейропроцессор для линейной аппроксимации.
// Это, по сути, один нейрон с двумя входами (x и bias=1) и линейной функцией активации.
// Он НЕ обучается. Он только использует готовые веса.
class NeuroProcessor {
private:
    // Веса модели: m - наклон, b - смещение (bias)
    double m_weight; 
    double b_weight;

public:
    NeuroProcessor() : m_weight(0.0), b_weight(0.0) {}

    // Загрузка предварительно рассчитанных весов в процессор
    void load_weights(double m, double b) {
        m_weight = m;
        b_weight = b;
    }

    // Прямой проход (инференс): вычисление выхода по входу
    double process(double x_input) const {
        // y = m*x + b
        return m_weight * x_input + b_weight;
    }

    // Получение текущих коэффициентов для внешних нужд (например, отрисовки)
    std::pair<double, double> get_coeffs() const {
        return {m_weight, b_weight};
    }
};