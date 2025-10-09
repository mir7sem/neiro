#pragma once
#include <utility>

// ���������� �������������� ��� �������� �������������.
// ���, �� ����, ���� ������ � ����� ������� (x � bias=1) � �������� �������� ���������.
// �� �� ���������. �� ������ ���������� ������� ����.
class NeuroProcessor {
private:
    // ���� ������: m - ������, b - �������� (bias)
    double m_weight; 
    double b_weight;

public:
    NeuroProcessor() : m_weight(0.0), b_weight(0.0) {}

    // �������� �������������� ������������ ����� � ���������
    void load_weights(double m, double b) {
        m_weight = m;
        b_weight = b;
    }

    // ������ ������ (��������): ���������� ������ �� �����
    double process(double x_input) const {
        // y = m*x + b
        return m_weight * x_input + b_weight;
    }

    // ��������� ������� ������������� ��� ������� ���� (��������, ���������)
    std::pair<double, double> get_coeffs() const {
        return {m_weight, b_weight};
    }
};