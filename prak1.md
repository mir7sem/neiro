#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <thread>   // <-- Для многопоточности
#include <mutex>    // <-- Для защиты общих данных
#include <optional> // <-- Для передачи новой точки

#include <SDL2/SDL.h>

// --- Структуры для обмена данными между потоками ---
std::mutex g_data_mutex; // Глобальный мьютекс для защиты данных
std::vector<std::pair<double, double>> g_points; // Общий список точек
std::optional<std::pair<double, double>> g_new_point; // Место для новой точки

// --- Классы VgaSimulator, LinearApproximatorHDL и функции отрисовки (БЕЗ ИЗМЕНЕНИЙ) ---
// (Здесь находится полный код этих классов, он не изменился по сравнению с предыдущей версией)
// --- Настройки "VGA" экрана ---
constexpr int SCREEN_WIDTH = 640;
constexpr int SCREEN_HEIGHT = 480;
class VgaSimulator {
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    std::vector<uint32_t> framebuffer; 
public:
    VgaSimulator() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) throw std::runtime_error("SDL init failed");
        window = SDL_CreateWindow("VGA Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
        if (!window) throw std::runtime_error("Window creation failed");
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) throw std::runtime_error("Renderer creation failed");
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);
        if (!texture) throw std::runtime_error("Texture creation failed");
        framebuffer.resize(SCREEN_WIDTH * SCREEN_HEIGHT, 0);
    }
    ~VgaSimulator() {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    void clear(uint32_t color) { std::fill(framebuffer.begin(), framebuffer.end(), color); }
    void draw_pixel(int x, int y, uint32_t color) { if (x >= 0 && x < SCREEN_WIDTH && y >= 0 && y < SCREEN_HEIGHT) framebuffer[y * SCREEN_WIDTH + x] = color; }
    void present() {
        SDL_UpdateTexture(texture, NULL, framebuffer.data(), SCREEN_WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }
    bool process_events() {
        SDL_Event e;
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) return false;
        }
        return true;
    }
};
constexpr int FIXED_POINT_BITS = 10; 
constexpr long long SCALE = 1LL << FIXED_POINT_BITS;
constexpr long long LEARNING_RATE_FIXED = static_cast<long long>(0.01 * SCALE);
long long double_to_fixed(double val) { return static_cast<long long>(round(val * SCALE)); }
double fixed_to_double(long long fixed_val) { return static_cast<double>(fixed_val) / SCALE; }
class LinearApproximatorHDL {
private:
    long long m_fixed;
    long long b_fixed;
public:
    LinearApproximatorHDL() : m_fixed(0), b_fixed(0) {}
    void reset() { m_fixed = 0; b_fixed = 0; } // <-- НОВЫЙ МЕТОД для сброса
    void update(long long x_fixed, long long y_fixed) {
        long long product = m_fixed * x_fixed;
        long long y_pred_fixed = (product >> FIXED_POINT_BITS) + b_fixed;
        long long error_fixed = y_pred_fixed - y_fixed;
        long long grad_m_fixed = (error_fixed * x_fixed) >> FIXED_POINT_BITS;
        long long grad_b_fixed = error_fixed;
        long long m_update = (LEARNING_RATE_FIXED * grad_m_fixed) >> FIXED_POINT_BITS;
        long long b_update = (LEARNING_RATE_FIXED * grad_b_fixed) >> FIXED_POINT_BITS;
        m_fixed -= m_update;
        b_fixed -= b_update;
    }
    std::pair<double, double> getCoeffsDouble() const { return {fixed_to_double(m_fixed), fixed_to_double(b_fixed)}; }
};


void draw_point_on_vga(VgaSimulator& vga, int cx, int cy, uint32_t color) { for (int y = cy - 1; y <= cy + 1; ++y) for (int x = cx - 1; x <= cx + 1; ++x) vga.draw_pixel(x, y, color); }
void draw_line_bresenham(VgaSimulator& vga, int x0, int y0, int x1, int y1, uint32_t color) {
    int dx = std::abs(x1-x0), sx = x0<x1 ? 1 : -1;
    int dy = -std::abs(y1-y0), sy = y0<y1 ? 1 : -1;
    int err = dx+dy, e2;
    for(;;){
        vga.draw_pixel(x0,y0,color);
        if (x0==x1 && y0==y1) break;
        e2 = 2*err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}
struct CoordMapper {
    double world_x_min, world_x_max, world_y_min, world_y_max;
    CoordMapper(const std::vector<std::pair<double, double>>& points, double m, double b) {
        if (points.empty()) { world_x_min = -10; world_x_max = 10; world_y_min = -10; world_y_max = 10; } 
        else {
            world_x_min = points[0].first; world_x_max = points[0].first;
            world_y_min = points[0].second; world_y_max = points[0].second;
            for (const auto& p : points) {
                world_x_min = std::min(world_x_min, p.first);
                world_x_max = std::max(world_x_max, p.first);
                world_y_min = std::min(world_y_min, p.second);
                world_y_max = std::max(world_y_max, p.second);
            }
            double y_at_xmin = m * world_x_min + b;
            double y_at_xmax = m * world_x_max + b;
            world_y_min = std::min({world_y_min, y_at_xmin, y_at_xmax});
            world_y_max = std::max({world_y_max, y_at_xmin, y_at_xmax});
        }
        double x_padding = (world_x_max - world_x_min) * 0.1;
        double y_padding = (world_y_max - world_y_min) * 0.1;
        if (x_padding < 1.0) x_padding = 1.0;
        if (y_padding < 1.0) y_padding = 1.0;
        world_x_min -= x_padding; world_x_max += x_padding;
        world_y_min -= y_padding; world_y_max += y_padding;
    }
    std::pair<int, int> world_to_screen(double wx, double wy) {
        double world_w = world_x_max - world_x_min;
        double world_h = world_y_max - world_y_min;
        if (world_w < 1e-6) world_w = 1.0;
        if (world_h < 1e-6) world_h = 1.0;
        int sx = static_cast<int>((wx - world_x_min) / world_w * SCREEN_WIDTH);
        int sy = static_cast<int>(SCREEN_HEIGHT - ((wy - world_y_min) / world_h * SCREEN_HEIGHT));
        return {sx, sy};
    }
};

// --- НОВИНКА: Функция для потока ввода ---
void input_thread_func() {
    std::string line;
    while (true) {
        std::cout << "\nВведите точку (x,y) > ";
        if (!std::getline(std::cin, line) || line == "stop" || line == "exit" || line == "q") {
            // Сигнал основному потоку о завершении (необязательно, но хорошая практика)
            break; 
        }

        std::stringstream ss(line);
        double x, y;
        char comma;
        if (!(ss >> x >> comma >> y) || comma != ',') {
            std::cerr << "Ошибка ввода: неверный формат. Ожидается 'x,y'." << std::endl;
            continue;
        }

        // Блокируем мьютекс, чтобы безопасно записать новую точку
        std::lock_guard<std::mutex> lock(g_data_mutex);
        g_new_point = {x, y};
    }
}

// --- Основная программа (СУЩЕСТВЕННО ИЗМЕНЕНА) ---
int main(int argc, char* argv[]) {
    try {
        VgaSimulator vga;
        LinearApproximatorHDL approximator;
        
        std::cout << "Интерактивный режим линейной аппроксимации с симуляцией VGA." << std::endl;
        std::cout << "Для выхода введите 'stop' в консоли или закройте окно." << std::endl;
        
        // Запускаем поток для ввода данных в фоне
        std::thread input_thread(input_thread_func);
        input_thread.detach(); // Отсоединяем поток, чтобы он жил своей жизнью

        bool running = true;
        while (running) {
            // 1. Обработка событий окна (не блокирует программу)
            if (!vga.process_events()) {
                running = false; // Пользователь закрыл окно
            }
            


            // 2. Проверка, не появилась ли новая точка от потока ввода
            { // Начало критической секции
                std::lock_guard<std::mutex> lock(g_data_mutex);
                if (g_new_point) {
                    // Есть новая точка, обрабатываем ее
                    g_points.push_back(*g_new_point);
                    g_new_point.reset(); // Сбрасываем, чтобы не обработать дважды

                    // --- ИЗМЕНЕНИЕ ЛОГИКИ ОБУЧЕНИЯ ---
                    // Сбрасываем модель и обучаем заново на ВСЕХ точках
                    approximator.reset();
                    constexpr int EPOCHS = 500; // Количество прогонов по всему датасету
                    for (int i = 0; i < EPOCHS; ++i) {
                        for (const auto& p : g_points) {
                            approximator.update(double_to_fixed(p.first), double_to_fixed(p.second));
                        }
                    }
                    auto [m_curr, b_curr] = approximator.getCoeffsDouble();
                    printf("Текущие коэффициенты: m = %.4f, b = %.4f\n", m_curr, b_curr);
                }
            } // Конец критической секции, мьютекс автоматически освобождается

            // 3. Рендеринг текущего состояния (всегда, на каждом кадре)
            auto [m, b] = approximator.getCoeffsDouble();
            
            // Блокируем мьютекс на короткое время, только чтобы безопасно прочитать g_points
            std::vector<std::pair<double, double>> points_copy;
            {
                std::lock_guard<std::mutex> lock(g_data_mutex);
                points_copy = g_points;
            }

            CoordMapper mapper(points_copy, m, b);

            vga.clear(0xFF101010);
            
            auto origin = mapper.world_to_screen(0, 0);
            draw_line_bresenham(vga, 0, origin.second, SCREEN_WIDTH - 1, origin.second, 0xFF404040);
            draw_line_bresenham(vga, origin.first, 0, origin.first, SCREEN_HEIGHT - 1, 0xFF404040);

            for (const auto& p : points_copy) {
                auto [sx, sy] = mapper.world_to_screen(p.first, p.second);
                draw_point_on_vga(vga, sx, sy, 0xFF00A0FF);
            }

            if (points_copy.size() > 1) {
                auto p1 = mapper.world_to_screen(mapper.world_x_min, m * mapper.world_x_min + b);
                auto p2 = mapper.world_to_screen(mapper.world_x_max, m * mapper.world_x_max + b);
                draw_line_bresenham(vga, p1.first, p1.second, p2.first, p2.second, 0xFFFF4040);
            }
            
            vga.present();
            SDL_Delay(16); // ~60 FPS
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
