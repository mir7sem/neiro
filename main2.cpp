#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <mutex>
#include <optional>

#include <SDL2/SDL.h>

// Подключаем наши новые модули
#include "NeuroProcessor.h"
#include "Trainer.h"

// --- Структуры для обмена данными между потоками ---
std::mutex g_data_mutex;
std::vector<std::pair<double, double>> g_points;
std::optional<std::pair<double, double>> g_new_point;
constexpr size_t MAX_POINTS = 1000; // Ограничение на количество точек

// --- НАЧАЛО ВСТАВЛЕННОГО КОДА ---
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
// --- КОНЕЦ ВСТАВЛЕННОГО КОДА ---


// --- Функция для потока ввода (с добавлением лимита) ---
void input_thread_func() {
    std::string line;
    while (true) {
        std::cout << "\nВведите точку (x,y) > ";
        if (!std::getline(std::cin, line) || line == "stop" || line == "exit" || line == "q") {
            break;
        }

        std::stringstream ss(line);
        double x, y;
        char comma;
        if (!(ss >> x >> comma >> y) || comma != ',') {
            std::cerr << "Ошибка ввода: неверный формат. Ожидается 'x,y'." << std::endl;
            continue;
        }

        std::lock_guard<std::mutex> lock(g_data_mutex);
        if (g_points.size() < MAX_POINTS) {
            g_new_point = {x, y};
        } else {
            std::cout << "Достигнут лимит в " << MAX_POINTS << " точек. Новые точки не принимаются." << std::endl;
        }
    }
}

// --- Основная программа ---
int main(int argc, char* argv[]) {
    try {
        VgaSimulator vga;
        NeuroProcessor neuro_processor;

        std::cout << "Интерактивный режим с предобученным нейропроцессором." << std::endl;
        std::cout << "Максимальное количество точек: " << MAX_POINTS << std::endl;
        std::cout << "Для выхода введите 'stop' в консоли или закройте окно." << std::endl;

        std::thread input_thread(input_thread_func);
        input_thread.detach();

        bool running = true;
        while (running) {
            if (!vga.process_events()) {
                running = false;
            }

            {
                std::lock_guard<std::mutex> lock(g_data_mutex);
                if (g_new_point) {
                    g_points.push_back(*g_new_point);
                    g_new_point.reset();

                    std::cout << "Новая точка добавлена. Пересчет весов..." << std::endl;
                    auto [new_m, new_b] = Trainer::calculate_weights_normal_equation(g_points);
                    neuro_processor.load_weights(new_m, new_b);
                    printf("Веса загружены в нейропроцессор: m = %.4f, b = %.4f\n", new_m, new_b);
                }
            }

            auto [m, b] = neuro_processor.get_coeffs();

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
            SDL_Delay(16);
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}