#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <functional>

#include <SDL2/SDL.h>

// =============================================================================
// КОНФИГУРАЦИЯ
// =============================================================================
namespace Config {
		constexpr int SCREEN_WIDTH = 640;
		constexpr int SCREEN_HEIGHT = 480;
		constexpr int TARGET_FPS = 60;

		constexpr uint32_t BACKGROUND_COLOR = 0xFF101010;
		constexpr uint32_t GRID_COLOR = 0xFF404040;
		constexpr uint32_t POINT_COLOR = 0xFF00A0FF;
		constexpr uint32_t LINE_COLOR = 0xFFFF4040;

		constexpr int FIXED_POINT_BITS = 10;
		constexpr double LEARNING_RATE = 0.01;
		constexpr int TRAINING_EPOCHS = 500;

		constexpr double PADDING_FACTOR = 0.1;
		constexpr double MIN_PADDING = 1.0;
		constexpr int POINT_SIZE = 1;
}

// =============================================================================
// СТРУКТУРЫ ДАННЫХ
// =============================================================================
struct Point {
		double x, y;

		Point() : x(0), y(0) {}
		Point(double x_val, double y_val) : x(x_val), y(y_val) {}

		static Point parse(const std::string& input) {
				std::stringstream ss(input);
				double x, y;
				char comma;

				if (!(ss >> x >> comma >> y) || comma != ',') {
						throw std::invalid_argument("Неверный формат. Ожидается 'x,y'");
				}

				return Point(x, y);
		}
};

struct Bounds {
		double x_min, x_max, y_min, y_max;
};

// =============================================================================
// ЛИНЕЙНАЯ РЕГРЕССИЯ
// =============================================================================
class LinearRegression {
public:
		struct Coefficients {
				double slope = 0.0;
				double intercept = 0.0;
		};

private:
		static constexpr long long SCALE = 1LL << Config::FIXED_POINT_BITS;
		long long slope_fixed = 0;
		long long intercept_fixed = 0;

		static long long to_fixed(double value) {
				return static_cast<long long>(round(value * SCALE));
		}

		static double from_fixed(long long fixed_value) {
				return static_cast<double>(fixed_value) / SCALE;
		}

public:
		void reset() {
				slope_fixed = 0;
				intercept_fixed = 0;
		}

		void train(const std::vector<Point>& points) {
				if (points.empty()) return;

				reset();

				const long long learning_rate_fixed = to_fixed(Config::LEARNING_RATE);

				for (int epoch = 0; epoch < Config::TRAINING_EPOCHS; ++epoch) {
						for (const auto& point : points) {
								long long x_fixed = to_fixed(point.x);
								long long y_fixed = to_fixed(point.y);

								long long y_pred_fixed = ((slope_fixed * x_fixed) >> Config::FIXED_POINT_BITS) + intercept_fixed;
								long long error_fixed = y_pred_fixed - y_fixed;

								long long grad_slope = (error_fixed * x_fixed) >> Config::FIXED_POINT_BITS;
								long long grad_intercept = error_fixed;

								slope_fixed -= (learning_rate_fixed * grad_slope) >> Config::FIXED_POINT_BITS;
								intercept_fixed -= (learning_rate_fixed * grad_intercept) >> Config::FIXED_POINT_BITS;
						}
				}
		}

		Coefficients get_coefficients() const {
				return {from_fixed(slope_fixed), from_fixed(intercept_fixed)};
		}
};

// =============================================================================
// ГРАФИКА
// =============================================================================
class Graphics {
private:
		std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window;
		std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)> renderer;
		std::unique_ptr<SDL_Texture, decltype(&SDL_DestroyTexture)> texture;
		std::vector<uint32_t> framebuffer;

		Bounds calculate_bounds(const std::vector<Point>& points, 
													 const LinearRegression::Coefficients& coeffs) const {
				if (points.empty()) {
						return {-10.0, 10.0, -10.0, 10.0};
				}

				Bounds bounds;
				bounds.x_min = bounds.x_max = points[0].x;
				bounds.y_min = bounds.y_max = points[0].y;

				for (const auto& point : points) {
						bounds.x_min = std::min(bounds.x_min, point.x);
						bounds.x_max = std::max(bounds.x_max, point.x);
						bounds.y_min = std::min(bounds.y_min, point.y);
						bounds.y_max = std::max(bounds.y_max, point.y);
				}

				if (points.size() > 1) {
						double y_at_min = coeffs.slope * bounds.x_min + coeffs.intercept;
						double y_at_max = coeffs.slope * bounds.x_max + coeffs.intercept;
						bounds.y_min = std::min({bounds.y_min, y_at_min, y_at_max});
						bounds.y_max = std::max({bounds.y_max, y_at_min, y_at_max});
				}

				double x_padding = std::max((bounds.x_max - bounds.x_min) * Config::PADDING_FACTOR, 
																	 Config::MIN_PADDING);
				double y_padding = std::max((bounds.y_max - bounds.y_min) * Config::PADDING_FACTOR, 
																	 Config::MIN_PADDING);

				bounds.x_min -= x_padding;
				bounds.x_max += x_padding;
				bounds.y_min -= y_padding;
				bounds.y_max += y_padding;

				return bounds;
		}

		std::pair<int, int> world_to_screen(double wx, double wy, const Bounds& bounds) const {
				double world_width = bounds.x_max - bounds.x_min;
				double world_height = bounds.y_max - bounds.y_min;

				if (world_width < 1e-6) world_width = 1.0;
				if (world_height < 1e-6) world_height = 1.0;

				int sx = static_cast<int>((wx - bounds.x_min) / world_width * Config::SCREEN_WIDTH);
				int sy = static_cast<int>(Config::SCREEN_HEIGHT - 
																 ((wy - bounds.y_min) / world_height * Config::SCREEN_HEIGHT));

				return {sx, sy};
		}

		void draw_pixel(int x, int y, uint32_t color) {
				if (x >= 0 && x < Config::SCREEN_WIDTH && y >= 0 && y < Config::SCREEN_HEIGHT) {
						framebuffer[y * Config::SCREEN_WIDTH + x] = color;
				}
		}

		void draw_line(int x0, int y0, int x1, int y1, uint32_t color) {
				int dx = std::abs(x1 - x0);
				int dy = std::abs(y1 - y0);
				int sx = (x0 < x1) ? 1 : -1;
				int sy = (y0 < y1) ? 1 : -1;
				int err = dx - dy;

				while (true) {
						draw_pixel(x0, y0, color);

						if (x0 == x1 && y0 == y1) break;

						int e2 = 2 * err;
						if (e2 > -dy) {
								err -= dy;
								x0 += sx;
						}
						if (e2 < dx) {
								err += dx;
								y0 += sy;
						}
				}
		}

		void draw_point(int x, int y, uint32_t color) {
				for (int dy = -Config::POINT_SIZE; dy <= Config::POINT_SIZE; ++dy) {
						for (int dx = -Config::POINT_SIZE; dx <= Config::POINT_SIZE; ++dx) {
								draw_pixel(x + dx, y + dy, color);
						}
				}
		}

public:
		Graphics() 
				: window(nullptr, SDL_DestroyWindow)
				, renderer(nullptr, SDL_DestroyRenderer)
				, texture(nullptr, SDL_DestroyTexture)
		{
				if (SDL_Init(SDL_INIT_VIDEO) < 0) {
						throw std::runtime_error("Не удалось инициализировать SDL: " + std::string(SDL_GetError()));
				}

				window.reset(SDL_CreateWindow("Линейная регрессия", 
																		 SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
																		 Config::SCREEN_WIDTH, Config::SCREEN_HEIGHT, 
																		 SDL_WINDOW_SHOWN));
				if (!window) {
						throw std::runtime_error("Не удалось создать окно: " + std::string(SDL_GetError()));
				}

				renderer.reset(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED));
				if (!renderer) {
						throw std::runtime_error("Не удалось создать рендерер: " + std::string(SDL_GetError()));
				}

				texture.reset(SDL_CreateTexture(renderer.get(), SDL_PIXELFORMAT_ARGB8888, 
																			 SDL_TEXTUREACCESS_STREAMING,
																			 Config::SCREEN_WIDTH, Config::SCREEN_HEIGHT));
				if (!texture) {
						throw std::runtime_error("Не удалось создать текстуру: " + std::string(SDL_GetError()));
				}

				framebuffer.resize(Config::SCREEN_WIDTH * Config::SCREEN_HEIGHT);
		}

		~Graphics() {
				SDL_Quit();
		}

		bool handle_events() {
				SDL_Event event;
				while (SDL_PollEvent(&event)) {
						if (event.type == SDL_QUIT) {
								return false;
						}
				}
				return true;
		}

		void render_scene(const std::vector<Point>& points, const LinearRegression& regression) {
				auto coeffs = regression.get_coefficients();
				auto bounds = calculate_bounds(points, coeffs);

				// Очистка экрана
				std::fill(framebuffer.begin(), framebuffer.end(), Config::BACKGROUND_COLOR);

				// Рисование осей
				auto origin = world_to_screen(0, 0, bounds);
				draw_line(0, origin.second, Config::SCREEN_WIDTH - 1, origin.second, Config::GRID_COLOR);
				draw_line(origin.first, 0, origin.first, Config::SCREEN_HEIGHT - 1, Config::GRID_COLOR);

				// Рисование точек
				for (const auto& point : points) {
						auto [sx, sy] = world_to_screen(point.x, point.y, bounds);
						draw_point(sx, sy, Config::POINT_COLOR);
				}

				// Рисование линии регрессии
				if (points.size() > 1 && (std::abs(coeffs.slope) > 1e-10 || std::abs(coeffs.intercept) > 1e-10)) {
						auto p1 = world_to_screen(bounds.x_min, coeffs.slope * bounds.x_min + coeffs.intercept, bounds);
						auto p2 = world_to_screen(bounds.x_max, coeffs.slope * bounds.x_max + coeffs.intercept, bounds);
						draw_line(p1.first, p1.second, p2.first, p2.second, Config::LINE_COLOR);
				}

				// Вывод на экран
				SDL_UpdateTexture(texture.get(), nullptr, framebuffer.data(), 
												 Config::SCREEN_WIDTH * sizeof(uint32_t));
				SDL_RenderClear(renderer.get());
				SDL_RenderCopy(renderer.get(), texture.get(), nullptr, nullptr);
				SDL_RenderPresent(renderer.get());
		}
};

// =============================================================================
// ОСНОВНОЕ ПРИЛОЖЕНИЕ
// =============================================================================
class Application {
private:
		Graphics graphics;
		LinearRegression regression;

		std::vector<Point> points;
		std::mutex points_mutex;
		bool running = true;

		void add_point(const Point& point) {
				{
						std::lock_guard<std::mutex> lock(points_mutex);
						points.push_back(point);
						regression.train(points);
				}

				auto coeffs = regression.get_coefficients();
				std::printf("Добавлена точка (%.2f, %.2f)\n", point.x, point.y);
				std::printf("Уравнение: y = %.4fx + %.4f\n\n", coeffs.slope, coeffs.intercept);
		}

		void start_input_thread() {
				std::thread([this]() {
						std::string line;
						std::cout << "=== Интерактивная линейная регрессия ===\n";
						std::cout << "Введите точки в формате 'x,y' или 'quit' для выхода\n";
						std::cout << "Примеры: 1,2 или 3.5,4.2 или -1,-2\n\n";

						while (running) {
								std::cout << "Точка > ";
								if (!std::getline(std::cin, line)) {
										break;
								}

								if (line == "quit" || line == "exit" || line == "q") {
										running = false;
										break;
								}

								if (line.empty()) continue;

								try {
										Point point = Point::parse(line);
										add_point(point);
								} catch (const std::exception& e) {
										std::cerr << "Ошибка: " << e.what() << std::endl;
								}
						}
				}).detach();
		}

public:
		void run() {
				start_input_thread();

				const auto frame_duration = std::chrono::milliseconds(1000 / Config::TARGET_FPS);

				while (running) {
						auto frame_start = std::chrono::steady_clock::now();

						if (!graphics.handle_events()) {
								running = false;
								break;
						}

						{
								std::lock_guard<std::mutex> lock(points_mutex);
								graphics.render_scene(points, regression);
						}

						auto frame_end = std::chrono::steady_clock::now();
						auto elapsed = frame_end - frame_start;
						if (elapsed < frame_duration) {
								std::this_thread::sleep_for(frame_duration - elapsed);
						}
				}
		}
};

// =============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// =============================================================================
int main() {
		try {
				Application app;
				app.run();
		} catch (const std::exception& e) {
				std::cerr << "Критическая ошибка: " << e.what() << std::endl;
				return 1;
		}

		return 0;
}
