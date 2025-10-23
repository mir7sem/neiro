# Практическая работа №1
### 1. Программа вычисления весов
#### Код
`generate_weights.cpp`:
```
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdint>

// --- Matrix struct and functions ---
struct Matrix { size_t r,c; std::vector<double> d; Matrix(size_t R=0,size_t C=0):r(R),c(C),d(R*C,0.0){} void randomize(unsigned int s){std::mt19937 g(s);std::uniform_real_distribution<> u(-1.,1.);for(auto&v:d)v=u(g)*std::sqrt(2.0/(r+c));} double& at(size_t R,size_t C){return d[R*c+C];} const double& at(size_t R,size_t C)const{return d[R*c+C];} };
Matrix multiply(const Matrix&a,const Matrix&b){Matrix r(a.r,b.c);for(size_t i=0;i<a.r;++i)for(size_t j=0;j<b.c;++j)for(size_t k=0;k<a.c;++k)r.at(i,j)+=a.at(i,k)*b.at(k,j);return r;}
Matrix add_bias(const Matrix&a,const Matrix&b){Matrix r=a;for(size_t i=0;i<a.r;++i)for(size_t j=0;j<a.c;++j)r.at(i,j)+=b.at(0,j);return r;}
Matrix apply_relu(const Matrix&m){Matrix r=m;for(auto&v:r.d)v=std::max(0.0,v);return r;}
Matrix relu_derivative(const Matrix&m){Matrix r=m;for(auto&v:r.d)v=(v>0)?1.0:0.0;return r;}
Matrix transpose(const Matrix&m){Matrix r(m.c,m.r);for(size_t i=0;i<m.r;++i)for(size_t j=0;j<m.c;++j)r.at(j,i)=m.at(i,j);return r;}
Matrix hadamard(const Matrix&a,const Matrix&b){Matrix r(a.r,a.c);for(size_t i=0;i<a.d.size();++i)r.d[i]=a.d[i]*b.d[i];return r;}
Matrix sum_rows(const Matrix&m){Matrix r(1,m.c);for(size_t j=0;j<m.c;++j)for(size_t i=0;i<m.r;++i)r.at(0,j)+=m.at(i,j);return r;}

// Функция для нормализации значения в диапазон [-1, 1]
double normalize(double val, double min, double max) { return 2.0 * (val - min) / (max - min) - 1.0; }

int main() {
    const size_t NUM_POINTS = 3, INPUT_SIZE=NUM_POINTS*2, HIDDEN1_SIZE=32, HIDDEN2_SIZE=16, OUTPUT_SIZE=2;
    Matrix W1(INPUT_SIZE,HIDDEN1_SIZE),B1(1,HIDDEN1_SIZE),W2(HIDDEN1_SIZE,HIDDEN2_SIZE),B2(1,HIDDEN2_SIZE),W3(HIDDEN2_SIZE,OUTPUT_SIZE),B3(1,OUTPUT_SIZE);
    W1.randomize(1);B1.randomize(2);W2.randomize(3);B2.randomize(4);W3.randomize(5);B3.randomize(6);

    double learning_rate=0.001; int steps=80000; int batch_size=128;
    std::mt19937 gen(1337);
    const double M_B_MIN=-5.0, M_B_MAX=5.0;
    const double X_MIN=-10.0, X_MAX=10.0;
    const double Y_MIN=M_B_MIN*X_MIN+M_B_MIN, Y_MAX=M_B_MAX*X_MAX+M_B_MAX;
    std::uniform_real_distribution<>m_b_dist(M_B_MIN,M_B_MAX), x_dist(X_MIN,X_MAX);

    std::cout << "Обучаем сеть с нормализацией данных..." << std::endl;
    for (int step=0; step<steps; ++step) {
        Matrix X_batch(batch_size,INPUT_SIZE), Y_batch(batch_size,OUTPUT_SIZE);
        for (int i=0; i<batch_size; ++i) {
            double true_m=m_b_dist(gen), true_b=m_b_dist(gen);
            for (int p=0; p<NUM_POINTS; ++p) {
                double x=x_dist(gen), y=true_m*x+true_b;
                X_batch.at(i,p*2+0)=normalize(x,X_MIN,X_MAX);
                X_batch.at(i,p*2+1)=normalize(y,Y_MIN,Y_MAX);
            }
            Y_batch.at(i,0)=normalize(true_m,M_B_MIN,M_B_MAX);
            Y_batch.at(i,1)=normalize(true_b,M_B_MIN,M_B_MAX);
        }
        Matrix Z1=add_bias(multiply(X_batch,W1),B1),A1=apply_relu(Z1);
        Matrix Z2=add_bias(multiply(A1,W2),B2),A2=apply_relu(Z2);
        Matrix Z3=add_bias(multiply(A2,W3),B3),Y_pred=Z3;
        Matrix error(batch_size,OUTPUT_SIZE); for(size_t i=0;i<error.d.size();++i)error.d[i]=Y_pred.d[i]-Y_batch.d[i];
        Matrix dZ3=error,dW3=multiply(transpose(A2),dZ3),dB3=sum_rows(dZ3),dA2=multiply(dZ3,transpose(W3));
        Matrix dZ2=hadamard(dA2,relu_derivative(Z2)),dW2=multiply(transpose(A1),dZ2),dB2=sum_rows(dZ2),dA1=multiply(dZ2,transpose(W2));
        Matrix dZ1=hadamard(dA1,relu_derivative(Z1)),dW1=multiply(transpose(X_batch),dZ1),dB1=sum_rows(dZ1);
        double N=static_cast<double>(batch_size);
        for(size_t i=0;i<W1.d.size();++i)W1.d[i]-=learning_rate*dW1.d[i]/N; for(size_t i=0;i<B1.d.size();++i)B1.d[i]-=learning_rate*dB1.d[i]/N;
        for(size_t i=0;i<W2.d.size();++i)W2.d[i]-=learning_rate*dW2.d[i]/N; for(size_t i=0;i<B2.d.size();++i)B2.d[i]-=learning_rate*dB2.d[i]/N;
        for(size_t i=0;i<W3.d.size();++i)W3.d[i]-=learning_rate*dW3.d[i]/N; for(size_t i=0;i<B3.d.size();++i)B3.d[i]-=learning_rate*dB3.d[i]/N;
        if(step%5000==0){double loss=0;for(const auto&e:error.d)loss+=e*e;std::cout<<"Шаг "<<std::setw(5)<<step<<", Потери: "<<loss/N<<std::endl;}
    }
    
    // Квантизация весов в целые числа
    std::cout << "Квантизация весов в целые числа..." << std::endl;
    const int64_t SCALE_FACTOR = 100000; // Коэффициент масштабирования
    
    auto quantize_matrix = [&](const Matrix& m, std::vector<int64_t>& quantized) {
        quantized.resize(m.d.size());
        for (size_t i = 0; i < m.d.size(); ++i) {
            quantized[i] = static_cast<int64_t>(std::round(m.d[i] * SCALE_FACTOR));
        }
    };
    
    std::vector<int64_t> qW1, qB1, qW2, qB2, qW3, qB3;
    quantize_matrix(W1, qW1);
    quantize_matrix(B1, qB1);
    quantize_matrix(W2, qW2);
    quantize_matrix(B2, qB2);
    quantize_matrix(W3, qW3);
    quantize_matrix(B3, qB3);
    
    std::ofstream outfile("network_weights.txt");
    outfile << SCALE_FACTOR << "\n";
    outfile << X_MIN << " " << X_MAX << "\n";
    outfile << Y_MIN << " " << Y_MAX << "\n";
    outfile << M_B_MIN << " " << M_B_MAX << "\n";
    
    auto save_quantized_matrix = [&](size_t r, size_t c, const std::vector<int64_t>& data) {
        outfile << r << " " << c << "\n";
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                outfile << data[i * c + j] << (j == c - 1 ? "" : " ");
            }
            outfile << "\n";
        }
    };
    
    save_quantized_matrix(W1.r, W1.c, qW1);
    save_quantized_matrix(B1.r, B1.c, qB1);
    save_quantized_matrix(W2.r, W2.c, qW2);
    save_quantized_matrix(B2.r, B2.c, qB2);
    save_quantized_matrix(W3.r, W3.c, qW3);
    save_quantized_matrix(B3.r, B3.c, qB3);
    
    outfile.close();
    std::cout << "Целочисленные веса и параметры сохранены." << std::endl;
    std::cout << "Коэффициент масштабирования: " << SCALE_FACTOR << std::endl;
    
    return 0;
}
```

#### Компиляция
```
g++ generate_weights.cpp -o generate_weights.exe -std=c++17 -O2
```
Либо через кодировку CP1251 (рекомендовано):
```
g++ generate_weights.cpp -o generate_weights.exe -std=c++17 -O2 -finput-charset=CP1251 -fexec-charset=CP866
```
#### Запуск
Программа автоматически сгенерирует веса и сохранит в файл `network_weights.txt`.
### 2. Веса
`network_weights.txt`:
```
100000
-10 10
45 55
-5 5
6 32
12411 25723 -17107 44849 -12116 -30969 -5461 7787 19999 14235 -22819 -1921 11280 -8856 8380 18993 -24168 -26899 33918 9857 22250 5199 -23629 17910 17935 34484 32497 -16527 -15260 -5498 -1044 18014
4761 -16346 -6542 -16194 12998 -3928 -15669 5691 7362 -12110 -6290 -5424 -14010 -5894 -14576 18508 -20972 2909 -7274 22899 16442 -10487 8922 -7364 4233 10186 -5290 -7318 -1434 4080 -9157 7313
37559 -29529 -15395 165 -21580 48774 11 -2387 18007 -10286 14366 9403 -5167 15124 9778 -17146 -50 -4937 -4168 11374 -28331 19230 36436 -34591 6462 -30073 19764 -24934 -26406 39035 -7749 16769
-2732 -6967 13032 -18536 3596 -21020 4395 -1333 12126 6616 -10782 22856 592 -18079 810 11139 -17615 480 -5942 5908 -7686 -21113 -4705 -17324 -6476 -18092 -2677 -15272 -3050 -8546 14718 -9788
-41675 -28943 4544 -8630 -6071 19302 4423 9973 5196 -13192 15817 -3904 12966 -10155 -28314 20059 35319 35573 -2930 -20300 23020 -21929 -24621 -15257 -21452 -19911 -10865 -20408 55475 -14875 -32113 4071
-4059 -8122 23219 10788 11897 -301 -22410 14403 12030 3020 -4354 -13544 -14983 2748 -11316 19108 -9702 -3837 11064 -4376 -21450 -17636 -10923 1709 -18804 -15118 4069 1743 5607 2307 -10038 -1842
1 32
-6819 22675 21497 1453 -8828 -5964 2815 -18714 -714 23064 14642 9943 -2389 3148 19374 -11891 -7857 -4345 -15132 22486 -11665 -16021 -6863 -26122 3313 3513 -26213 22660 11327 -18018 -8997 -9276
32 16
-17637 12654 -32078 -10062 11924 -19770 -27842 -24859 -23032 7867 -14478 -497 -11575 6542 -381 12731
10338 17832 15180 -8997 5804 9618 -12031 -2175 16321 13228 -9897 17154 7325 -31079 -18045 -5650
12350 -4115 -3801 -3440 11455 8737 -19787 -7510 1301 10137 4475 -11618 6043 7270 14680 12128
-7261 -17656 14911 -25337 25178 3935 -944 -2923 -7061 -1268 12589 -6139 13000 -15158 -1633 18897
886 -14591 -8064 12535 -7376 -19546 -717 -19347 234 17066 -11963 17389 -9091 -13226 20016 7865
8341 -16253 -42521 17197 -22685 -17852 1419 -3809 4957 2880 15559 13040 -6683 26580 14092 5458
-9451 13719 1614 15451 -16431 -7968 15314 8847 -9687 -1315 103 -10257 -9715 -1436 12071 -916
2949 -6433 4476 -15489 -18823 20413 -9474 -9920 -9108 -10135 16238 13684 -14439 20227 -5891 1931
-342 -15119 14675 -11504 13259 20342 20054 -13990 -1210 8468 -6532 8013 10141 12961 14194 -17081
-8502 2387 24352 -24426 -437 4785 3676 -10236 17558 105 8318 4529 4228 17954 -16040 15383
-9144 -16293 -21422 14772 -23162 4804 -17659 14250 -383 12263 8963 3154 -11506 9968 4314 10555
11710 -5482 -10260 -18669 -687 14708 14029 5876 2784 -14952 -19510 -4478 -6678 18374 -3907 -924
-3 -6993 -5877 14121 13591 -2618 14188 -9420 -11118 -11735 -3255 -19283 18716 -2934 892 20905
-9234 -3226 -11545 13437 -306 -1943 -18216 -19294 -10808 -19745 -9200 26219 -9787 -6617 -11055 13225
-4178 7666 4191 -18803 12155 -6850 4737 11156 10019 10691 5636 20099 -21920 8589 -17277 9534
-19391 -17723 -7094 3426 -16305 79 12288 -17540 13835 7768 -15344 1370 14062 4792 20188 -6769
11899 -5244 1491 22485 -29930 10245 15759 1962 -11819 -12340 2710 -31602 -19742 17974 8723 -23103
-12665 5552 22917 19376 -4035 -18307 1626 29693 19147 -12023 -10950 21663 9573 -3040 19312 19690
-18868 1338 25176 1174 16170 2510 13353 12097 6075 -18444 -9286 10300 2995 11613 1358 2889
8571 -7611 -5370 1910 -10173 8275 -2963 -5897 16849 -18208 9081 4030 -3658 -13575 -5535 -10046
4294 16432 21049 8645 11358 17889 12371 1405 -1471 -10787 -20010 -36027 -66 -13542 6140 7274
-18701 -7808 -7106 -7827 5690 -5992 -2607 11856 -14018 4075 -18957 3702 15483 16413 11363 -12452
-16885 -8549 -37379 -14918 13229 3328 -5395 21629 9545 -17197 -11403 28338 15545 19472 -11159 1590
-17718 -15174 9696 -20717 29758 10159 -4342 208 -10759 10098 13347 -11657 -987 -23748 -20497 -22387
-1702 -7073 11336 11308 14854 -22041 -15627 -13659 8540 -5396 2606 21702 10303 -3802 -18618 2871
6876 -18957 7191 -22073 16916 18458 1896 9134 5782 -7618 -17404 -3105 -15201 -29405 -12387 7267
1632 6990 17636 12487 -13985 -5882 -14919 29137 -23467 15996 -14089 -2555 17926 -13265 -8578 -7231
-7177 18282 10099 -10553 -15068 10296 -3698 -1188 26349 9028 -13447 9792 2126 -4625 11503 -12083
14444 -681 -36180 -14774 10867 -4463 7132 -37824 334 7447 5981 -35550 -7766 -3806 14224 3175
-19180 17891 -11090 22342 -20389 -6249 -320 22365 8523 -9444 18815 20507 -10284 17158 745 -802
10072 15937 6363 -21111 10910 -106 -9437 -7768 -2915 14997 18038 -9433 14049 6882 -9657 -15340
5430 -3865 9814 -1478 15370 -5703 19263 14435 -10051 -14239 13838 12773 -16223 -7679 6888 2139
1 16
26616 -21368 20809 -2639 3182 -26385 -12019 3565 73154 9128 -33946 17736 -9883 20334 -6994 42980
16 2
-29284 22266
-10066 31400
-85189 -4824
-54797 6855
67778 3660
33851 3800
-27119 3393
-53534 -36600
-56222 60012
1073 -5173
15458 -32361
-76585 14205
22234 25999
63604 883
-25959 32549
1703 44875
1 2
66792 6553
```
### 3. Программа визуализации
#### Код
`visualize_approximation.cpp`:
```
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <mutex>
#include <optional>
#include <iomanip>
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

struct Matrix { size_t r,c; std::vector<double> d; Matrix(size_t R=0,size_t C=0):r(R),c(C),d(R*C,0.0){} double& at(size_t R,size_t C){return d[R*c+C];} const double& at(size_t R,size_t C)const{return d[R*c+C];}};

class NeuroProcessor {
private:
    Matrix W1,B1,W2,B2,W3,B3;
    double x_min, x_max, y_min, y_max, mb_min, mb_max;
    int64_t scale_factor;
    
    static Matrix add_bias(const Matrix&a,const Matrix&b){Matrix r=a;for(size_t i=0;i<a.r;++i)for(size_t j=0;j<a.c;++j)r.at(i,j)+=b.at(0,j);return r;}
    static Matrix multiply(const Matrix&a,const Matrix&b){Matrix r(a.r,b.c);for(size_t i=0;i<a.r;++i)for(size_t j=0;j<b.c;++j)for(size_t k=0;k<a.c;++k)r.at(i,j)+=a.at(i,k)*b.at(k,j);return r;}
    static Matrix relu(const Matrix&m){Matrix r=m;for(auto&v:r.d)v=std::max(0.0,v);return r;}
    double normalize(double val, double min, double max) const { return 2.0 * (val - min) / (max - min) - 1.0; }
    double denormalize(double val, double min, double max) const { return (val + 1.0) / 2.0 * (max - min) + min; }
    
public:
    void load_weights(const std::string& filename) {
        std::ifstream infile(filename); 
        if(!infile) throw std::runtime_error("Нет файла: "+filename);
        
        infile >> scale_factor;
        infile >> x_min >> x_max >> y_min >> y_max >> mb_min >> mb_max;
        
        auto read_quantized = [&](Matrix& m) {
            infile >> m.r >> m.c;
            m.d.resize(m.r * m.c);
            for (auto& v : m.d) {
                int64_t quantized_val;
                infile >> quantized_val;
                v = static_cast<double>(quantized_val) / static_cast<double>(scale_factor);
            }
        };
        
        read_quantized(W1);
        read_quantized(B1);
        read_quantized(W2);
        read_quantized(B2);
        read_quantized(W3);
        read_quantized(B3);
        
        std::cout << "Целочисленные веса успешно загружены и деквантизованы." << std::endl;
        std::cout << "Коэффициент масштабирования: " << scale_factor << std::endl;
    }
    
    std::pair<double, double> process(const std::vector<std::pair<double, double>>& points) {
        Matrix input(1, W1.r);
        for(size_t i=0; i<points.size() && i*2+1<input.d.size(); ++i) {
            input.at(0, i*2) = normalize(points[i].first, x_min, x_max);
            input.at(0, i*2+1) = normalize(points[i].second, y_min, y_max);
        }
        Matrix Z1=add_bias(multiply(input,W1),B1), A1=relu(Z1);
        Matrix Z2=add_bias(multiply(A1,W2),B2), A2=relu(Z2);
        Matrix Z3=add_bias(multiply(A2,W3),B3);
        double m = denormalize(Z3.at(0,0), mb_min, mb_max);
        double b = denormalize(Z3.at(0,1), mb_min, mb_max);
        return {m, b};
    }
};

constexpr int SCREEN_WIDTH = 640; constexpr int SCREEN_HEIGHT = 480;
class VgaSimulator{SDL_Window*w=nullptr;SDL_Renderer*r=nullptr;SDL_Texture*t=nullptr;std::vector<uint32_t>fb;public:VgaSimulator(){if(SDL_Init(SDL_INIT_VIDEO)<0)throw std::runtime_error("SDL fail");w=SDL_CreateWindow("VGA Sim",100,100,SCREEN_WIDTH,SCREEN_HEIGHT,SDL_WINDOW_SHOWN);r=SDL_CreateRenderer(w,-1,SDL_RENDERER_ACCELERATED);t=SDL_CreateTexture(r,SDL_PIXELFORMAT_ARGB8888,SDL_TEXTUREACCESS_STREAMING,SCREEN_WIDTH,SCREEN_HEIGHT);fb.resize(SCREEN_WIDTH*SCREEN_HEIGHT,0);}~VgaSimulator(){SDL_DestroyTexture(t);SDL_DestroyRenderer(r);SDL_DestroyWindow(w);SDL_Quit();}void clear(uint32_t c){std::fill(fb.begin(),fb.end(),c);}void draw_pixel(int x,int y,uint32_t c){if(x>=0&&x<SCREEN_WIDTH&&y>=0&&y<SCREEN_HEIGHT)fb[y*SCREEN_WIDTH+x]=c;}void present(){SDL_UpdateTexture(t,NULL,fb.data(),SCREEN_WIDTH*sizeof(uint32_t));SDL_RenderClear(r);SDL_RenderCopy(r,t,NULL,NULL);SDL_RenderPresent(r);}bool process_events(){SDL_Event e;while(SDL_PollEvent(&e)!=0)if(e.type==SDL_QUIT)return false;return true;}};
void draw_point_on_vga(VgaSimulator&v,int cx,int cy,uint32_t c){for(int y=cy-2;y<=cy+2;++y)for(int x=cx-2;x<=cx+2;++x)v.draw_pixel(x,y,c);}
void draw_line_bresenham(VgaSimulator&v,int x0,int y0,int x1,int y1,uint32_t c){int dx=std::abs(x1-x0),sx=x0<x1?1:-1,dy=-std::abs(y1-y0),sy=y0<y1?1:-1,err=dx+dy,e2;for(;;){v.draw_pixel(x0,y0,c);if(x0==x1&&y0==y1)break;e2=2*err;if(e2>=dy){err+=dy;x0+=sx;}if(e2<=dx){err+=dx;y0+=sy;}}}
struct CoordMapper{double xmin,xmax,ymin,ymax;CoordMapper(const std::vector<std::pair<double,double>>&p,double m,double b){if(p.empty()){xmin=-10;xmax=10;ymin=-10;ymax=10;}else{xmin=p[0].first;xmax=p[0].first;ymin=p[0].second;ymax=p[0].second;for(const auto&pt:p){xmin=std::min(xmin,pt.first);xmax=std::max(xmax,pt.first);ymin=std::min(ymin,pt.second);ymax=std::max(ymax,pt.second);}}if(!p.empty()){double y1=m*xmin+b,y2=m*xmax+b;ymin=std::min({ymin,y1,y2});ymax=std::max({ymax,y1,y2});}double xp=(xmax-xmin)*0.1+1.0,yp=(ymax-ymin)*0.1+1.0;xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp;}std::pair<int,int>world_to_screen(double wx,double wy){double ww=xmax-xmin,wh=ymax-ymin;if(std::abs(ww)<1e-6)ww=1.0;if(std::abs(wh)<1e-6)wh=1.0;return{static_cast<int>((wx-xmin)/ww*SCREEN_WIDTH),static_cast<int>(SCREEN_HEIGHT-((wy-ymin)/wh*SCREEN_HEIGHT))};}};
std::mutex g_data_mutex; std::vector<std::pair<double,double>> g_user_points; std::optional<std::pair<double,double>>g_new_point;
void input_thread_func(){std::string line;while(true){std::cout<<"\nВведите точку (x,y) > ";if(!std::getline(std::cin,line)||line=="stop"||line=="exit")break;std::stringstream ss(line);double x,y;char c;if(!(ss>>x>>c>>y)||c!=','){std::cerr<<"Ошибка ввода. Формат 'x,y'."<<std::endl;continue;}std::lock_guard<std::mutex>lock(g_data_mutex);g_new_point={x,y};}}

int main(int argc, char* argv[]) {
    try {
        VgaSimulator vga; NeuroProcessor neuro_processor;
        neuro_processor.load_weights("network_weights.txt");
        std::thread input_thread(input_thread_func); input_thread.detach();
        double m=0.0, b=0.0; bool running=true;
        while(running){
            if(!vga.process_events())running=false;
            bool needs_recalc=false;
            {std::lock_guard<std::mutex>lock(g_data_mutex);if(g_new_point){g_user_points.push_back(*g_new_point);g_new_point.reset();needs_recalc=true;}}
            if(needs_recalc&&g_user_points.size()>=2){
                std::vector<std::pair<double,double>>points_for_nn;
                {std::lock_guard<std::mutex>lock(g_data_mutex);auto start=g_user_points.size()>3?g_user_points.end()-3:g_user_points.begin();points_for_nn=std::vector<std::pair<double,double>>(start,g_user_points.end());}
                std::tie(m,b)=neuro_processor.process(points_for_nn);
                std::cout << "Новые коэффициенты (вычислены сетью): m=" << std::fixed << std::setprecision(4) << m << ", b=" << b << std::endl;
            }
            std::vector<std::pair<double,double>>points_copy;{std::lock_guard<std::mutex>lock(g_data_mutex);points_copy=g_user_points;}
            CoordMapper mapper(points_copy,m,b);vga.clear(0xFF101010);
            auto origin=mapper.world_to_screen(0,0);
            draw_line_bresenham(vga,0,origin.second,SCREEN_WIDTH-1,origin.second,0xFF404040);
            draw_line_bresenham(vga,origin.first,0,origin.first,SCREEN_HEIGHT-1,0xFF404040);
            for(const auto&p:points_copy){auto[sx,sy]=mapper.world_to_screen(p.first,p.second);draw_point_on_vga(vga,sx,sy,0xFF00A0FF);}
            if(points_copy.size()>=2){auto p1=mapper.world_to_screen(mapper.xmin,m*mapper.xmin+b);auto p2=mapper.world_to_screen(mapper.xmax,m*mapper.xmax+b);draw_line_bresenham(vga,p1.first,p1.second,p2.first,p2.second,0xFFFF4040);}
            vga.present();SDL_Delay(16);
        }
    }catch(const std::exception& e){std::cerr<<"Критическая ошибка: "<<e.what()<<std::endl;return 1;}
    return 0;
}
```
#### Компиляция
```
g++ visualize_approximation.cpp -o visualize_approximation.exe -std=c++17 -lSDL2 -mconsole -finput-charset=CP1251 -fexec-charset=CP866
```
#### Запуск
Программа принимает на вход координаты точек и даёт на выход получившиеся коэффициенты линейной функции.

### 4. Пояснение

В данной работе реализована и обучена нейронная сеть для решения задачи регрессии: аппроксимации линейной функции по набору из трёх точек. Ниже приведено описание архитектуры сети и ключевых аспектов её реализации.

#### Архитектура сети

*   **Тип:** Полносвязная нейронная сеть прямого распространения (Feedforward Neural Network), также известная как многослойный перцептрон (Multi-Layer Perceptron, MLP).
*   **Структура:** Сеть состоит из входного слоя, двух скрытых слоев и одного выходного слоя. Такая архитектура является стандартной для решения нелинейных задач регрессии и обеспечивает достаточную сложность для аппроксимации целевой зависимости.

#### Количество слоёв

*   **Всего слоёв с нейронами:** 4 (1 входной, 2 скрытых, 1 выходной).
*   **Всего обучаемых слоёв:** 3. Обучаемым слоем считается преобразование, включающее матрицу весов (`W`) и вектор смещений (`B`). В коде это три пары: `(W1, B1)`, `(W2, B2)` и `(W3, B3)`.

#### Количество нейронов

*   **Входной слой:** **6 нейронов**. Размер входного слоя определяется константой `INPUT_SIZE`, которая вычисляется как `NUM_POINTS * 2`. Так как сеть принимает на вход координаты трёх точек (`NUM_POINTS = 3`), общее количество входов равно 3 точки × 2 координаты (x, y) = 6.
*   **Первый скрытый слой:** **32 нейрона** (константа `HIDDEN1_SIZE`).
*   **Второй скрытый слой:** **16 нейронов** (константа `HIDDEN2_SIZE`).
*   **Выходной слой:** **2 нейрона** (константа `OUTPUT_SIZE`). Сеть предсказывает два непрерывных значения: коэффициент наклона `m` и свободный член `b` для искомой прямой `y = m*x + b`.

#### Количество эпох

*   **Понятие "эпоха" не применяется в чистом виде.** Эпоха означает один полный проход по всему набору данных. В данной реализации набор данных не является фиксированным; вместо этого на каждом шаге обучения генерируется новый случайный батч данных.
*   Процесс обучения контролируется **количеством шагов (итераций)**.
*   **Количество шагов:** **80 000** (переменная `steps=80000`). На каждой итерации генерируется батч, выполняется прямое и обратное распространение ошибки, и веса обновляются.

#### Активация

*   **Скрытые слои:** **ReLU** (Rectified Linear Unit). Эта функция активации применяется к выходам обоих скрытых слоёв (`A1=apply_relu(Z1)` и `A2=apply_relu(Z2)`). ReLU выбрана за свою вычислительную простоту и эффективность в борьбе с проблемой затухающих градиентов.
*   **Выходной слой:** **Линейная (тождественная)**. К выходу последнего слоя (`Z3`) не применяется никакая нелинейная функция активации (`Y_pred=Z3`). Это стандартный подход для задач регрессии, поскольку выходные значения (`m` и `b`) могут принимать любые действительные значения, а не ограничиваться определённым диапазоном.

#### Функция ошибки

*   **Среднеквадратичная ошибка (Mean Squared Error, MSE)**. Выбор этой функции виден из нескольких частей кода:
    1.  **Вычисление ошибки:** `error.d[i]=Y_pred.d[i]-Y_batch.d[i]`.
    2.  **Градиент на выходе:** `Matrix dZ3=error;`. Производная MSE по выходу сети как раз равна разности `(предсказание - истинное значение)`.
    3.  **Вывод потерь:** `loss+=e*e; ... std::cout << "Потери: " << loss/N << std::endl;`. Здесь вычисляется и выводится среднее значение квадратов ошибок по батчу, что и является MSE.

#### Особенность реализации: Квантизация весов

Ключевой особенностью финальной реализации является **квантизация весов**. После завершения обучения с использованием чисел с плавающей запятой (`double`) все веса и смещения преобразуются в целочисленный формат перед сохранением в файл.

*   **Процесс:**
    1.  **Масштабирование:** Каждый весовой коэффициент умножается на большой коэффициент `SCALE_FACTOR = 100000`.
    2.  **Округление и преобразование:** Полученное значение округляется до ближайшего целого и сохраняется как 64-битное целое число (`int64_t`).
*   **Хранение:** В файл `network_weights.txt` сначала записывается сам `SCALE_FACTOR`, а затем — целочисленные матрицы весов.
*   **Применение:** Программа визуализации (`visualize_approximation.cpp`) считывает `SCALE_FACTOR` и целочисленные веса, а затем выполняет **обратное преобразование (деквантизацию)** — делит каждый весовой коэффициент на `SCALE_FACTOR` для восстановления его значения с плавающей запятой.

Это позволяет значительно уменьшить размер файла с весами и симулирует работу сети в условиях вычислений с фиксированной точкой, что актуально для встраиваемых систем и микроконтроллеров.
