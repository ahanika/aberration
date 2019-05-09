#ifndef PROJECT_INCLUDE_UTILS_H_
#define PROJECT_INCLUDE_UTILS_H_

// Функция расчета разницы между координатам в различных каналах
double calc_channel_diff(
    cv::KeyPoint B,
    cv::KeyPoint G,
    cv::KeyPoint R
);

// Функция для поиска номера координат центра изображения в векторе
int get_image_center_number(std::vector<std::vector<cv::KeyPoint>> keypoints);

// Функция суммирования, возведенных в степень значений (матрица a)
double sum_pow_a(
    std::vector<cv::KeyPoint> points_to_sum,
    int x_pow_1,
    int y_pow_1,
    int x_pow_2,
    int y_pow_2
);

// Функция суммирования, возведенных в степень значений (вектор y)
double sum_pow_y(
    std::vector<cv::KeyPoint> points_to_sum_g,
    std::vector<cv::KeyPoint> points_to_sum,
    int x_pow,
    int y_pow,
    bool use_x
);

// Функция для решения СЛУ
void solve_SLE(
    double* result,
    double* a_matr,
    double* y_vec,
    int size,
    int refine
);

// Функция для расчета коэффициентов полинома для каналов
void calc_pol(
    double* result,
    std::vector<cv::KeyPoint> channel_green,
    std::vector<cv::KeyPoint> channel_to_calc,
    int pol_number,
    bool calc_x,
    int max_power,
    int refine_sol
);

// Функция для вычисления новой координаты субпикселя полиномом
int calc_new_value_pol(
    double old_value_x,
    double old_value_y,
    double* pol_values,
    int pol_number,
    int max_power,
    float center
);

// Функция, заменяющая значения субпикселя по координате
void change_subpixel(
    cv::Mat image,
    int x,
    int y,
    int value,
    int channel
);

// Функция для нахождения ближайшей окружности из вектора
int get_closest(
    cv::KeyPoint point,
    std::vector<cv::KeyPoint> channel,
    int number
);

// Выравниваем остальные каналу по зеленому
void align_keypoints_by_green(std::vector<std::vector<cv::KeyPoint>> &keypoints);

// Функция для удаления аберрации с изображения
cv::Mat delete_abberation(
    cv::Mat original,
    double* pxb,
    double* pyb,
    double* pxr,
    double* pyr,
    int size,
    int max_power,
    cv::KeyPoint center
);

// Пишем коэффициенты полинома в файл для далнейшего переиспользования
void write_to_file(
    std::string file,
    int degree,
    double* pxb,
    double* pyb,
    double* pxr,
    double* pyr,
    int arr_size,
    float center_x,
    float center_y
);

// Функция чтения данных из файла параметров
int read_config(
    char* path,
    std::string& image_path,
    int& max_power,
    int& refine,
    int& circles,
    cv::SimpleBlobDetector::Params& params,
    std::string& result_path,
    bool& preview
);

#endif  // PROJECT_INCLUDE_UTILS_H_
