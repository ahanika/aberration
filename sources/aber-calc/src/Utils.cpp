#include "opencv2/opencv.hpp"
#include <vector>
#include <gsl/gsl_linalg.h>

#include "Utils.h"

// Функция расчета разницы между координатам в различных каналах
double calc_channel_diff(
    cv::KeyPoint B,
    cv::KeyPoint G,
    cv::KeyPoint R)
{
    return sqrt(
        pow((G.pt.x - B.pt.x), 2)
        + pow((G.pt.y - B.pt.y), 2)
        + pow((G.pt.x - R.pt.x), 2)
        + pow((G.pt.y - R.pt.y), 2)
    );
}

// Функция для поиска номера координат центра изображения в векторе
int get_image_center_number(std::vector<std::vector<cv::KeyPoint>> keypoints)
{
    int number = 0;
    double min = calc_channel_diff(
        keypoints[0][0],
        keypoints[1][0],
        keypoints[2][0]
    );

    double current;
    for (int i = 1; i < keypoints[0].size(); ++i)
    {
        current = calc_channel_diff(
            keypoints[0][i],
            keypoints[1][i],
            keypoints[2][i]
        );

        if (current < min)
        {
            number = i;
            min = current;
        }
    }
    return number;
}

// Функция суммирования, возведенных в степень значений (матрица a)
double sum_pow_a(
    std::vector<cv::KeyPoint> points_to_sum,
    int x_pow_1,
    int y_pow_1,
    int x_pow_2,
    int y_pow_2)
{
    double result = 0;
    for (int i = 0; i < points_to_sum.size(); ++i)
    {
        result += pow(points_to_sum[i].pt.x, x_pow_1)
            * pow(points_to_sum[i].pt.y, y_pow_1)
            * pow(points_to_sum[i].pt.x, x_pow_2)
            * pow(points_to_sum[i].pt.y, y_pow_2);
    }
    return result;
}

// Функция суммирования, возведенных в степень значений (вектор y)
double sum_pow_y(
    std::vector<cv::KeyPoint> points_to_sum_g,
    std::vector<cv::KeyPoint> points_to_sum,
    int x_pow,
    int y_pow,
    bool use_x)
{
    double result = 0;
    for (int i = 0; i < points_to_sum.size(); ++i)
    {
        double temp_sum;
        if (use_x)
        {
            temp_sum = points_to_sum_g[i].pt.x;
        }
        else
        {
            temp_sum = points_to_sum_g[i].pt.y;
        }

        result += pow(points_to_sum[i].pt.x, x_pow)
            * pow(points_to_sum[i].pt.y, y_pow)
            * temp_sum;
    }
    return result;
}

// Функция для решения СЛУ
void solve_SLE(
    double* result,
    double* a_matr,
    double* y_vec,
    int size,
    int refine)
{
    // Решаем СЛУ
    gsl_matrix_view m = gsl_matrix_view_array(a_matr, size, size);

    gsl_vector_view b = gsl_vector_view_array(y_vec, size);

    gsl_vector* x = gsl_vector_alloc(size);

    int s;

    gsl_permutation* p = gsl_permutation_alloc(size);

    gsl_matrix_view LU = gsl_matrix_view_array(a_matr, size, size);

    gsl_vector* work = gsl_vector_alloc(size);

    gsl_linalg_LU_decomp(&LU.matrix, p, &s);

    gsl_linalg_LU_solve(&LU.matrix, p, &b.vector, x);

    // Улучшаем решением повторными вызовами функции gsl_linalg_LU_refine
    for (int i = 0; i < refine; i++)
    {
        gsl_linalg_LU_refine(&m.matrix, &LU.matrix, p, &b.vector, x, work);
    }

    for (int i = 0; i < size; i++)
    {
        result[i] = x->data[i];
    }

    gsl_permutation_free(p);
    gsl_vector_free(x);
    gsl_vector_free(work);
}

// Функция для расчета коэффициентов полинома для каналов
void calc_pol(
    double* result,
    std::vector<cv::KeyPoint> channel_green,
    std::vector<cv::KeyPoint> channel_to_calc,
    int pol_number,
    bool calc_x,
    int max_power,
    int refine_sol)
{
    // Инициализируем промежуточные массивы для решения системы уравнений
    double* a = new double[pol_number * pol_number];
    double* y = new double[pol_number];

    // Вычисляем коэффициенты матрицы a
    int temp_x1 = max_power;
    int temp_y1 = 0;
    int temp_x2 = max_power;
    int temp_y2 = 0;
    for (int i = 0; i < pol_number; ++i)
    {
        temp_x2 = max_power;
        temp_y2 = 0;
        for (int j = 0; j < pol_number; ++j)
        {
            a[pol_number * i + j] = sum_pow_a(
                channel_to_calc,
                temp_x1,
                temp_y1,
                temp_x2,
                temp_y2
            );

            if (temp_x2 == 0)
            {
                temp_x2 = temp_y2 - 1;
                temp_y2 = 0;
            }
            else
            {
                --temp_x2;
                ++temp_y2;
            }
        }
        if (temp_x1 == 0)
        {
            temp_x1 = temp_y1 - 1;
            temp_y1 = 0;
        }
        else
        {
            --temp_x1;
            ++temp_y1;
        }
    }
    
    // Вычисляем коэффициенты вектора y
    int temp_x = max_power;
    int temp_y = 0;
    for (int i = 0; i < pol_number; ++i)
    {
        y[i] = sum_pow_y(
            channel_green,
            channel_to_calc,
            temp_x,
            temp_y,
            calc_x
        );
        
        if (temp_x == 0)
        {
            temp_x = temp_y - 1;
            temp_y = 0;
        }
        else
        {
            --temp_x;
            ++temp_y;
        }
    }

    // Решаем СЛУ
    solve_SLE(
        result,
        a,
        y,
        pol_number,
        refine_sol
    );

    delete(y);
    delete(a);
}

// Функция для вычисления новой координаты субпикселя полиномом
int calc_new_value_pol(
    double old_value_x,
    double old_value_y,
    double* pol_values,
    int pol_number,
    int max_power,
    float center)
{
    double new_value = 0;
    int pow_x = max_power;
    int pow_y = 0;

    for (int i = 0; i < pol_number; ++i)
    {
        new_value += pol_values[i]
            * pow(old_value_x, pow_x)
            * pow(old_value_y, pow_y);

        if (pow_x == 0)
        {
            pow_x = pow_y - 1;
            pow_y = 0;
        }
        else
        {
            --pow_x;
            ++pow_y;
        }
    }

    return (int)round(new_value + center);
}

// Функция, заменяющая значения субпикселя по координате
void change_subpixel(
    cv::Mat image,
    int x,
    int y,
    int value,
    int channel)
{
    cv::Vec3b temp_vec;
    switch (channel)
    {
    case 0:
        temp_vec[0] = value;
        temp_vec[1] = image.at<cv::Vec3b>(cv::Point(y, x))[1];
        temp_vec[2] = image.at<cv::Vec3b>(cv::Point(y, x))[2];
        break;

    case 1:
        temp_vec[0] = image.at<cv::Vec3b>(cv::Point(y, x))[0];
        temp_vec[1] = value;
        temp_vec[2] = image.at<cv::Vec3b>(cv::Point(y, x))[2];
        break;

    case 2:
        temp_vec[0] = image.at<cv::Vec3b>(cv::Point(y, x))[0];
        temp_vec[1] = image.at<cv::Vec3b>(cv::Point(y, x))[1];
        temp_vec[2] = value;
        break;
    }

    image.at<cv::Vec3b>(cv::Point(y, x)) = temp_vec;
}

// Функция для нахождения ближайшей окружности из вектора
int get_closest(
    cv::KeyPoint point,
    std::vector<cv::KeyPoint> channel,
    int number)
{
    int result = 0;

    float min = sqrt(
        pow((point.pt.x - channel[0].pt.x), 2)
        + pow((point.pt.y - channel[0].pt.y), 2)
    );

    float current = 0;

    for (int i = 1; i < channel.size(); i++)
    {
        current = sqrt(
            pow((point.pt.x - channel[i].pt.x), 2)
            + pow((point.pt.y - channel[i].pt.y), 2)
        );

        if (current < min)
        {
            min = current;
            result = i;
        }
    }

    return result;
}

// Выравниваем остальные каналу по зеленому
void align_keypoints_by_green(std::vector<std::vector<cv::KeyPoint>> &keypoints)
{
    float temp = 0;
    int closest = 0;

    // Допускаем, что зеленый канал всегда правильный, т.е.
    // отсортирован справа-налево снизу-вверх.
    // Здесь можно добавить сортировку зеленого канала
    for (int i = 0; i < keypoints[1].size(); i++)
    {
        // Синий канал
        closest = get_closest(keypoints[1][i], keypoints[0], i);

        if (closest != i)
        {
            temp = keypoints[0][i].pt.x;
            keypoints[0][i].pt.x = keypoints[0][closest].pt.x;
            keypoints[0][closest].pt.x = temp;

            temp = keypoints[0][i].pt.y;
            keypoints[0][i].pt.y = keypoints[0][closest].pt.y;
            keypoints[0][closest].pt.y = temp;
        }

        // Красный канал
        closest = get_closest(keypoints[1][i], keypoints[2], i);

        if (closest != i)
        {
            temp = keypoints[2][i].pt.x;
            keypoints[2][i].pt.x = keypoints[2][closest].pt.x;
            keypoints[2][closest].pt.x = temp;

            temp = keypoints[2][i].pt.y;
            keypoints[2][i].pt.y = keypoints[2][closest].pt.y;
            keypoints[2][closest].pt.y = temp;
        }
    }
}

// Функция для удаления аберрации с изображения
cv::Mat delete_abberation(
    cv::Mat original,
    double* pxb,
    double* pyb,
    double* pxr,
    double* pyr,
    int size,
    int max_power,
    cv::KeyPoint center)
{
    int rows = original.rows;
    int cols = original.cols;

    // Инициализируем пустую картинку той же размерности, что и оригинальная
    cv::Mat corrected(
        rows,
        cols,
        original.type(),
        cv::Scalar(0, 0, 0)
    );

    int new_x = 0;
    int new_y = 0;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // Синий канал
            new_x = calc_new_value_pol(
                j - center.pt.x,
                i - center.pt.y,
                pxb,
                size,
                max_power,
                center.pt.x
            );

            new_y = calc_new_value_pol(
                j - center.pt.x,
                i - center.pt.y,
                pyb,
                size,
                max_power,
                center.pt.y
            );

            if (new_x >= 0 && new_x < cols && new_y >= 0 && new_y < rows)
            {
                change_subpixel(
                    corrected,
                    new_x,
                    new_y,
                    original.at<cv::Vec3b>(cv::Point(i, j))[0],
                    0
                );
            }

            // Зеленый канал
            change_subpixel(
                corrected,
                j,
                i,
                original.at<cv::Vec3b>(cv::Point(i, j))[1],
                1
            );

            // Красный канал
            new_x = calc_new_value_pol(
                j - center.pt.x,
                i - center.pt.y,
                pxr,
                size,
                max_power,
                center.pt.x
            );

            new_y = calc_new_value_pol(
                j - center.pt.x,
                i - center.pt.y,
                pyr,
                size,
                max_power,
                center.pt.y
            );

            if (new_x >= 0 && new_x < cols && new_y >= 0 && new_y < rows)
            {
                change_subpixel(
                    corrected,
                    new_x,
                    new_y,
                    original.at<cv::Vec3b>(cv::Point(i, j))[2],
                    2
                );
            }
        }
    }

    return corrected;
}

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
    float center_y)
{
    std::ofstream out;
    out.open(file);

    out << degree << std::endl;

    for (int i = 0; i < arr_size; i++)
    {
        out << pxb[i] << "\t";
    }

    out << std::endl;

    for (int i = 0; i < arr_size; i++)
    {
        out << pyb[i] << "\t";
    }

    out << std::endl;

    for (int i = 0; i < arr_size; i++)
    {
        out << pxr[i] << "\t";
    }

    out << std::endl;

    for (int i = 0; i < arr_size; i++)
    {
        out << pyr[i] << "\t";
    }

    out << std::endl << center_x << "\t" << center_y << std::endl;

    out.close();
}

// Функция чтения данных из файла параметров
int read_config(
    char* path,
    std::string& image_path,
    int& max_power,
    int& refine,
    int& circles,
    cv::SimpleBlobDetector::Params& params,
    std::string& result_path,
    bool& preview)
{
    std::ifstream file;
    file.open(path, std::ios::in);

    if (!file.is_open())
    {
        return -1;
    }

    // Путь к изображению
    file >> image_path;

    // Максимальная степень полинома
    file >> max_power;

    // Количество вызовов функции для улучшения решения СЛУ
    file >> refine;

    // Количество кругов на изображении
    file >> circles;

    // Настраиваем параметры для blobDetector
    file >> params.minThreshold;
    file >> params.maxThreshold;

    params.filterByArea = true;
    file >> params.minArea;
    file >> params.maxArea;

    params.filterByCircularity = true;
    file >> params.minCircularity;

    params.filterByConvexity = true;
    file >> params.minConvexity;

    params.filterByInertia = true;
    file >> params.minInertiaRatio;

    // Путь к файлу, в который сохранятся результаты работы программы
    file >> result_path;

    // Выводим или не выводим превью изображения
    file >> preview;

    return 0;
}
