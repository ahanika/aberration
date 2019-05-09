#include "opencv2/opencv.hpp"
#include <vector>

#include "Utils.h"

// Функция для записи строки файла в вектор
std::vector<double> read_vector_from_file(
    std::ifstream& file,
    const int pol_number)
{
    std::vector<double> result;
    double temp = 0;

    for (int i = 0; i < pol_number; i++)
    {
        file >> temp;
        result.push_back(temp);
    }

    return result;
}

// Функция для вычисления новой координаты субпикселя полиномом
int calc_new_value_pol(
    const double old_x,
    const double old_y,
    const std::vector<double> polynom,
    const int max_power,
    const double center)
{
    double result = 0;

    int pow_x = max_power;
    int pow_y = 0;

    for (int i = 0; i < polynom.size(); i++)
    {
        result += polynom[i] * pow(old_x, pow_x) * pow(old_y, pow_y);

        // Тут пересчитываем степени для каждого очередного коэффициента полинома
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

    return (int)round(result + center);
}

// Функция для замены субпикселя в изображении
void insert_subp(cv::Mat image, const int x, const int y, const int value, const int channel)
{
    cv::Vec3b temp_vec;
    switch (channel)
    {
    case 0:
        // Синий канал
        temp_vec[0] = value;
        temp_vec[1] = image.at<cv::Vec3b>(cv::Point(y, x))[1];
        temp_vec[2] = image.at<cv::Vec3b>(cv::Point(y, x))[2];
        break;

    case 1:
        // Зеленый канал
        temp_vec[0] = image.at<cv::Vec3b>(cv::Point(y, x))[0];
        temp_vec[1] = value;
        temp_vec[2] = image.at<cv::Vec3b>(cv::Point(y, x))[2];
        break;

    case 2:
        // Красный канал
        temp_vec[0] = image.at<cv::Vec3b>(cv::Point(y, x))[0];
        temp_vec[1] = image.at<cv::Vec3b>(cv::Point(y, x))[1];
        temp_vec[2] = value;
        break;
    }

    image.at<cv::Vec3b>(cv::Point(y, x)) = temp_vec;
}

// Функция, применяющая полином к каждому субпикселю
cv::Mat delete_abberation(
    const cv::Mat original,
    const std::vector<double> xb,
    const std::vector<double> yb,
    const std::vector<double> xr,
    const std::vector<double> yr,
    const double cent_x,
    const double cent_y,
    const int max_power)
{
    int rows = original.rows;
    int cols = original.cols;

    cv::Mat result(rows, cols, original.type(), cv::Scalar(0, 0, 0));

    int new_x = 0;
    int new_y = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Синий канал
            new_x = calc_new_value_pol(j - cent_x, i - cent_y, xb, max_power, cent_x);
            new_y = calc_new_value_pol(j - cent_x, i - cent_y, yb, max_power, cent_y);

            // Если субпиксель в результате вылазит за пределы изображения, то не отрисовываем его
            if (new_x >= 0 && new_x < cols && new_y >= 0 && new_y < rows)
            {
                insert_subp(result, new_x, new_y, original.at<cv::Vec3b>(cv::Point(i, j))[0], 0);
            }

            // Зеленый канал
            insert_subp(result, j, i, original.at<cv::Vec3b>(cv::Point(i, j))[1], 1);

            // Красный канал
            new_x = calc_new_value_pol(j - cent_x, i - cent_y, xr, max_power, cent_x);
            new_y = calc_new_value_pol(j - cent_x, i - cent_y, yr, max_power, cent_y);

            // Если субпиксель в результате вылазит за пределы изображения, то не отрисовываем его
            if (new_x >= 0 && new_x < cols && new_y >= 0 && new_y < rows)
            {
                insert_subp(result, new_x, new_y, original.at<cv::Vec3b>(cv::Point(i, j))[2], 2);
            }
        }
    }

    return result;
}
