#include "opencv2/opencv.hpp"
#include <vector>

#include "Utils.h"

int main(int argc, char** argv)
{
    // Если в аргументах нет имени файла или имени изображения,
    // то выводим сообщение об ошибке и прерываем программу
    if (argc < 3)
    {
        std::cerr << "Missing call parameters!" << std::endl;
        return -1;
    }

    // Получаем названия файла с коэффициентами полинома и изображения для обработки
    const std::string filename = argv[1];
    const std::string imagename = argv[2];

    // Имя результата по умолчанию или же считываем из параметров запуска
    std::string result_name = DEFAULT_NAME;
    if (argc >= 4)
    {
        result_name = argv[3];
    }

    // Если 5 аргументом идет FLAG_NO_PREVIEW, то просто сохраняем файл
    bool preview = true;
    if (argc >= 5 && strcmp(FLAG_NO_PREVIEW, argv[4]) == 0)
    {
        preview = false;
    }

    // Открываем файл на чтение
    std::ifstream file(filename, std::ios::in);

    if (!file.is_open())
    {
        std::cerr << "Can't open " << filename << std::endl;
        return -1;
    }

    // Максимальная степень полинома
    int M = 0;

    file >> M;

    // Количество элементов в полиноме
    int pol_number = ((M + 1) * (M + 2)) / 2;

    // Вектора для хранения весов
    std::vector<double> pxb;
    std::vector<double> pxr;
    std::vector<double> pyb;
    std::vector<double> pyr;

    pxb = read_vector_from_file(file, pol_number);
    pyb = read_vector_from_file(file, pol_number);
    pxr = read_vector_from_file(file, pol_number);
    pyr = read_vector_from_file(file, pol_number);

    // Координаты центра аберрации
    double center_x = 0;
    double center_y = 0;

    file >> center_x;
    file >> center_y;

    file.close();

    cv::Mat image_origin = cv::imread(imagename, CV_LOAD_IMAGE_COLOR);

    if (image_origin.empty())
    {
        std::cerr << "Can't open " << imagename << std::endl;
        return -1;
    }

    // Преобразуем изображение
    cv::Mat image_result = delete_abberation(
        image_origin,
        pxb,
        pyb,
        pxr,
        pyr,
        center_x,
        center_y,
        M
    );

    if (preview)
    {
        imshow("Result", image_result);
        cv::waitKey(0);
    }

    imwrite(result_name, image_result);

    return 0;
}
