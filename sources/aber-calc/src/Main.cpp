#include "opencv2/opencv.hpp"
#include <vector>

#include "Utils.h"

int main(int argc, char** argv)
{
    // Выводим сообщение об ошибке, если в параметрах нет пути к файлу параметров
    if (argc < 2)
    {
        std::cerr << "Missing path to parameter file!" << std::endl;
        return -1;
    }

    // Путь к калибровочному изображению
    std::string image_original = "";

    // Максимальная степень полинома
    int M = 0;

    // Количество вызовов функции для улучшения решения СЛУ
    int refine = 0;

    // Количество кругов на калибровочном изображении
    int circle_num = 0;

    // Объявляем детектор блобов
    cv::SimpleBlobDetector::Params params;

    // Путь к файлу, в которой запишется результат работы программы
    std::string file_result = "";

    // Выводим превью изображения или нет
    bool preview = false;

    if (read_config(
        argv[1],
        image_original,
        M,
        refine,
        circle_num,
        params,
        file_result,
        preview
    ))
    {
        std::cerr << "Can't open " << argv[1] << std::endl;
        return -1;
    }

    // Векторы хранения центров блобов
    std::vector<std::vector<cv::KeyPoint>> keypoints_BGR(
        3,
        std::vector<cv::KeyPoint>(circle_num)
    );

    // Векторы хранения центров пересчитанных блобов
    std::vector<std::vector<cv::KeyPoint>> keypoints_new_BGR(
        3,
        std::vector<cv::KeyPoint>(circle_num)
    );

    // Создаем детектор блобов с параметрами
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // Считываем картинку
    cv::Mat image_origin = cv::imread(image_original, CV_LOAD_IMAGE_COLOR);

    if (image_origin.empty())
    {
        std::cerr << "Can't open " << image_original << std::endl;
        return -1;
    }

    // Массив для каналов картинки (bgr)
    cv::Mat image_channels[3];

    // Разбиваем картинку на цветовые каналы
    split(image_origin, image_channels);
    
    // Определяем центры кругов на каждом цветовом канале
    for (int i = 0; i < 3; i++)
    {
        detector->detect(image_channels[i], keypoints_BGR[i]);
    }

    // Если мы не нашли все круги, то выводим сообщение об ошибке
    if (!(circle_num == keypoints_BGR[0].size()
        && circle_num == keypoints_BGR[1].size()
        && circle_num == keypoints_BGR[2].size()))
    {
        std::cerr << "Can't detect all circles!" << std::endl;
        std::cerr << "Change some parameters and try again!" << std::endl;
        return -1;
    }

    // Здесь переставляем круги в нужном порядке:
    // справа-налево, снизу-вверх
    align_keypoints_by_green(keypoints_BGR);

    // Получаем номер координат центра изображения в векторе
     int center_number = get_image_center_number(keypoints_BGR);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < keypoints_BGR[i].size(); ++j)
        {
            keypoints_new_BGR[i][j].pt.x = keypoints_BGR[i][j].pt.x
                - keypoints_BGR[1][center_number].pt.x;

            keypoints_new_BGR[i][j].pt.y = keypoints_BGR[i][j].pt.y
                - keypoints_BGR[1][center_number].pt.y;
        }
    }

    // Число коэффициентов в полиноме
    int pol_num = ((M + 1) * (M + 2)) / 2;

    double *pxb = new double[pol_num];
    double *pyb = new double[pol_num];
    double *pxr = new double[pol_num];
    double *pyr = new double[pol_num];

    // Считаем коэфициенты полинома для каждого канала
    // Синий X координаты
    calc_pol(
        pxb,
        keypoints_new_BGR[1],
        keypoints_new_BGR[0],
        pol_num,
        true,
        M,
        refine
    );

    // Синиий Y координаты
    calc_pol(
        pyb,
        keypoints_new_BGR[1],
        keypoints_new_BGR[0],
        pol_num,
        false,
        M,
        refine
    );

    // Красный X координаты
    calc_pol(
        pxr,
        keypoints_new_BGR[1],
        keypoints_new_BGR[2],
        pol_num,
        true,
        M,
        refine
    );

    // Красный Y координаты
    calc_pol(
        pyr,
        keypoints_new_BGR[1],
        keypoints_new_BGR[2],
        pol_num,
        false,
        M,
        refine
    );

    // Удаляем аберрацию с картинки
    cv::Mat image_corrected = delete_abberation(
        image_origin,
        pxb,
        pyb,
        pxr,
        pyr,
        pol_num,
        M,
        keypoints_BGR[1][center_number]
    );

    // Выводим изображение в зависимости от параметров
    if (preview)
    {
        imshow("Corrected", image_corrected);
        cv::waitKey(0);
    }

    // Пишем результаты работы программы в файл
    write_to_file(
        file_result,
        M,
        pxb,
        pyb,
        pxr,
        pyr,
        pol_num,
        keypoints_BGR[1][center_number].pt.x,
        keypoints_BGR[1][center_number].pt.y
    );

    delete(pxb);
    delete(pyb);
    delete(pxr);
    delete(pyr);

    return 0;
}
