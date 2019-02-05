#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <iostream>
#include <float.h>
#include <cmath>
#include <gsl/gsl_linalg.h>

using namespace cv;
using namespace std;

const double EPSILON = DBL_EPSILON;
//const double EPSILON = 0.00000001;
const int M = 3; // Максимальная степень полинома

				 // Метод Гауса для поиска коэффициентов полинома
std::vector<double> gauss(std::vector<std::vector<double>> a, std::vector<double> y, int n)
{
	double max;
	std::vector<double> x(n, 0);
	int k, index;
	k = 0;
	while (k < n)
	{
		// Поиск строки с максимальным a[i][k]
		max = abs(a[k][k]);
		index = k;
		for (int i = k + 1; i < n; i++)
		{
			if (abs(a[i][k]) > max)
			{
				max = abs(a[i][k]);
				index = i;
			}
		}
		// Перестановка строк
		if (max < EPSILON)
		{
			// нет нулевых диагональных элементов
			cout << "It`s impossible to take a solution because of o-column";
			cout << index << "of matrix A" << endl;
			return {};
		}
		for (int j = 0; j < n; j++)
		{
			double temp = a[k][j];
			a[k][j] = a[index][j];
			a[index][j] = temp;
		}
		double temp = y[k];
		y[k] = y[index];
		y[index] = temp;
		// Нормализация уравнений
		for (int i = k; i < n; i++)
		{
			double temp = a[i][k];
			if (abs(temp) < EPSILON) continue; // для нулевого коэффициента пропустить
			for (int j = 0; j < n; j++)
			{
				a[i][j] = a[i][j] / temp;
			}
			y[i] = y[i] / temp;
			if (i == k) continue; // уравнение не вычитать из самого себя
			for (int j = 0; j < n; j++)
			{
				a[i][j] = a[i][j] - a[k][j];
			}
			y[i] = y[i] - y[k];
		}
		k++;
	}
	// обратная подстановка
	for (k = n - 1; k >= 0; k--)
	{
		x[k] = y[k];
		for (int i = 0; i < k; i++)
		{
			y[i] = y[i] - a[i][k] * x[k];
		}
	}
	return x;
}

// Функция расчета разницы между координатам в различных каналах
double CalcChannelDif(KeyPoint B, KeyPoint G, KeyPoint R)
{
	//return pow((pow((G.pt.x - B.pt.x), 2) + pow((G.pt.y - B.pt.y), 2) + pow((G.pt.x - R.pt.x), 2) + pow((G.pt.y - R.pt.y), 2)), 1/2);
	return abs(G.pt.x - B.pt.x) + abs(G.pt.y - B.pt.y) + abs(G.pt.x - R.pt.x) + abs(G.pt.y - R.pt.y);
}

// Функция для поиска номера координат центра изображения в векторе
int GetImageCenterNumber(vector<vector<KeyPoint>> keypoints)
{
	int number = 0;
	double min = CalcChannelDif(keypoints[0][0], keypoints[1][0], keypoints[2][0]);
	double current;
	for (int i = 1; i < keypoints[0].size(); ++i)
	{
		current = CalcChannelDif(keypoints[0][i], keypoints[1][i], keypoints[2][i]);
		if (current < min)
		{
			number = i;
			min = current;
		}
	}
	return number;
}

// Функция суммирования, возведенных в степень значений (матрица a)
double sumPowA(vector<KeyPoint> pointsToSum, int xPow1, int yPow1, int xPow2, int yPow2)
{
	double result = 0;
	for (int i = 0; i < pointsToSum.size(); ++i)
	{
		result += pow(pointsToSum[i].pt.x, xPow1) * pow(pointsToSum[i].pt.y, yPow1) * pow(pointsToSum[i].pt.x, xPow2) * pow(pointsToSum[i].pt.y, yPow2);
	}
	return result;
}

// Функция суммирования, возведенных в степень значений (вектор y)
double sumPowY(vector<KeyPoint> pointsToSumG, vector<KeyPoint> pointsToSum, int xPow, int yPow, bool useX)
{
	double result = 0;
	for (int i = 0; i < pointsToSum.size(); ++i)
	{
		double tempSum;
		if (useX)
		{
			tempSum = pointsToSumG[i].pt.x;
		}
		else
		{
			tempSum = pointsToSumG[i].pt.y;
		}
		result += pow(pointsToSum[i].pt.x, xPow) * pow(pointsToSum[i].pt.y, yPow) * tempSum;
	}
	return result;
}

// Функция для расчета коэффициентов полинома для каналов
void calcPol(double * result, vector<KeyPoint> channelGreen, vector<KeyPoint> channelToCalc, int polNumber, bool calcX)
{
	// Инициализируем промежуточные массивы для решения системы уравнений
	//std::vector<std::vector<double>> a(polNumber, std::vector<double>(polNumber, 0));
	//std::vector<double> y(polNumber, 0);

	double *a = new double[polNumber * polNumber];
	double *y = new double[polNumber];

	// Вычисляем коэффициенты матрицы a
	int tempX1 = M;
	int tempY1 = 0;
	int tempX2 = M;
	int tempY2 = 0;
	for (int i = 0; i < polNumber; ++i)
	{
		printf("\n");
		tempX2 = M;
		tempY2 = 0;
		for (int j = 0; j < polNumber; ++j)
		{
			a[polNumber * i + j] = sumPowA(channelToCalc, tempX1, tempY1, tempX2, tempY2);
			printf("%f\t", a[polNumber * i + j]);
			if (tempX2 == 0)
			{
				tempX2 = tempY2 - 1;
				tempY2 = 0;
			}
			else
			{
				--tempX2;
				++tempY2;
			}
		}
		if (tempX1 == 0)
		{
			tempX1 = tempY1 - 1;
			tempY1 = 0;
		}
		else
		{
			--tempX1;
			++tempY1;
		}
	}
	printf("\n");
	// Вычисляем коэффициенты вектора y
	int tempX = M;
	int tempY = 0;
	for (int i = 0; i < polNumber; ++i)
	{
		y[i] = sumPowY(channelGreen, channelToCalc, tempX, tempY, calcX);
		printf("%f\t", y[i]);
		if (tempX == 0)
		{
			tempX = tempY - 1;
			tempY = 0;
		}
		else
		{
			--tempX;
			++tempY;
		}
	}
	printf("\n");
	// Вычисляем коэффициенты полинома
	//return gauss(a, y, polNumber);

	gsl_matrix_view m = gsl_matrix_view_array(a, polNumber, polNumber);

	gsl_vector_view b = gsl_vector_view_array(y, polNumber);

	gsl_vector *x = gsl_vector_alloc(polNumber);

	int s;

	gsl_permutation * p = gsl_permutation_alloc(polNumber);

	gsl_linalg_LU_decomp(&m.matrix, p, &s);

	gsl_linalg_LU_solve(&m.matrix, p, &b.vector, x);

	//result = x->data;
	
	for (int i = 0; i < polNumber; i++)
	{
		result[i] = x->data[i];
	}

	for (int i = 0; i < polNumber; i++)
	{
		printf("%f\t", result[i]);
	}
	printf("\n\n\n");
	gsl_permutation_free(p);
	gsl_vector_free(x);
}

// Функция для вычисления новой координаты субпикселя полиномом
double calcNewValuePol(double oldValueX, double oldValueY, double * polValues, int polNumber)
{
	double newValue = 0;
	int powX = M;
	int powY = 0;

	for (int i = 0; i < polNumber; ++i)
	{
		//printf("%f\t", polValues[i]);
		newValue += polValues[i] * pow(oldValueX, powX) * pow(oldValueY, powY);
		if (powX == 0)
		{
			powX = powY - 1;
			powY = 0;
		}
		else
		{
			--powX;
			++powY;
		}
	}
	//printf("\n");
	return newValue;
}

// Функция, заменяющая значения субпикселя по координате
void changeSubP(Mat image, int x, int y, int value, int channel)
{
	Vec3b tempVec;
	switch (channel)
	{
	case 0:
		tempVec[0] = value;
		tempVec[1] = image.at<Vec3b>(Point(y, x))[1];
		tempVec[2] = image.at<Vec3b>(Point(y, x))[2];
		break;

	case 1:
		tempVec[0] = image.at<Vec3b>(Point(y, x))[0];
		tempVec[1] = value;
		tempVec[2] = image.at<Vec3b>(Point(y, x))[2];
		break;

	case 2:
		tempVec[0] = image.at<Vec3b>(Point(y, x))[0];
		tempVec[1] = image.at<Vec3b>(Point(y, x))[1];
		tempVec[2] = value;
		break;
	}

	image.at<Vec3b>(Point(y, x)) = tempVec;
}

// Объявляем детектор блобов
SimpleBlobDetector::Params params;

// Задаем количество кружков на изображении
int circleNum = 9;

// Номер координат центра изображения
int centerNumber;

// Векторы хранения центров блобов
vector<vector<KeyPoint>> keypointsBGR;

// Векторы хранения центров пересчитанных блобов
vector<vector<KeyPoint>> keypointsNewBGR;

// Оригинальная картинка
Mat imageOrigin;

// Скорректированная картинка
Mat imageCorrected;

// Массив для каналов картинки (bgr)
Mat imageChannels[3];

// Массивы для хранения значений полиномов
std::vector<double> pxb, pyb, pxr, pyr;

int main(int argc, char** argv)
{
	// Уточняем размерность вектора центров окружностей
	keypointsBGR.resize(3, vector<KeyPoint>(circleNum));
	keypointsNewBGR.resize(3, vector<KeyPoint>(circleNum));

	// Здесь начинаются параметры для поиска блобов
	// Приходится определять их здесь, иначе ошибки
	params.minThreshold = 10;
	params.maxThreshold = 200;

	params.filterByArea = true;
	params.minArea = 1500;
	params.maxArea = 100000;

	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;
	// Здесь заканчиваются параметры для поиска блобов
	// Нужно потом попробовать уменьшить их количество

	// Создаем детектор блобов с параметрами
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Считываем картинку из папки
	// Сделать так, чтобы программа принимала параметр из параметра командной строки?
	imageOrigin = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// Инициализируем пустую картинку той же размерности, что и оригинальная
	Mat imageCorrected(imageOrigin.rows, imageOrigin.cols, imageOrigin.type(), Scalar(0, 0, 0));

	// Разбиваем картинку на цветовые каналы
	split(imageOrigin, imageChannels);

	// Определяем центры кругов на каждом цветовом канале
	for (int i = 0; i < 3; ++i)
	{
		detector->detect(imageChannels[i], keypointsBGR[i]);
	}

	// Получаем номер координат центра изображения в векторе
	centerNumber = GetImageCenterNumber(keypointsBGR);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < keypointsBGR[i].size(); ++j)
		{
			keypointsNewBGR[i][j].pt.x = keypointsBGR[i][j].pt.x - keypointsBGR[1][centerNumber].pt.x;
			keypointsNewBGR[i][j].pt.y = keypointsBGR[i][j].pt.y - keypointsBGR[1][centerNumber].pt.y;
		}
	}

	//// Выводим пересчитанные относительно нового центра координаты центров окружностей
	//for (int i = 0; i < 3; ++i)
	//{
	//	printf("\nChannel %d:\n", i + 1);
	//	for (int j = 0; j < keypointsNewBGR[i].size(); ++j)
	//	{
	//		printf("%f %f\n", keypointsNewBGR[i][j].pt.x, keypointsNewBGR[i][j].pt.y);
	//	}
	//}

	// Число коэффициентов в полиноме
	int pNum = ((M + 1) * (M + 2)) / 2;

	// Инициализируем вектора для искомых значений полиномов
	//std::vector<double> pxb(pNum, 0);
	//std::vector<double> pyb(pNum, 0);
	//std::vector<double> pxr(pNum, 0);
	//std::vector<double> pyr(pNum, 0);

	//double *pxb = new double[pNum];
	//double *pyb = new double[pNum];
	//double *pxr = new double[pNum];
	//double *pyr = new double[pNum];

	double *pxb = new double[pNum];
	double *pyb = new double[pNum];
	double *pxr = new double[pNum];
	double *pyr = new double[pNum];

	// Считаем коэфициенты полинома для каждого канала
	calcPol(pxb, keypointsNewBGR[1], keypointsBGR[0], pNum, true); // Синий X координаты
	calcPol(pyb, keypointsNewBGR[1], keypointsBGR[0], pNum, false); // Синиий Y координаты
	calcPol(pxr, keypointsNewBGR[1], keypointsBGR[2], pNum, true); // Красный X координаты
	calcPol(pyr, keypointsNewBGR[1], keypointsBGR[2], pNum, false); // Красный Y координаты

																	/*printf("\nBlue X coordinates\n");
																	for (int i = 0; i < pNum; ++i)
																	{
																	printf("%f\n", pxb[i]);
																	}

																	printf("\nBlue Y coordinates\n");
																	for (int i = 0; i < pNum; ++i)
																	{
																	printf("%f\n", pyb[i]);
																	}

																	printf("\nRed X coordinates\n");
																	for (int i = 0; i < pNum; ++i)
																	{
																	printf("%f\n", pxr[i]);
																	}

																	printf("\nRed Y coordinates\n");
																	for (int i = 0; i < pNum; ++i)
																	{
																	printf("%f\n", pyr[i]);
																	}*/

																	// Пересчитываем координаты субпикселей в новое изображение
	double oldTempBX, oldTempBY;
	int newCoordBX, newCoordBY;
	int rows = imageOrigin.rows;
	int cols = imageOrigin.cols;
	Vec3b newValue;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			// Синий канал
			oldTempBX = j - keypointsBGR[1][centerNumber].pt.x;
			oldTempBY = i - keypointsBGR[1][centerNumber].pt.y;
			newCoordBX = int(calcNewValuePol(oldTempBX, oldTempBY, pxb, pNum) + keypointsBGR[1][centerNumber].pt.x);
			newCoordBY = int(calcNewValuePol(oldTempBX, oldTempBY, pyb, pNum) + keypointsBGR[1][centerNumber].pt.y);

			//printf("(%d %d) -> (%d %d)\n", int(oldTempBX), int(oldTempBY), int(newCoordBX), int(newCoordBY));

			if (newCoordBX >= 0 && newCoordBX < cols && newCoordBY >= 0 && newCoordBY < rows)
			{
				changeSubP(imageCorrected, newCoordBX, newCoordBY, imageOrigin.at<Vec3b>(Point(i, j))[0], 0);
			}

			// Зеленый канал
			changeSubP(imageCorrected, j, i, imageOrigin.at<Vec3b>(Point(i, j))[1], 1);

			// Красный канал
			oldTempBX = keypointsBGR[1][centerNumber].pt.x - j;
			oldTempBY = keypointsBGR[1][centerNumber].pt.y - i;
			newCoordBX = int(calcNewValuePol(oldTempBX, oldTempBY, pxr, pNum) + keypointsBGR[1][centerNumber].pt.x);
			newCoordBY = int(calcNewValuePol(oldTempBX, oldTempBY, pyr, pNum) + keypointsBGR[1][centerNumber].pt.y);
			if (newCoordBX >= 0 && newCoordBX < cols && newCoordBY >= 0 && newCoordBY < rows)
			{
				changeSubP(imageCorrected, newCoordBX, newCoordBY, imageOrigin.at<Vec3b>(Point(i, j))[2], 2);
			}
		}
	}

	imshow("Corrected", imageCorrected);

	waitKey(0);

}
