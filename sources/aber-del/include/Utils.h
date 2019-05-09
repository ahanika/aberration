#ifndef PROJECT_INCLUDE_UTILS_H_
#define PROJECT_INCLUDE_UTILS_H_

constexpr auto DEFAULT_NAME = "result.jpg";
constexpr auto FLAG_NO_PREVIEW = "--no-preview";

// Функция для записи строки файла в вектор
std::vector<double> read_vector_from_file(std::ifstream& file, const int pol_number);

// Функция для вычисления новой координаты субпикселя полиномом
int calc_new_value_pol(
	const double old_x,
	const double old_y,
	const std::vector<double> polynom,
	const int max_power,
	const double center);

// Функция для замены субпикселя в изображении
void insert_subp(cv::Mat image, const int x, const int y, const int value, const int channel);

// Функция, применяющая полином к каждому субпикселю
cv::Mat delete_abberation(
	const cv::Mat original,
	const std::vector<double> xb,
	const std::vector<double> yb,
	const std::vector<double> xr,
	const std::vector<double> yr,
	const double cent_x,
	const double cent_y,
	const int max_power);

#endif  // PROJECT_INCLUDE_UTILS_H_
