# Программа aber-calc

## Начало работы

Для корректной работы программы необходимо калибровочное изображение - черные круги на белом фоне и файл конфигурации со следующей структурой:

* путь к калибровочному изображению

* максимальная степень полинома

* количество вызовов функции REFINE для улучшения решения системы линейных уравнений

* количество кругов на изображении

* `minThreshold` параметр у SimpleBlobDetector. Необходим для детектирования кругов на калибровочном изображении. Подробно можно почитать [здесь](https://docs.opencv.org/3.4.1/d0/d7a/classcv_1_1SimpleBlobDetector.html)

* `maxThreshold` параметр у SimpleBlobDetector. Аналогично `minThreshold`

* `maxThreshold` параметр у SimpleBlobDetector. Аналогично `minThreshold`

* `minArea` параметр у SimpleBlobDetector. Аналогично `minThreshold`

* `maxArea` параметр у SimpleBlobDetector. Аналогично `minThreshold`

* `minCircularity` параметр у SimpleBlobDetector. Аналогично `minThreshold`

* `minConvexity` параметр у SimpleBlobDetector. Аналогично `minThreshold`

* `minInertiaRatio` параметр у SimpleBlobDetector. Аналогично `minThreshold`

* файл, в который будет записана информация, необходимая для работы aber-del

* 1 - вывод предварительного просмотра изображения без аберации, 0 - без предварительного просмотра

Пример файла конфигурации в `config.txt` в текущей директории.

## Запуск программы

Программа принимает один аргумент - путь к файлу конфигурации.  
Пример: `aber-calc.exe ./config.txt`  
При необходимости, на windows можно создать bat-файл для этого. Пример такого файла в `start.bat` в текущей директории.
