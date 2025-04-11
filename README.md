# Нейронные сети с нуля
Библиотека для работы с feed-forward нейронными сетями, реализованная с нуля в качестве курсового проекта.

После скачивания репозитория, для тестов MNIST нужно скачать папку `data` из https://disk.yandex.ru/d/VxnjG83fZZnArw и поместить в ее в корень проекта.

Инструкция для сборки и запуска тестов:
```sh
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make mnist_tests
tests/mnist_tests
```