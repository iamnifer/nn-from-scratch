

file(GLOB SOURCES "../*.cpp")

add_executable(basic_tests basic_tests.cpp ${SOURCES})
target_link_libraries(basic_tests PRIVATE gtest gtest_main)
target_include_directories(basic_tests PRIVATE ../extern ../extern/eigen)

add_executable(mnist_tests mnist_tests.cpp ${SOURCES})
target_link_libraries(mnist_tests PRIVATE gtest gtest_main)
target_include_directories(mnist_tests PRIVATE ../extern ../extern/eigen)

add_test(NAME basic COMMAND basic_tests)
add_test(NAME mnist COMMAND mnist_tests)