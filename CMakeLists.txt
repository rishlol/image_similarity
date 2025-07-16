cmake_minimum_required(VERSION 3.10)
project(image_similarity)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Print OpenCV info (optional, for debugging)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

# Add executable
add_executable(image_similarity src/main.cpp)

# Include OpenCV headers
target_include_directories(image_similarity PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
target_link_libraries(image_similarity PRIVATE ${OpenCV_LIBS})
