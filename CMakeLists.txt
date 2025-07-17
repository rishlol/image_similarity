cmake_minimum_required(VERSION 3.10)
project(image_similarity)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED CONFIG COMPONENTS program_options)

# Print OpenCV info (optional, for debugging)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

# Add executable
add_executable(image_similarity src/main.cpp)

# Include headers
target_include_directories(image_similarity PRIVATE
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(image_similarity PRIVATE 
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)
