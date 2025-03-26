cmake_minimum_required(VERSION 3.10)
project(workspace)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(realsense2 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Print package information for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "RealSense library: ${realsense2_LIBRARY}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}    # Root directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Include directory
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Add realsense include directory if available
if(realsense2_INCLUDE_DIR)
    include_directories(${realsense2_INCLUDE_DIR})
else()
    # Fallback for systems where the package config doesn't set the include dir
    include_directories(/usr/include/librealsense2)
endif()

# Add library
add_library(workspace_definition
    src/workspace_definition.cpp
)

# Link libraries
target_link_libraries(workspace_definition
    ${OpenCV_LIBS}
)
# target_link_libraries(workspace_definition PRIVATE yaml-cpp)

# Link with RealSense library - handle different variable names
if(realsense2_LIBRARY)
    target_link_libraries(workspace_definition ${realsense2_LIBRARY})
else()
    # Fallback method
    target_link_libraries(workspace_definition realsense2)
endif()

# Add executable
add_executable(workspace_tool
    src/workspace_tool.cpp
)

# Link executable with our library
target_link_libraries(workspace_tool
    workspace_definition
)

target_link_libraries(workspace_tool yaml-cpp)

# Install targets
install(TARGETS workspace_tool DESTINATION bin)