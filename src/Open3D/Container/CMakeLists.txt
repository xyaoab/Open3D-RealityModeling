# TODO: make CUDA optional
find_package(CUDA 8.0 REQUIRED)

# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp" "*.cu")

# create object library
cuda_add_library(Container ${ALL_SOURCE_FILES})
target_include_directories(Container PUBLIC ${CUDA_INCLUDE_DIRS})
