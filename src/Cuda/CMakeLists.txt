# Modified to add cuda files
macro(add_cuda_source_group module_name)
    file(GLOB MODULE_HEADER_FILES "${module_name}/*.cuh")
    source_group("Header Files\\${module_name}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${module_name}/*.cu")
    source_group("Source Files\\${module_name}" FILES ${MODULE_SOURCE_FILES})
endmacro()


find_package(CUDA 8.0 REQUIRED)
# NOTE: Eigen <= 3.3.5 has problem with CUDA >= 9.0
# Source code have to be modified in Eigen/Core to ignore fp16:
# Comment the following lines:
# #include "src/Core/arch/CUDA/Half.h"
# #include "src/Core/arch/CUDA/PacketMathHalf.h"
# #include "src/Core/arch/CUDA/TypeCasting.h"

list(APPEND 3RDPARTY_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})
list(APPEND 3RDPARTY_LIBRARIES
        ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${3RDPARTY_INCLUDE_DIRS})
link_directories(${3RDPARTY_LIBRARY_DIRS})

# build
file(GLOB_RECURSE CUDA_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE CUDA_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE CUDA_ALL_CUDA_HEADER_FILES "*.cuh")
file(GLOB_RECURSE CUDA_ALL_CUDA_SOURCE_FILES "*.cu")

# (testing) flags for cuda code generation
set(CUDA_NVCC_FLAGS
        -gencode=arch=compute_50,code=sm_60
        -gencode=arch=compute_50,code=sm_50
        --expt-relaxed-constexpr)

# create (should be static) library
set(OPEN3D_CUDA_MODULE "Open3DCuda" CACHE STRING "CUDA component library")
cuda_add_library(${OPEN3D_CUDA_MODULE}
        ${CUDA_ALL_HEADER_FILES} ${CUDA_ALL_SOURCE_FILES}
        ${CUDA_ALL_CUDA_HEADER_FILES} ${CUDA_ALL_CUDA_SOURCE_FILES}
        $<TARGET_OBJECTS:Camera>
        $<TARGET_OBJECTS:ColorMap>
        $<TARGET_OBJECTS:Geometry>
        $<TARGET_OBJECTS:Integration>
        $<TARGET_OBJECTS:Odometry>
        $<TARGET_OBJECTS:Registration>
        $<TARGET_OBJECTS:Utility>
        $<TARGET_OBJECTS:IO>
        $<TARGET_OBJECTS:Visualization>)
target_link_libraries(${OPEN3D_CUDA_MODULE} ${3RDPARTY_LIBRARIES} -lz) # for zlib
target_include_directories(${OPEN3D_CUDA_MODULE} PUBLIC ${3RDPARTY_INCLUDE_DIRS})

# create subgroups
add_source_group(Common)
add_source_group(Camera)
add_source_group(Container)
add_source_group(Geometry)
add_source_group(Integration)
add_source_group(Registration)
add_source_group(IO)
add_source_group(Odometry)
add_source_group(Visualization)

# and also for cuda
add_cuda_source_group(Common)
add_cuda_source_group(Camera)
add_cuda_source_group(Container)
add_cuda_source_group(Geometry)
add_cuda_source_group(Integration)
add_cuda_source_group(Registration)
add_cuda_source_group(IO)
add_cuda_source_group(Odometry)
add_cuda_source_group(Visualization)

# Directories(. CUDA_INCLUDE_DIRS)
# set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
target_include_directories(${OPEN3D_CUDA_MODULE} PUBLIC ${CUDA_INCLUDE_DIRS})

#install headers
install(DIRECTORY "."
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN "*.cpp" EXCLUDE
        PATTERN "*.cuh" EXCLUDE
        PATTERN "*.cu" EXCLUDE
        PATTERN "*.txt" EXCLUDE)
