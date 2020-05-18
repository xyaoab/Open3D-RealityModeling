function(open3d_show_and_abort_on_warning trgt)
    if(MSVC)
        # target_compile_options(${trgt} PRIVATE /W4 /WX)
        target_compile_options(${trgt} PRIVATE /W3)
    else()
        # target_compile_options(${trgt} PRIVATE -Wall -Wextra -Werror)
        target_compile_options(${trgt} PRIVATE -Wall -Werror)
    endif()
endfunction()

add_subdirectory(Open3D)
add_subdirectory(Tools)

# > Enable this by default in this branch
if (BUILD_CUDA_MODULE)
    add_subdirectory(Cuda)
endif()

# > Disable this before we fix everything
if (BUILD_UNIT_TESTS)
    add_subdirectory(UnitTest)
endif()
if (BUILD_BENCHMARKS)
    add_subdirectory(Benchmark)
endif()
if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif()
if (BUILD_TENSORFLOW_OPS AND WIN32)
    message( FATAL_ERROR "Building Tensorflow ops on Windows is currently not supported." )
    # see https://github.com/tensorflow/custom-op/issues/24
endif()
if (BUILD_TENSORFLOW_OPS)
    add_subdirectory(TensorflowOps)
endif()

# find_package Open3D
configure_package_config_file(Open3DConfig.cmake.in
                              "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Open3DConfig.cmake"
                              INSTALL_DESTINATION ${Open3D_INSTALL_CMAKE_DIR}
                              PATH_VARS Open3D_INSTALL_INCLUDE_DIR
)

# find_package Open3D Version
write_basic_package_version_file("${PROJECT_BINARY_DIR}/Open3DConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY ExactVersion)


