cmake_minimum_required (VERSION 3.9)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0072 NEW)
project (galax)

if(GALAX_LINK_CUDA)
    enable_language(CUDA)
endif(GALAX_LINK_CUDA)

set(CMAKE_CUDA_ARCHITECTURES 52 61 75)

option(GALAX_LINK_SDL2 "Link with the SDL2 library (used for graphical display)" OFF)
option(GALAX_LINK_CUDA "Link with the CUDA library (used for GPU model)"         OFF)
option(GALAX_LINK_OMP  "Link with the OMP  library (used for CPU_FAST model)"    OFF)

# Enable C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify bin path
set (EXECUTABLE_OUTPUT_PATH bin/)

# Generate the source files lists
set(SRC_FILES_CUDA    ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Model_GPU/kernel.cu
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Model_GPU/Model_GPU.cpp)

set(SRC_FILES_DISPLAY ${CMAKE_CURRENT_SOURCE_DIR}/src/Display/Display.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Display/Display_SDL2/Display_SDL2.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Display/Display_NO/Display_NO.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Particles.cpp)

set(SRC_FILES_MAIN    ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Model.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Model_CPU/Model_CPU.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Model_CPU/Model_CPU_naive/Model_CPU_naive.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Model_CPU/Model_CPU_fast/Model_CPU_fast.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Initstate.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Timing/Timing.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/src/config.hpp" @ONLY)
include_directories (PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/src/")

# add library & executable
add_library (galax_display SHARED ${SRC_FILES_DISPLAY})
if(GALAX_LINK_CUDA)
    add_library (galax_cuda SHARED ${SRC_FILES_CUDA})
endif(GALAX_LINK_CUDA)
add_executable(galax ${SRC_FILES_MAIN})

# cli
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/cli/include/CLI/CLI.hpp")
    option(CLI_COMPILE_STATIC_LIB "" OFF)
    option(CLI_COMPILE_SHARED_LIB "" OFF)
    option(CLI_STATUS_MESSAGE_QUIET "" ON)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/cli/ EXCLUDE_FROM_ALL)
    message(STATUS "Galax - Sub-project found: cli")
    include_directories (PUBLIC "lib/cli/include")
else()
    message(FATAL_ERROR "Galax - cli can't be found, try to init the submodule with the following cmd:\n"
                        "$ git submodule update --init --recursive")
endif()

# XSIMD
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/xsimd/include/xsimd/xsimd.hpp")
    include_directories(PUBLIC "lib/xsimd/include")
    message(STATUS "Galax - Header found: XSIMD")
else()
    message(FATAL_ERROR "Galax - XSIMD can't be found, try to init the submodule with the following cmd:\n"
            "$ git submodule update --init -- ../lib/XSIMD/")
endif()

if(GALAX_LINK_CUDA)
    target_link_libraries(galax_cuda PUBLIC galax_display)
    target_link_libraries(galax      PUBLIC galax_cuda)
endif(GALAX_LINK_CUDA)

target_link_libraries(galax      PUBLIC galax_display)

# Find OMP
if(GALAX_LINK_OMP)
    target_compile_definitions(galax PUBLIC GALAX_MODEL_CPU_FAST)

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Galax - Library found: OpenMP")
        target_link_libraries(galax PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif(GALAX_LINK_OMP)

# Find Cuda
if(GALAX_LINK_CUDA)
    target_compile_definitions(galax_cuda PUBLIC GALAX_MODEL_GPU)
    find_package(CUDA REQUIRED QUIET)
    if(CUDA_FOUND)
        message(STATUS "Galax - Library found: CUDA")
        include_directories(PUBLIC "${CUDA_INCLUDE_DIRS}")
        target_link_libraries(galax_cuda PUBLIC ${CUDA_LIBRARIES}   )
    endif(CUDA_FOUND)
endif(GALAX_LINK_CUDA)

if(GALAX_LINK_SDL2)
    target_compile_definitions(galax_display PUBLIC GALAX_DISPLAY_SDL2)
    target_compile_definitions(galax         PUBLIC GALAX_DISPLAY_SDL2)

    # Find OpenGL
    find_package(OpenGL REQUIRED QUIET)
    if(OPENGL_FOUND)
        message(STATUS "Galax - Library found: OPENGL")
        include_directories(PUBLIC "${OPENGL_INCLUDE_DIR}")
        target_link_libraries(galax_display PUBLIC ${OPENGL_LIBRARIES})
    endif(OPENGL_FOUND)

    find_package(GLUT REQUIRED)
    if(GLUT_FOUND)
        message(STATUS "Galax - Library found: GLUT")
        if(${GLUT_INCLUDE_DIR}) # In fedora this is not necessarily set for some reason, but everything still works...
            include_directories(PUBLIC "${GLUT_INCLUDE_DIR}")
        endif()
        target_link_libraries(galax_display PUBLIC ${GLUT_LIBRARY})
    endif(GLUT_FOUND)

    find_package(GLEW REQUIRED)
    if(GLEW_FOUND)
        message(STATUS "Galax - Library found: GLEW")
        include_directories(PUBLIC "${GLEW_INCLUDE_DIR}")
        target_link_libraries(galax_display PUBLIC ${GLEW_LIBRARIES})
    endif(GLEW_FOUND)

    # Find SDL2
    find_package(SDL2 REQUIRED)
    if(SDL2_FOUND)
        message(STATUS "Galax - Library found: SDL2")
        include_directories(${SDL2_INCLUDE_DIRS})
        target_link_libraries(galax_display PUBLIC ${SDL2_LIBRARIES})
    endif(SDL2_FOUND)
endif(GALAX_LINK_SDL2)

