cmake_minimum_required(VERSION 3.15)
project(blaze-pixel)

set(CXX_STANDARD 17)
set(CXX_EXTENSIONS OFF)
set(CXX_STANDARD_REQUIRED ON)

add_library(BlazePixel INTERFACE)
add_library(BlazePixel::BlazePixel ALIAS BlazePixel)
target_include_directories(BlazePixel INTERFACE include)


### BlazeComplete: Link blaze with Intel MKL into one
add_library(BlazeComplete INTERFACE)

find_package(blaze REQUIRED)

if (DEFINED MKLROOT)
    message(STATUS "MKLROOT variable is set, setting environment variable MKLROOT, with possible override")
    set(ENV{MKLROOT} ${MKLROOT})
endif()

if (NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "Please set MKLROOT environment variable, as Intel MKL is required")
endif()

find_package(BLAS REQUIRED)
target_include_directories(BlazeComplete INTERFACE $ENV{MKLROOT}/include)
target_link_libraries(BlazeComplete INTERFACE blaze::blaze BLAS::BLAS)
### end BlazeComplete

find_package(Boost REQUIRED COMPONENTS filesystem)

target_link_libraries(BlazePixel INTERFACE Boost::headers Boost::filesystem BlazeComplete)

add_subdirectory(test)