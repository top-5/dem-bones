cmake_minimum_required(VERSION 3.20)
project(dem_bones_avatar_smoke LANGUAGES CXX)

include(FetchContent)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

add_executable(dem_bones_avatar_smoke synthetic_solver.cpp)
target_compile_features(dem_bones_avatar_smoke PRIVATE cxx_std_17)
target_include_directories(dem_bones_avatar_smoke PRIVATE ../../../include)
target_link_libraries(dem_bones_avatar_smoke PRIVATE Eigen3::Eigen)
