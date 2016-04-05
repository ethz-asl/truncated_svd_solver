cmake_minimum_required(VERSION 2.8.3)
project(truncated_svd_solver)

find_package(catkin_simple REQUIRED)
catkin_simple()

add_definitions("-std=c++11")

###########
# LIBRARY #
###########
cs_add_library(${PROJECT_NAME}
  src/linear-algebra-helpers.cc
  src/marginalization.cc
  src/timing.cc
  src/tsvd-solver.cc
  src/tsvd-solver-options.cc
)
target_link_libraries(${PROJECT_NAME} pthread ${TBB_LIBRARIES})

#########
# TESTS #
#########
catkin_add_gtest(test_svd_solver
  test/test-tsvd-solver.cc
)
target_link_libraries(test_svd_solver ${PROJECT_NAME})

catkin_add_gtest(test_marginalization
  test/test-marginalization.cc
)
target_link_libraries(test_marginalization ${PROJECT_NAME})

###########
# EXPORTS #
###########
cs_install()
cs_export()
