cmake_minimum_required(VERSION 2.8.3)
project(depth_colorizer)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  roscpp
  rospy
)
find_package(PythonLibs REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/DynamicParams.cfg
  )

include_directories(
  ${catkin_INCLUDE_DIRS}
)
