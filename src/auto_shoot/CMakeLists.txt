cmake_minimum_required(VERSION 3.5)
project(auto_shoot)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tdt_interface REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	include/auto_shoot
	src
)


file(GLOB SOURCES 
	src/*.cpp 
	src/PreProcess/*.cpp 
	src/Detect/*.cpp 
	src/Armor/*.cpp 
	src/camera/*.cpp 
	src/Kalman/Kalman.cpp
)


add_executable(auto_shoot_node ${SOURCES})


ament_target_dependencies(
	auto_shoot_node 
	rclcpp nav_msgs 
	geometry_msgs 
	sensor_msgs 
	cv_bridge 
	OpenCV 
	tdt_interface 
	Eigen3
)

target_link_libraries(auto_shoot_node Eigen3::Eigen ${OpenCV_LIBS})



install(TARGETS
  auto_shoot_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

