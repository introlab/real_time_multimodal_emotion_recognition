cmake_minimum_required(VERSION 3.5)
project(chatbot_gui_emotion)


# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(t_top_hbba_lite REQUIRED)
find_package(opentera_link_srvs REQUIRED)
find_package(opentera_link_msgs REQUIRED)



set(chatbot_gui_emotion_components Core Widgets Gui OpenGL WebKitWidgets)
find_package(Qt5 COMPONENTS ${chatbot_gui_emotion_components} REQUIRED)

set(moc_headers
  src/widgets/MainWindow.h
  src/widgets/FaceView.h
  src/widgets/UserSelectorDialog.h
)

set(srcs
  src/chatbot_gui_emotion_node.cpp
  src/widgets/MainWindow.cpp
  src/widgets/FaceView.cpp
  src/widgets/UserSelectorDialog.cpp
)

qt5_wrap_cpp(project_moc_srcs ${moc_headers})

add_executable(${PROJECT_NAME}_node ${srcs} ${uis} ${moc_headers} ${other_headers} ${project_moc_uis} ${project_moc_srcs})
qt5_use_modules(${PROJECT_NAME}_node ${chatbot_gui_emotion_components})
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  rcl_interfaces
  t_top_hbba_lite
  opentera_link_srvs
  opentera_link_msgs
)

include_directories(src)

install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})

# Launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
