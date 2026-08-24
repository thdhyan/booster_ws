cmake_minimum_required(VERSION 3.8)
project(k1_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes config launch assets
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS scripts/generate_k1_urdf.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
