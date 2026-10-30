cmake_minimum_required(VERSION 3.8)
project(k1_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes config launch
  DESTINATION share/${PROJECT_NAME}/
)

# booster_assets is a root-level submodule; install its robot files to the
# same share/<pkg>/assets/robots/... layout that scripts/generate_k1_urdf.py expects.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../booster_assets/robots
  DESTINATION share/${PROJECT_NAME}/assets
)

install(PROGRAMS scripts/generate_k1_urdf.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
