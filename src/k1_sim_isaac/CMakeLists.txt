cmake_minimum_required(VERSION 3.16)
project(k1_sim_isaac)

find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

# Install config files
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# Install scripts
install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts)

ament_package()
