add_executable(TestPixelUndistorter TestPixelUndistorter.cc)
target_link_libraries(TestPixelUndistorter dso-ssl)
target_include_directories(TestPixelUndistorter PRIVATE ${PROJECT_SOURCE_DIR}/include)


add_executable(TestPhotoUndistorter TestPhotoUndistorter.cc)
target_link_libraries(TestPhotoUndistorter dso-ssl)
target_include_directories(TestPhotoUndistorter PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestUndistorter TestUndistorter.cc)
target_link_libraries(TestUndistorter dso-ssl)
target_include_directories(TestUndistorter PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestFrame TestFrame.cc)
target_link_libraries(TestFrame dso-ssl)
target_include_directories(TestFrame PRIVATE ${PROJECT_SOURCE_DIR}/include)