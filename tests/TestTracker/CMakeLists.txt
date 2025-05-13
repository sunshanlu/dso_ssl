add_executable(TestUpdateTracker TestUpdateTracker.cc)
target_link_libraries(TestUpdateTracker dso-ssl)
target_include_directories(TestUpdateTracker PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestTryOnce TestTryOnce.cc)
target_link_libraries(TestTryOnce dso-ssl)
target_include_directories(TestTryOnce PRIVATE ${PROJECT_SOURCE_DIR}/include)


add_executable(TestTrackActivateFrame TestTrackActivateFrame.cc)
target_link_libraries(TestTrackActivateFrame dso-ssl)
target_include_directories(TestTrackActivateFrame PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestIsNeedKeyFrame TestIsNeedKeyFrame.cc)
target_link_libraries(TestIsNeedKeyFrame dso-ssl)
target_include_directories(TestIsNeedKeyFrame PRIVATE ${PROJECT_SOURCE_DIR}/include)