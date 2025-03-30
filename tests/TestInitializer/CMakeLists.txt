add_executable(TestSetReference TestSetReference.cc)
target_link_libraries(TestSetReference dso-ssl)
target_include_directories(TestProject PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestLayerOptimizer TestLayerOptimizer.cc)
target_link_libraries(TestLayerOptimizer dso-ssl)
target_include_directories(TestLayerOptimizer PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestInitializer TestInitializer.cc)
target_link_libraries(TestInitializer dso-ssl)
target_include_directories(TestInitializer PRIVATE ${PROJECT_SOURCE_DIR}/include)