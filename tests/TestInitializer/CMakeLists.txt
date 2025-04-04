add_executable(TestSetReference TestSetReference.cc)
target_link_libraries(TestSetReference dso-ssl)
target_include_directories(TestSetReference PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestSetReference2 TestSetReference2.cc)
target_link_libraries(TestSetReference2 dso-ssl)
target_include_directories(TestSetReference2 PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestComputeJacobianAndError TestComputeJacobianAndError.cc)
target_link_libraries(TestComputeJacobianAndError dso-ssl)
target_include_directories(TestComputeJacobianAndError PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestLayerOptimizer TestLayerOptimizer.cc)
target_link_libraries(TestLayerOptimizer dso-ssl)
target_include_directories(TestLayerOptimizer PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestInitializer TestInitializer.cc)
target_link_libraries(TestInitializer dso-ssl)
target_include_directories(TestInitializer PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(TestInitializer2 TestInitializer2.cc)
target_link_libraries(TestInitializer2 dso-ssl)
target_include_directories(TestInitializer2 PRIVATE ${PROJECT_SOURCE_DIR}/include)