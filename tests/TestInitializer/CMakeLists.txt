add_executable(TestSetReference TestSetReference.cc)
target_link_libraries(TestSetReference dso-ssl)
target_include_directories(TestProject PRIVATE ${PROJECT_SOURCE_DIR}/include)