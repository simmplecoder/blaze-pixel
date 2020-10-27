include(Catch.cmake)

set(test_names pixel_to_vector_tests vector_to_pixel_tests)

add_executable(test_program ${test_names})
target_link_libraries(test_program PRIVATE BlazePixel::BlazePixel)

#catch_discover_tests(test_program)