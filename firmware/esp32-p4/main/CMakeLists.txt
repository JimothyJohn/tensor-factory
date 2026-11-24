# Every main/models/*.espdl, *.rgb and *.jpg is embedded; tools/ write models.inc,
# detect.inc, frames.inc and jpegs.inc to match (empty files are fine: that part is then skipped).
file(GLOB model_files "${CMAKE_CURRENT_SOURCE_DIR}/models/*.espdl")
file(GLOB frame_files "${CMAKE_CURRENT_SOURCE_DIR}/models/*.rgb")
file(GLOB jpeg_files "${CMAKE_CURRENT_SOURCE_DIR}/models/*.jpg")
foreach(inc models.inc detect.inc frames.inc jpegs.inc)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/${inc}")
        file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/models/${inc}" "")
    endif()
endforeach()

idf_component_register(SRCS app_main.cpp
                       INCLUDE_DIRS models
                       REQUIRES esp-dl esp_timer esp_rom)

idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)  # utilities.cmake reads this name
include(${cmake_dir}/utilities.cmake)
foreach(f ${model_files} ${frame_files})
    file(RELATIVE_PATH rel ${CMAKE_CURRENT_SOURCE_DIR} ${f})
    target_add_aligned_binary_data(${COMPONENT_LIB} ${rel} BINARY)
endforeach()
foreach(f ${jpeg_files})  # plain embed: these need an _end symbol for their length
    target_add_binary_data(${COMPONENT_LIB} ${f} BINARY)
endforeach()
