# Every main/models/*.espdl and main/models/*.rgb is embedded; tools/ write models.inc,
# detect.inc and frames.inc to match (empty files are fine: that part is then skipped).
file(GLOB model_files "${CMAKE_CURRENT_SOURCE_DIR}/models/*.espdl")
file(GLOB frame_files "${CMAKE_CURRENT_SOURCE_DIR}/models/*.rgb")
foreach(inc models.inc detect.inc frames.inc)
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
