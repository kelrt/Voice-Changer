cmake_minimum_required(VERSION 3.20)
project(voice-changer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(voice_changer_app
        src/main.cpp
        src/PythonBridge.cpp
)

target_include_directories(voice_changer_app PRIVATE
        third_party        # miniaudio.h (header-only, vendored)
        src
        src/indicators
)

# miniaudio loads ALSA/PulseAudio at runtime: needs libdl + pthreads, not audio dev packages
target_link_libraries(voice_changer_app PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)
