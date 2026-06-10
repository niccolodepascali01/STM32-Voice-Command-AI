cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(zephyr_ai_demo)

# Includi le cartelle degli header
target_include_directories(app PRIVATE src/cubeai/Inc)
target_include_directories(app PRIVATE src/cubeai)

# Aggiungi i file da compilare (ATTENZIONE: metti i nomi esatti dei TUOI file qui!)
target_sources(app PRIVATE
    src/main.c
    src/cubeai/network.c
    src/cubeai/network_data.c
    src/audio_data.c
)

# Collega la libreria statica della rete neurale (Controlla il nome esatto!)
target_link_libraries(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib/NetworkRuntime1201_CM4_GCC.a)