# CPU

if(MNN_SUPPORT_RENDER)
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/compute/* ${CMAKE_CURRENT_LIST_DIR}/render/*)
else()
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/compute/*)
endif()
add_library(MNNCPU OBJECT ${MNN_CPU_SRC})

# AME instructions (XSAI matrix ext) need the zames/zmasync march for this TU.
# xsai_alloc.c maps the reserved physical pool for AME operand storage.
# AME 操作数池的**编译期兜底**地址。运行时的解析顺序是
#   XSAI_AME_RESERVED_BASE 环境变量 → /proc/device-tree/reserved-memory/direct_map@*/reg
#   → 这里的常量（命中会打 "[xsai_alloc] WARNING: falling back to the compile-time ..."）。
# 默认沿用 appliance 的 8GB 布局；mega（16GB 总内存 / 4GB 保留，池 @0x380000000）重编时传
#   -DXSAI_RESERVED_PHYS_BASE_ADDR=0x380000000ULL -DXSAI_RESERVED_MEMORY_SIZE=(4096L*1024L*1024L)
set(XSAI_RESERVED_PHYS_BASE_ADDR "0x100000000ULL" CACHE STRING "AME pool compile-time physical base")
set(XSAI_RESERVED_MEMORY_SIZE   "(64L*1024L*1024L)" CACHE STRING "AME pool compile-time size")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64")
    target_sources(MNNCPU PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ame/xsai_alloc.c)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/compute/ame_xsa.c
        PROPERTIES COMPILE_OPTIONS "-march=rv64gcv_zames_zmasync_zvl128b")
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/ame/xsai_alloc.c
        PROPERTIES COMPILE_OPTIONS "-DRESERVED_PHYS_BASE_ADDR=${XSAI_RESERVED_PHYS_BASE_ADDR};-DRESERVED_MEMORY_SIZE=${XSAI_RESERVED_MEMORY_SIZE}")
endif()
if (MNN_PIPELINE_PROFILE)
    target_compile_options(MNNCPU PRIVATE -DMNN_PIPELINE_PROFILE)
endif()
if (MNN_SUPPORT_BF16)
    include(${CMAKE_CURRENT_LIST_DIR}/bf16/CMakeLists.txt)
    list(APPEND MNN_TARGETS MNN_BF16)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_BF16>)
    target_compile_options(MNNCPU PRIVATE -DMNN_SUPPORT_BF16)
endif()
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCPU>)
list(APPEND MNN_TARGETS MNNCPU)


if(MNN_USE_SPARSE_COMPUTE)
  target_compile_options(MNNCPU PRIVATE -DMNN_USE_SPARSE_COMPUTE)
endif()

# X86_64 AVX/SSE
if (MNN_USE_SSE)
    include(${CMAKE_CURRENT_LIST_DIR}/x86_x64/CMakeLists.txt)
endif()

# AArch32/64 Assemblies
include(${CMAKE_CURRENT_LIST_DIR}/arm/CMakeLists.txt)
if(MNN_USE_RVV)
    include(${CMAKE_CURRENT_LIST_DIR}/riscv/CMakeLists.txt)
endif()
IF(NOT DEFINED IOS_ARCH)
    set(IOS_ARCH "")
ENDIF()

# ARM82 Assemblies
IF(MNN_ARM82)
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64")
        target_compile_options(MNNCPU PRIVATE -DENABLE_ARMV82)
        include(${CMAKE_CURRENT_LIST_DIR}/../arm82/CMakeLists.txt)
        list(APPEND MNN_TARGETS MNN_Arm82)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Arm82>)
    ENDIF()
ENDIF()