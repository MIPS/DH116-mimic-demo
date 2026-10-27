set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV_TRIPLE "riscv64-linux-gnu" CACHE STRING "Cross compiler prefix; the tools must be on PATH")
set(CMAKE_C_COMPILER "${RISCV_TRIPLE}-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TRIPLE}-g++")
set(CMAKE_ASM_COMPILER "${RISCV_TRIPLE}-gcc")

# RVA23U64 written out because GCC 15 rejects -march=rva23u64. No zvl*b suffix beyond the
# V baseline (VLEN >= 128): a zvl1024b build crashes on cores with shorter vectors (the
# SpacemiT K3 has VLEN 256), while code built for VLEN 128 runs unchanged on 256 and 1024.
set(RVA23_MARCH "rv64imafdcv_zicsr_zifencei_zicntr_zihpm_zihintpause_zihintntl_zicond_zimop_zcmop_zba_zbb_zbs_zfa_zfhmin_zcb_zawrs_zvbb_zvkt_zvfhmin")
set(_rva23_flags "-march=${RVA23_MARCH} -mabi=lp64d")
set(CMAKE_C_FLAGS_INIT "${_rva23_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_rva23_flags}")
set(CMAKE_ASM_FLAGS_INIT "${_rva23_flags}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")

# Builds that pass -DCMAKE_<LANG>_FLAGS (ONNX Runtime's build.py) bypass *_FLAGS_INIT.
foreach(_lang C CXX)
  if(DEFINED CMAKE_${_lang}_FLAGS AND NOT CMAKE_${_lang}_FLAGS MATCHES "-march=")
    string(APPEND CMAKE_${_lang}_FLAGS " ${_rva23_flags}")
  endif()
endforeach()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
