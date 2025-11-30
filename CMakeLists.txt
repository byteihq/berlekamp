cmake_minimum_required (VERSION 3.11)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(/W3 /WX)

project ("berlekamp")

# Main reference (original) executable
add_executable(berlekamp berlekamp.cpp  poly.cpp)

# Try to find OpenCL and link to 'parallel' target if found
find_package(OpenCL)
if (OpenCL_FOUND)
  message(STATUS "Found OpenCL: ${OpenCL_INCLUDE_DIRS}")
  target_include_directories(parallel PRIVATE ${OpenCL_INCLUDE_DIRS})
  if (TARGET OpenCL::OpenCL)
    target_link_libraries(parallel PRIVATE OpenCL::OpenCL)
  else()
    target_link_libraries(parallel PRIVATE ${OpenCL_LIBRARIES})
  endif()
else()
  message(WARNING "OpenCL not found. 'parallel' will build but use CPU fallback.")
endif()
