# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /NIRAL/tools/CMake/cmake-3.5.1/bin/cmake

# The command to remove a file.
RM = /NIRAL/tools/CMake/cmake-3.5.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin

# Utility rule file for NightlyTest.

# Include the progress variables for this target.
include CMakeFiles/NightlyTest.dir/progress.make

CMakeFiles/NightlyTest:
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin && /NIRAL/tools/CMake/cmake-3.5.1/bin/ctest -D NightlyTest

NightlyTest: CMakeFiles/NightlyTest
NightlyTest: CMakeFiles/NightlyTest.dir/build.make

.PHONY : NightlyTest

# Rule to build all files generated by this target.
CMakeFiles/NightlyTest.dir/build: NightlyTest

.PHONY : CMakeFiles/NightlyTest.dir/build

CMakeFiles/NightlyTest.dir/clean:
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin && $(CMAKE_COMMAND) -P CMakeFiles/NightlyTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NightlyTest.dir/clean

CMakeFiles/NightlyTest.dir/depend:
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/CMakeFiles/NightlyTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NightlyTest.dir/depend

