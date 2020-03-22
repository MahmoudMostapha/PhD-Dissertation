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

# Include any dependencies generated for this target.
include Source/CMakeFiles/vtkMeshGeodesics.dir/depend.make

# Include the progress variables for this target.
include Source/CMakeFiles/vtkMeshGeodesics.dir/progress.make

# Include the compile flags for this target's objects.
include Source/CMakeFiles/vtkMeshGeodesics.dir/flags.make

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o: Source/CMakeFiles/vtkMeshGeodesics.dir/flags.make
Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o: ../Source/vtkPolyDataGeodesicDistance.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o -c /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkPolyDataGeodesicDistance.cxx

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.i"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkPolyDataGeodesicDistance.cxx > CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.i

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.s"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkPolyDataGeodesicDistance.cxx -o CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.s

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.requires:

.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.requires

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.provides: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.requires
	$(MAKE) -f Source/CMakeFiles/vtkMeshGeodesics.dir/build.make Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.provides.build
.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.provides

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.provides.build: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o


Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o: Source/CMakeFiles/vtkMeshGeodesics.dir/flags.make
Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o: ../Source/vtkFastMarchingGeodesicDistance.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o -c /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkFastMarchingGeodesicDistance.cxx

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.i"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkFastMarchingGeodesicDistance.cxx > CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.i

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.s"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkFastMarchingGeodesicDistance.cxx -o CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.s

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.requires:

.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.requires

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.provides: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.requires
	$(MAKE) -f Source/CMakeFiles/vtkMeshGeodesics.dir/build.make Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.provides.build
.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.provides

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.provides.build: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o


Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o: Source/CMakeFiles/vtkMeshGeodesics.dir/flags.make
Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o: ../Source/vtkFastMarchingGeodesicPath.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o -c /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkFastMarchingGeodesicPath.cxx

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.i"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkFastMarchingGeodesicPath.cxx > CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.i

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.s"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkFastMarchingGeodesicPath.cxx -o CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.s

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.requires:

.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.requires

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.provides: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.requires
	$(MAKE) -f Source/CMakeFiles/vtkMeshGeodesics.dir/build.make Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.provides.build
.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.provides

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.provides.build: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o


Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o: Source/CMakeFiles/vtkMeshGeodesics.dir/flags.make
Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o: ../Source/vtkPolygonalSurfaceContourLineInterpolator2.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o -c /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkPolygonalSurfaceContourLineInterpolator2.cxx

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.i"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkPolygonalSurfaceContourLineInterpolator2.cxx > CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.i

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.s"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source/vtkPolygonalSurfaceContourLineInterpolator2.cxx -o CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.s

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.requires:

.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.requires

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.provides: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.requires
	$(MAKE) -f Source/CMakeFiles/vtkMeshGeodesics.dir/build.make Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.provides.build
.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.provides

Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.provides.build: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o


# Object files for target vtkMeshGeodesics
vtkMeshGeodesics_OBJECTS = \
"CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o" \
"CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o" \
"CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o" \
"CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o"

# External object files for target vtkMeshGeodesics
vtkMeshGeodesics_EXTERNAL_OBJECTS =

bin/libvtkMeshGeodesics.a: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o
bin/libvtkMeshGeodesics.a: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o
bin/libvtkMeshGeodesics.a: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o
bin/libvtkMeshGeodesics.a: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o
bin/libvtkMeshGeodesics.a: Source/CMakeFiles/vtkMeshGeodesics.dir/build.make
bin/libvtkMeshGeodesics.a: Source/CMakeFiles/vtkMeshGeodesics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library ../bin/libvtkMeshGeodesics.a"
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && $(CMAKE_COMMAND) -P CMakeFiles/vtkMeshGeodesics.dir/cmake_clean_target.cmake
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vtkMeshGeodesics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Source/CMakeFiles/vtkMeshGeodesics.dir/build: bin/libvtkMeshGeodesics.a

.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/build

Source/CMakeFiles/vtkMeshGeodesics.dir/requires: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolyDataGeodesicDistance.cxx.o.requires
Source/CMakeFiles/vtkMeshGeodesics.dir/requires: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicDistance.cxx.o.requires
Source/CMakeFiles/vtkMeshGeodesics.dir/requires: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkFastMarchingGeodesicPath.cxx.o.requires
Source/CMakeFiles/vtkMeshGeodesics.dir/requires: Source/CMakeFiles/vtkMeshGeodesics.dir/vtkPolygonalSurfaceContourLineInterpolator2.cxx.o.requires

.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/requires

Source/CMakeFiles/vtkMeshGeodesics.dir/clean:
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source && $(CMAKE_COMMAND) -P CMakeFiles/vtkMeshGeodesics.dir/cmake_clean.cmake
.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/clean

Source/CMakeFiles/vtkMeshGeodesics.dir/depend:
	cd /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/Source /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source /work/mahmoudm/Subcortical_Processing/Code/GeodesicsOnMeshes/bin/Source/CMakeFiles/vtkMeshGeodesics.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Source/CMakeFiles/vtkMeshGeodesics.dir/depend

