# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shenao/Summer/FCNs_Wild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shenao/Summer/FCNs_Wild/build

# Include any dependencies generated for this target.
include lib/constraintloss/CMakeFiles/constraintloss.dir/depend.make

# Include the progress variables for this target.
include lib/constraintloss/CMakeFiles/constraintloss.dir/progress.make

# Include the compile flags for this target's objects.
include lib/constraintloss/CMakeFiles/constraintloss.dir/flags.make

lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o: lib/constraintloss/CMakeFiles/constraintloss.dir/flags.make
lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o: ../lib/constraintloss/constraintsoftmax.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shenao/Summer/FCNs_Wild/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o"
	cd /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o -c /home/shenao/Summer/FCNs_Wild/lib/constraintloss/constraintsoftmax.cpp

lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.i"
	cd /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shenao/Summer/FCNs_Wild/lib/constraintloss/constraintsoftmax.cpp > CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.i

lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.s"
	cd /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shenao/Summer/FCNs_Wild/lib/constraintloss/constraintsoftmax.cpp -o CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.s

lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.requires:

.PHONY : lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.requires

lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.provides: lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.requires
	$(MAKE) -f lib/constraintloss/CMakeFiles/constraintloss.dir/build.make lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.provides.build
.PHONY : lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.provides

lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.provides.build: lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o


# Object files for target constraintloss
constraintloss_OBJECTS = \
"CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o"

# External object files for target constraintloss
constraintloss_EXTERNAL_OBJECTS =

lib/constraintloss/libconstraintloss.a: lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o
lib/constraintloss/libconstraintloss.a: lib/constraintloss/CMakeFiles/constraintloss.dir/build.make
lib/constraintloss/libconstraintloss.a: lib/constraintloss/CMakeFiles/constraintloss.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shenao/Summer/FCNs_Wild/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libconstraintloss.a"
	cd /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss && $(CMAKE_COMMAND) -P CMakeFiles/constraintloss.dir/cmake_clean_target.cmake
	cd /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/constraintloss.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/constraintloss/CMakeFiles/constraintloss.dir/build: lib/constraintloss/libconstraintloss.a

.PHONY : lib/constraintloss/CMakeFiles/constraintloss.dir/build

lib/constraintloss/CMakeFiles/constraintloss.dir/requires: lib/constraintloss/CMakeFiles/constraintloss.dir/constraintsoftmax.cpp.o.requires

.PHONY : lib/constraintloss/CMakeFiles/constraintloss.dir/requires

lib/constraintloss/CMakeFiles/constraintloss.dir/clean:
	cd /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss && $(CMAKE_COMMAND) -P CMakeFiles/constraintloss.dir/cmake_clean.cmake
.PHONY : lib/constraintloss/CMakeFiles/constraintloss.dir/clean

lib/constraintloss/CMakeFiles/constraintloss.dir/depend:
	cd /home/shenao/Summer/FCNs_Wild/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shenao/Summer/FCNs_Wild /home/shenao/Summer/FCNs_Wild/lib/constraintloss /home/shenao/Summer/FCNs_Wild/build /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss /home/shenao/Summer/FCNs_Wild/build/lib/constraintloss/CMakeFiles/constraintloss.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/constraintloss/CMakeFiles/constraintloss.dir/depend
