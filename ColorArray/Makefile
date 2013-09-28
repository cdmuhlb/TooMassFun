# User settings
# =============

# Compute capability of target GPU (passed to '-arch' flag)
CUDA_ARCH = sm_30

# Location of the CUDA Toolkit
CUDA_PATH = /usr/local/cuda

# Compilers and flags
CXX = g++
CXXFLAGS = -O3 -march=native -Wall -Wextra
NVCC = nvcc
NVCCFLAGS = -O3 -arch=$(CUDA_ARCH)

# Library paths
#   Example LIB: -L /usr/lib
#   Example INCLUDE: -I /usr/include
OPENCV_LIB =
OPENCV_INCLUDE =


# Internal settings
# =================

# Includes and libraries
CUDA_INCLUDE = -I $(CUDA_PATH)/include
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui
INCLUDES = $(OPENCV_INCLUDE) $(CUDA_INCLUDE)
LIBRARIES = $(OPENCV_LIB) $(OPENCV_LIBS)


# Targets
# =======

all: bin/colorarray

bin/colorarray: bin build/main.o build/kernels.o
	$(NVCC) build/main.o build/kernels.o -o $@ $(LIBRARIES) $(NVCCFLAGS)

bin:
	mkdir -p bin

build/main.o: build src/main.cc src/utils.h src/kernels.h
	$(CXX) -c src/main.cc -o $@ $(INCLUDES) $(CXXFLAGS)

build/kernels.o: build src/kernels.cu src/kernels.h src/utils.h
	$(NVCC) -c src/kernels.cu -o $@ $(NVCCFLAGS)

build:
	mkdir -p build

clean:
	rm -f build/main.o build/kernels.o bin/colorarray
