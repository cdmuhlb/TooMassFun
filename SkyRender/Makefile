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
NVCC_INCLUDES = $(OPENCV_INCLUDE)
CXX_INCLUDES = $(NVCC_INCLUDES) $(CUDA_INCLUDE)
LIBRARIES = $(OPENCV_LIB) $(OPENCV_LIBS)


# Targets
# =======

all: bin/skyrender

bin/skyrender: build/main.o build/kernels.o
	mkdir -p bin
	$(NVCC) build/main.o build/kernels.o -o $@ $(LIBRARIES) $(NVCCFLAGS)

build/main.o: src/main.cu src/kernels.h src/utils.h
	mkdir -p build
	$(NVCC) -c src/main.cu -o $@ $(NVCC_INCLUDES) $(NVCCFLAGS)

build/kernels.o: src/kernels.cu src/kernels.h src/utils.h
	mkdir -p build
	$(NVCC) -c src/kernels.cu -o $@ $(NVCC_INCLUDES) $(NVCCFLAGS)

clean:
	rm -f build/main.o build/kernels.o bin/skyrender
