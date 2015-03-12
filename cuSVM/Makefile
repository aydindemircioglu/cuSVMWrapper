#=================================================================================
# Makefile : cuSVM Build File for Linux Fedora 8 64 bit
#---------------------------------------------------------------------------------------------------------------------------------------------
# References:
#   
#----------------------------------------------------------------------------------------------------------------------------------------------
# Notes:
#   1)  Makefile assumes both nvopts.sh and nvmex are in the user's local bin directory
#
#   2)  Makefile assumes default installation directory for both CUDA and Matlab i.e. /usr/local/cuda and /usr/
#	  local/matlab
#
#   3) Makefile assumes the NVIDIA_CUDA_SDK folder is located in the user's home directory
#----------------------------------------------------------------------------------------------------------------------------------------------
# Author:  Shaun Greenamyer
#
# Creation Date: 07/08/2010
#----------------------------------------------------------------------------------------------------------------------------------------------
# Revision History:
#================================================================================

# Include Paths
CUDAHOME    = /usr/local/cuda

SDK_HOME    = /home/$(USER)/NVIDIA_CUDA_SDK
MATLAB	    :=/usr/local/matlab
PREDICT     :=cuSVMPredict
TRAIN	    :=cuSVMTrain

INCLUDEDIR  := -I$(CUDAHOME)/include -I$(SDK_HOME)/C/common/inc

INCLUDECCDIR :=$(INCLUDEDIR) -I$(MATLAB)/extern/include -Iinc

INCLUDELIB  := -L$(CUDAHOME)/lib64 -lcufft -lcudart -Wl,-rpath,$(CUDAHOME)/lib64

LIB  := -lcublas -lcuda

# compiler flags
CFLAGS      :=-fPIC -D_GNU_SOURCE
COMMONFLAGS := 
NVCCFLAGS   :=--compiler-options $(CFLAGS) 

FASTMATH  := --use_fast_math

NVCCFLAGS += -maxrregcount 64
NVCCFLAGS += $(COMMONFLAGS)


#compilers
NVMEX = nvmex
NVCC  = nvcc


#source files
PCUFILES :=cuSVMPredictKernel.cu

TCUFILES :=cuSVMSolver.cu

PCPPFILES :=cuSVMPredict.cpp

TCPPFILES :=cuSVMTrain.cpp

.SUFFIXES : .cu .cpp .o .mexa64

all: $(PCUFILES).o $(TCUFILES).o $(PCPPFILES).mexa64 $(TCPPFILES).mexa64 

$(PCUFILES).o: 
	$(NVCC) -o $(PREDICT)/$@ -c $(PREDICT)/$(PCUFILES) $(NVCCFLAGS) $(INCLUDECCDIR) 

$(TCUFILES).o:
	$(NVCC) -o $(TRAIN)/$@ -c $(TRAIN)/$(TCUFILES) $(NVCCFLAGS) $(INCLUDECCDIR) 

$(PCPPFILES).mexa64:
	$(NVMEX) -f $(HOME)/bin/nvopts.sh $(PREDICT)/$(PCPPFILES) $(PREDICT)/$(PCUFILES).o $(INCLUDELIB) $(LIB)

$(TCPPFILES).mexa64:
	$(NVMEX) -f $(HOME)/bin/nvopts.sh $(TRAIN)/$(TCPPFILES) $(TRAIN)/$(TCUFILES).o $(INCLUDELIB) $(LIB)

	
clean:
	rm -f $(PREDICT)/$(PCUFILES).o
	rm -f $(TRAIN)/$(TCUFILES).o

