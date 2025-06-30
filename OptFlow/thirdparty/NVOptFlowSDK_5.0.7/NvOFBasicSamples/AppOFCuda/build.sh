INCLUDES="-I ../../Common/NvOFBase -I ../../Common/Utils/ -I ../../Common/External/FreeImage/  -I../../NvOFInterface/"
CUDAFLAGS=`pkg-config --libs --cflags cuda-12.8 cudart-12.8` 
EXTRALIBS=-lfreeimage
echo $CUDAFLAGS
echo $INCLUDES


nvcc -c ../../Common/Utils/kernel.cu
g++ $INCLUDES -c ../../Common/Utils/NvOFUtils.cpp
g++ $INCLUDES -c ../../Common/Utils/NvOFUtilsCuda.cpp
g++ $INCLUDES -c ../../Common/Utils/NvOFDataLoader.cpp 
g++ $INCLUDES -c ../../Common/NvOFBase/NvOF.cpp 
g++ $INCLUDES -c ../../Common/NvOFBase/NvOFCuda.cpp 

g++ $INCLUDES -o appOFCuda kernel.o NvOF.o NvOFCuda.o NvOFUtils.o NvOFUtilsCuda.o NvOFDataLoader.o AppOFCuda.cpp $CUDAFLAGS $EXTRALIBS 
#g++ $INCLUDES  NvOF.o NvOFCuda.o NvOFUtils.o NvOFUtilsCuda.o NvOFDataLoader.o AppOFCuda.cpp $CUDAFLAGS 

