PREFIX_PATH=../../thirdparty/NVOptFlowSDK_5.0.7
INCLUDES="-I ../../thirdparty/NVOptFlowSDK_5.0.7/Common/NvOFBase -I ../../thirdparty/NVOptFlowSDK_5.0.7/Common/Utils/ -I../../thirdparty/NVOptFlowSDK_5.0.7/NvOFInterface"
CUDAFLAGS=`pkg-config --libs --cflags cuda-12.8 cudart-12.8` 
echo $CUDAFLAGS
echo $INCLUDES


nvcc --compiler-options '-fPIC' -c $PREFIX_PATH/Common/Utils/kernel.cu
g++ $INCLUDES -fPIC -c $PREFIX_PATH/Common/NvOFBase/NvOF.cpp 
g++ $INCLUDES -fPIC -c $PREFIX_PATH/Common/NvOFBase/NvOFCuda.cpp 

g++ $INCLUDES -fPIC -shared -o nv_opt_flow_lib.so kernel.o NvOF.o NvOFCuda.o nv_opt_flow_lib.cpp $CUDAFLAGS 
