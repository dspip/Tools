set PREFIX_PATH=..\..\thirdparty\NVOptFlowSDK_5.0.7
set INCLUDES=/I ..\..\thirdparty\NVOptFlowSDK_5.0.7\Common\NvOFBase /I ..\..\thirdparty\NVOptFlowSDK_5.0.7\Common\Utils /I ..\..\thirdparty\NVOptFlowSDK_5.0.7\NvOFInterface
::set CUDAFLAGS=/I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
::set CUDALIBS="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"
set CUDAFLAGS=/I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
set CUDALIBS="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"

nvcc -c %PREFIX_PATH%\Common\Utils\kernel.cu
::cl %INCLUDES% -c %PREFIX_PATH%\Common/Utils/NvOFUtils.cpp
::cl %INCLUDES% /c %CUDAFLAGS% %PREFIX_PATH%\Common\Utils\NvOFUtilsCuda.cpp
cl %INCLUDES% /EHsc /c %PREFIX_PATH%\Common\NvOFBase\NvOF.cpp 
cl %INCLUDES% %CUDAFLAGS% /EHsc /c %PREFIX_PATH%\Common\NvOFBase\NvOFCuda.cpp 

cl /Zi /D __WINDOWS %INCLUDES%  kernel.obj NvOF.obj NvOFCuda.obj nv_opt_flow_lib.cpp /EHsc /LD %CUDAFLAGS% /link /DEBUG /DLL /LIBPATH:%CUDALIBS% /MACHINE:X64 /INCREMENTAL:NO cudart.lib cuda.lib /implib:nv_opt_flow_lib.lib /OUT:nv_opt_flow_lib.dll

