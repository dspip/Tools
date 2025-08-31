:: run this from the x64 native tools command prompt 
mkdir Debug
mkdir Release
set INCLUDES=/I C:\dev\opencv_4_12_0\build\install\include /I ..\thirdparty\tracy\tracy 
set LIBS=/LIBPATH:"C:\dev\opencv_4_12_0\build\install\x64\vc17\lib" "opencv_core4120.lib" "opencv_video4120.lib" "opencv_imgproc4120.lib" "opencv_videoio4120.lib" "opencv_highgui4120.lib"
 
cl /D TRACY_ENABLE /Z7 /MD /fp:precise %INCLUDES% ..\thirdparty\tracy\TracyClient.cpp nx_vmd_ocv.cpp /Fd:Debug\ /Fo:Debug\ /Fe:Debug\nx_vmd_ocv_test.exe /link /INCREMENTAL:NO /MACHINE:X64 %LIBS% 

::cl /O2 /Z7 /MD /fp:precise %INCLUDES% thirdparty\tracy\TracyClient.cpp nx_osd_renderer.cpp /Fd:Release\ /Fo:Release\ /Fe:Release\nx_osd_test.exe /link /INCREMENTAL:NO /MACHINE:X64 %LIBS% 
::
::cl /D TRACY_ENABLE /Z7 /MD /fp:precise %INCLUDES% thirdparty\tracy\TracyClient.cpp nx_vmd.cpp /Fd:Debug\ /Fo:Debug\ /Fe:Debug\nx_vmd_test.exe /link /INCREMENTAL:NO /MACHINE:X64 %LIBS% 
