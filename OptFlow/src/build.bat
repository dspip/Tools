mkdir Debug

set gstreamer_include="C:\gstreamer\1.0\msvc_x86_64\include\gstreamer-1.0\\"
set gstreamer_lib_path="C:\gstreamer\1.0\msvc_x86_64\lib\\"
set glib_include="C:\gstreamer\1.0\msvc_x86_64\include\glib-2.0\\"
set glib_include_="C:\gstreamer\1.0\msvc_x86_64\lib\glib-2.0\include\\"
set gstreamer_libs="gstapp-1.0.lib" "gstbase-1.0.lib" "gstreamer-1.0.lib" "gobject-2.0.lib" "glib-2.0.lib"

::cl test_29_6_25.cpp /Z7 /fp:precise /D "__WINDOWS" /EHsc /I %gstreamer_include% /I %glib_include% /I %glib_include_% /link /LIBPATH:%gstreamer_lib_path% /DYNAMICBASE %gstreamer_libs% /MACHINE:X64 /INCREMENTAL:NO /DEBUG /OUT:Debug\test_29_6_25.exe
cl test_nv_opt.cpp /Z7 /fp:precise /D "__WINDOWS" /EHsc /I %gstreamer_include% /I %glib_include% /I %glib_include_% /link /LIBPATH:%gstreamer_lib_path% /DYNAMICBASE %gstreamer_libs% .\NvOptFlowLib\nv_opt_flow_lib.lib /MACHINE:X64 /INCREMENTAL:NO /DEBUG /OUT:Debug\test_nv_opt.exe

copy .\NvOptFlowLib\nv_opt_flow_lib.dll  .\Debug\
copy .\NvOptFlowLib\nv_opt_flow_lib.pdb .\Debug\

