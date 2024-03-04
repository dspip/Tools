mv hqlinear_debayer p_hqlinear_debayer
time nvcc -G -g `pkg-config --cflags --libs glib-2.0` -o hqlinear_debayer hqlinear.cu 
