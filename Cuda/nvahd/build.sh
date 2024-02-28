mv cuda_ahd p_cuda_ahd
time nvcc -G -g `pkg-config --cflags --libs glib-2.0` -o cuda_ahd ahd.cu
