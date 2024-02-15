rm cuda_test
time nvcc `pkg-config --cflags --libs glib-2.0` -o cuda_test ahd.cu
