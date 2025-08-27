#include <stdint.h>
#ifdef linux 
#define DLL extern "C"
#else 
#ifdef __WINDOWS 
#define DLL extern "C" __declspec(dllexport)
#else 
#define DLL 
#endif
#endif
DLL void * nv_opt_flow_get_context(uint32_t w, uint32_t h, uint32_t gridsize, int bf);
DLL void nv_opt_flow_get_flow_field(void * contextptr, uint8_t * &data, uint8_t * out_data,uint32_t & w,uint32_t & h);
