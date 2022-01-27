/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

int MHE_func(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int MHE_func_alloc_mem(void);
int MHE_func_init_mem(int mem);
void MHE_func_free_mem(int mem);
int MHE_func_checkout(void);
void MHE_func_release(int mem);
void MHE_func_incref(void);
void MHE_func_decref(void);
casadi_int MHE_func_n_out(void);
casadi_int MHE_func_n_in(void);
casadi_real MHE_func_default_in(casadi_int i);
const char* MHE_func_name_in(casadi_int i);
const char* MHE_func_name_out(casadi_int i);
const casadi_int* MHE_func_sparsity_in(casadi_int i);
const casadi_int* MHE_func_sparsity_out(casadi_int i);
int MHE_func_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#ifdef __cplusplus
} /* extern "C" */
#endif
