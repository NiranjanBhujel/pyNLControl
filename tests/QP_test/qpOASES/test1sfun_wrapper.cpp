
/*
 * Include Files
 *
 */
#if defined(MATLAB_MEX_FILE)
#include "tmwtypes.h"
#include "simstruc_types.h"
#else
#include "rtwtypes.h"
#endif



/* %%%-SFUNWIZ_wrapper_includes_Changes_BEGIN --- EDIT HERE TO _END */
#include <math.h>
#include "Test1_Exported/test1.h"
#include "Test1_Exported/test1_EVAL_CODE.h"
#include "Test1_Exported/test1_EVAL_CODE_Call.h"
/* %%%-SFUNWIZ_wrapper_includes_Changes_END --- EDIT HERE TO _BEGIN */
#define u_width 2
#define y_width 1

/*
 * Create external references here.  
 *
 */
/* %%%-SFUNWIZ_wrapper_externs_Changes_BEGIN --- EDIT HERE TO _END */
/* extern double func(double a); */
/* %%%-SFUNWIZ_wrapper_externs_Changes_END --- EDIT HERE TO _BEGIN */

/*
 * Output function
 *
 */
void test1sfun_Outputs_wrapper(const real_T *xGuess,
			const real_T *a,
			const real_T *b,
			real_T *xOpt_VAL,
			real_T *Obj_VAL)
{
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_BEGIN --- EDIT HERE TO _END */
double *xGuess_TEMP = (double *)xGuess;
double *a_TEMP = (double *)a;
double *b_TEMP = (double *)b;
test1_Call(xGuess_TEMP, a_TEMP, b_TEMP, xOpt_VAL, Obj_VAL);
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_END --- EDIT HERE TO _BEGIN */
}


