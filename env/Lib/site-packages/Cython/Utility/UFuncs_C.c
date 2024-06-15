///////////////////////// UFuncsInit.proto /////////////////////////
//@proto_block: utility_code_proto_before_types

#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

// account for change in type of arguments to PyUFuncGenericFunction in Numpy 1.19.x
// Unfortunately we can only test against Numpy version 1.20.x since it wasn't marked
// as an API break. Therefore, I'm "solving" the issue by casting function pointer types
// on lower Numpy versions.
#if NPY_API_VERSION >= 0x0000000e // Numpy 1.20.x
#define __PYX_PYUFUNCGENERICFUNCTION_CAST(x) x
#else
#define __PYX_PYUFUNCGENERICFUNCTION_CAST(x) (PyUFuncGenericFunction)x
#endif

/////////////////////// UFuncConsts.proto ////////////////////

// getter functions because we can't forward-declare arrays
static PyUFuncGenericFunction* {{ufunc_funcs_name}}(void); /* proto */
static char* {{ufunc_types_name}}(void); /* proto */
static void* {{ufunc_data_name}}[] = {NULL};  /* always null */

/////////////////////// UFuncConsts /////////////////////////

static PyUFuncGenericFunction* {{ufunc_funcs_name}}(void) {
    static PyUFuncGenericFunction arr[] = {
        {{for loop, cname in looper(func_cnames)}}
        __PYX_PYUFUNCGENERICFUNCTION_CAST(&{{cname}}){{if not loop.last}},{{endif}}
        {{endfor}}
    };
    return arr;
}

static char* {{ufunc_types_name}}(void) {
    static char arr[] = {
        {{for loop, tp in looper(type_constants)}}
        {{tp}}{{if not loop.last}},{{endif}}
        {{endfor}}
    };
    return arr;
}
