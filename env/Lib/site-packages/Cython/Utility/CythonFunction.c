
//////////////////// CythonFunctionShared.proto ////////////////////

#define __Pyx_CyFunction_USED

#define __Pyx_CYFUNCTION_STATICMETHOD  0x01
#define __Pyx_CYFUNCTION_CLASSMETHOD   0x02
#define __Pyx_CYFUNCTION_CCLASS        0x04
#define __Pyx_CYFUNCTION_COROUTINE     0x08

#define __Pyx_CyFunction_GetClosure(f) \
    (((__pyx_CyFunctionObject *) (f))->func_closure)

#if PY_VERSION_HEX < 0x030900B1 || CYTHON_COMPILING_IN_LIMITED_API
  #define __Pyx_CyFunction_GetClassObj(f) \
      (((__pyx_CyFunctionObject *) (f))->func_classobj)
#else
  #define __Pyx_CyFunction_GetClassObj(f) \
      ((PyObject*) ((PyCMethodObject *) (f))->mm_class)
#endif
#define __Pyx_CyFunction_SetClassObj(f, classobj)  \
    __Pyx__CyFunction_SetClassObj((__pyx_CyFunctionObject *) (f), (classobj))

#define __Pyx_CyFunction_Defaults(type, f) \
    ((type *)(((__pyx_CyFunctionObject *) (f))->defaults))
#define __Pyx_CyFunction_SetDefaultsGetter(f, g) \
    ((__pyx_CyFunctionObject *) (f))->defaults_getter = (g)


typedef struct {
#if CYTHON_COMPILING_IN_LIMITED_API
    PyObject_HEAD
    // We can't "inherit" from func, but we can use it as a data store
    PyObject *func;
#elif PY_VERSION_HEX < 0x030900B1
    PyCFunctionObject func;
#else
    // PEP-573: PyCFunctionObject + mm_class
    PyCMethodObject func;
#endif
#if CYTHON_BACKPORT_VECTORCALL
    __pyx_vectorcallfunc func_vectorcall;
#endif
#if PY_VERSION_HEX < 0x030500A0 || CYTHON_COMPILING_IN_LIMITED_API
    PyObject *func_weakreflist;
#endif
    PyObject *func_dict;
    PyObject *func_name;
    PyObject *func_qualname;
    PyObject *func_doc;
    PyObject *func_globals;
    PyObject *func_code;
    PyObject *func_closure;
#if PY_VERSION_HEX < 0x030900B1 || CYTHON_COMPILING_IN_LIMITED_API
    // No-args super() class cell
    PyObject *func_classobj;
#endif
    // Dynamic default args and annotations
    void *defaults;
    int defaults_pyobjects;
    size_t defaults_size;  /* used by FusedFunction for copying defaults */
    int flags;

    // Defaults info
    PyObject *defaults_tuple;   /* Const defaults tuple */
    PyObject *defaults_kwdict;  /* Const kwonly defaults dict */
    PyObject *(*defaults_getter)(PyObject *);
    PyObject *func_annotations; /* function annotations dict */

    // Coroutine marker
    PyObject *func_is_coroutine;
} __pyx_CyFunctionObject;

#undef __Pyx_CyOrPyCFunction_Check
#define __Pyx_CyFunction_Check(obj)  __Pyx_TypeCheck(obj, __pyx_CyFunctionType)
#define __Pyx_CyOrPyCFunction_Check(obj)  __Pyx_TypeCheck2(obj, __pyx_CyFunctionType, &PyCFunction_Type)
#define __Pyx_CyFunction_CheckExact(obj)  __Pyx_IS_TYPE(obj, __pyx_CyFunctionType)
static CYTHON_INLINE int __Pyx__IsSameCyOrCFunction(PyObject *func, void *cfunc);/*proto*/
#undef __Pyx_IsSameCFunction
#define __Pyx_IsSameCFunction(func, cfunc)   __Pyx__IsSameCyOrCFunction(func, cfunc)

static PyObject *__Pyx_CyFunction_Init(__pyx_CyFunctionObject* op, PyMethodDef *ml,
                                      int flags, PyObject* qualname,
                                      PyObject *closure,
                                      PyObject *module, PyObject *globals,
                                      PyObject* code);

static CYTHON_INLINE void __Pyx__CyFunction_SetClassObj(__pyx_CyFunctionObject* f, PyObject* classobj);
static CYTHON_INLINE void *__Pyx_CyFunction_InitDefaults(PyObject *m,
                                                         size_t size,
                                                         int pyobjects);
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsTuple(PyObject *m,
                                                            PyObject *tuple);
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsKwDict(PyObject *m,
                                                             PyObject *dict);
static CYTHON_INLINE void __Pyx_CyFunction_SetAnnotationsDict(PyObject *m,
                                                              PyObject *dict);


static int __pyx_CyFunction_init(PyObject *module);

#if CYTHON_METH_FASTCALL
static PyObject * __Pyx_CyFunction_Vectorcall_NOARGS(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames);
static PyObject * __Pyx_CyFunction_Vectorcall_O(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames);
static PyObject * __Pyx_CyFunction_Vectorcall_FASTCALL_KEYWORDS(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames);
static PyObject * __Pyx_CyFunction_Vectorcall_FASTCALL_KEYWORDS_METHOD(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames);
#if CYTHON_BACKPORT_VECTORCALL
#define __Pyx_CyFunction_func_vectorcall(f) (((__pyx_CyFunctionObject*)f)->func_vectorcall)
#else
#define __Pyx_CyFunction_func_vectorcall(f) (((PyCFunctionObject*)f)->vectorcall)
#endif
#endif

//////////////////// CythonFunctionShared ////////////////////
//@substitute: naming
//@requires: CommonStructures.c::FetchCommonType
//@requires: ObjectHandling.c::PyMethodNew
//@requires: ObjectHandling.c::PyVectorcallFastCallDict
//@requires: ModuleSetupCode.c::IncludeStructmemberH
//@requires: ObjectHandling.c::PyObjectGetAttrStr

#if CYTHON_COMPILING_IN_LIMITED_API
static CYTHON_INLINE int __Pyx__IsSameCyOrCFunction(PyObject *func, void *cfunc) {
    if (__Pyx_CyFunction_Check(func)) {
        return PyCFunction_GetFunction(((__pyx_CyFunctionObject*)func)->func) == (PyCFunction) cfunc;
    } else if (PyCFunction_Check(func)) {
        return PyCFunction_GetFunction(func) == (PyCFunction) cfunc;
    }
    return 0;
}
#else
static CYTHON_INLINE int __Pyx__IsSameCyOrCFunction(PyObject *func, void *cfunc) {
    return __Pyx_CyOrPyCFunction_Check(func) && __Pyx_CyOrPyCFunction_GET_FUNCTION(func) == (PyCFunction) cfunc;
}
#endif

static CYTHON_INLINE void __Pyx__CyFunction_SetClassObj(__pyx_CyFunctionObject* f, PyObject* classobj) {
#if PY_VERSION_HEX < 0x030900B1 || CYTHON_COMPILING_IN_LIMITED_API
    __Pyx_Py_XDECREF_SET(
        __Pyx_CyFunction_GetClassObj(f),
            ((classobj) ? __Pyx_NewRef(classobj) : NULL));
#else
    __Pyx_Py_XDECREF_SET(
        // assigning to "mm_class", which is a "PyTypeObject*"
        ((PyCMethodObject *) (f))->mm_class,
        (PyTypeObject*)((classobj) ? __Pyx_NewRef(classobj) : NULL));
#endif
}

static PyObject *
__Pyx_CyFunction_get_doc(__pyx_CyFunctionObject *op, void *closure)
{
    CYTHON_UNUSED_VAR(closure);
    if (unlikely(op->func_doc == NULL)) {
#if CYTHON_COMPILING_IN_LIMITED_API
        op->func_doc = PyObject_GetAttrString(op->func, "__doc__");
        if (unlikely(!op->func_doc)) return NULL;
#else
        if (((PyCFunctionObject*)op)->m_ml->ml_doc) {
#if PY_MAJOR_VERSION >= 3
            op->func_doc = PyUnicode_FromString(((PyCFunctionObject*)op)->m_ml->ml_doc);
#else
            op->func_doc = PyString_FromString(((PyCFunctionObject*)op)->m_ml->ml_doc);
#endif
            if (unlikely(op->func_doc == NULL))
                return NULL;
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
#endif   /* CYTHON_COMPILING_IN_LIMITED_API */
    }
    Py_INCREF(op->func_doc);
    return op->func_doc;
}

static int
__Pyx_CyFunction_set_doc(__pyx_CyFunctionObject *op, PyObject *value, void *context)
{
    CYTHON_UNUSED_VAR(context);
    if (value == NULL) {
        // Mark as deleted
        value = Py_None;
    }
    Py_INCREF(value);
    __Pyx_Py_XDECREF_SET(op->func_doc, value);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_name(__pyx_CyFunctionObject *op, void *context)
{
    CYTHON_UNUSED_VAR(context);
    if (unlikely(op->func_name == NULL)) {
#if CYTHON_COMPILING_IN_LIMITED_API
        op->func_name = PyObject_GetAttrString(op->func, "__name__");
#elif PY_MAJOR_VERSION >= 3
        op->func_name = PyUnicode_InternFromString(((PyCFunctionObject*)op)->m_ml->ml_name);
#else
        op->func_name = PyString_InternFromString(((PyCFunctionObject*)op)->m_ml->ml_name);
#endif  /* CYTHON_COMPILING_IN_LIMITED_API */
        if (unlikely(op->func_name == NULL))
            return NULL;
    }
    Py_INCREF(op->func_name);
    return op->func_name;
}

static int
__Pyx_CyFunction_set_name(__pyx_CyFunctionObject *op, PyObject *value, void *context)
{
    CYTHON_UNUSED_VAR(context);
#if PY_MAJOR_VERSION >= 3
    if (unlikely(value == NULL || !PyUnicode_Check(value)))
#else
    if (unlikely(value == NULL || !PyString_Check(value)))
#endif
    {
        PyErr_SetString(PyExc_TypeError,
                        "__name__ must be set to a string object");
        return -1;
    }
    Py_INCREF(value);
    __Pyx_Py_XDECREF_SET(op->func_name, value);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_qualname(__pyx_CyFunctionObject *op, void *context)
{
    CYTHON_UNUSED_VAR(context);
    Py_INCREF(op->func_qualname);
    return op->func_qualname;
}

static int
__Pyx_CyFunction_set_qualname(__pyx_CyFunctionObject *op, PyObject *value, void *context)
{
    CYTHON_UNUSED_VAR(context);
#if PY_MAJOR_VERSION >= 3
    if (unlikely(value == NULL || !PyUnicode_Check(value)))
#else
    if (unlikely(value == NULL || !PyString_Check(value)))
#endif
    {
        PyErr_SetString(PyExc_TypeError,
                        "__qualname__ must be set to a string object");
        return -1;
    }
    Py_INCREF(value);
    __Pyx_Py_XDECREF_SET(op->func_qualname, value);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_dict(__pyx_CyFunctionObject *op, void *context)
{
    CYTHON_UNUSED_VAR(context);
    if (unlikely(op->func_dict == NULL)) {
        op->func_dict = PyDict_New();
        if (unlikely(op->func_dict == NULL))
            return NULL;
    }
    Py_INCREF(op->func_dict);
    return op->func_dict;
}

static int
__Pyx_CyFunction_set_dict(__pyx_CyFunctionObject *op, PyObject *value, void *context)
{
    CYTHON_UNUSED_VAR(context);
    if (unlikely(value == NULL)) {
        PyErr_SetString(PyExc_TypeError,
               "function's dictionary may not be deleted");
        return -1;
    }
    if (unlikely(!PyDict_Check(value))) {
        PyErr_SetString(PyExc_TypeError,
               "setting function's dictionary to a non-dict");
        return -1;
    }
    Py_INCREF(value);
    __Pyx_Py_XDECREF_SET(op->func_dict, value);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_globals(__pyx_CyFunctionObject *op, void *context)
{
    CYTHON_UNUSED_VAR(context);
    Py_INCREF(op->func_globals);
    return op->func_globals;
}

static PyObject *
__Pyx_CyFunction_get_closure(__pyx_CyFunctionObject *op, void *context)
{
    CYTHON_UNUSED_VAR(op);
    CYTHON_UNUSED_VAR(context);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
__Pyx_CyFunction_get_code(__pyx_CyFunctionObject *op, void *context)
{
    PyObject* result = (op->func_code) ? op->func_code : Py_None;
    CYTHON_UNUSED_VAR(context);
    Py_INCREF(result);
    return result;
}

static int
__Pyx_CyFunction_init_defaults(__pyx_CyFunctionObject *op) {
    int result = 0;
    PyObject *res = op->defaults_getter((PyObject *) op);
    if (unlikely(!res))
        return -1;

    // Cache result
    #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    op->defaults_tuple = PyTuple_GET_ITEM(res, 0);
    Py_INCREF(op->defaults_tuple);
    op->defaults_kwdict = PyTuple_GET_ITEM(res, 1);
    Py_INCREF(op->defaults_kwdict);
    #else
    op->defaults_tuple = __Pyx_PySequence_ITEM(res, 0);
    if (unlikely(!op->defaults_tuple)) result = -1;
    else {
        op->defaults_kwdict = __Pyx_PySequence_ITEM(res, 1);
        if (unlikely(!op->defaults_kwdict)) result = -1;
    }
    #endif
    Py_DECREF(res);
    return result;
}

static int
__Pyx_CyFunction_set_defaults(__pyx_CyFunctionObject *op, PyObject* value, void *context) {
    CYTHON_UNUSED_VAR(context);
    if (!value) {
        // del => explicit None to prevent rebuilding
        value = Py_None;
    } else if (unlikely(value != Py_None && !PyTuple_Check(value))) {
        PyErr_SetString(PyExc_TypeError,
                        "__defaults__ must be set to a tuple object");
        return -1;
    }
    PyErr_WarnEx(PyExc_RuntimeWarning, "changes to cyfunction.__defaults__ will not "
                 "currently affect the values used in function calls", 1);
    Py_INCREF(value);
    __Pyx_Py_XDECREF_SET(op->defaults_tuple, value);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_defaults(__pyx_CyFunctionObject *op, void *context) {
    PyObject* result = op->defaults_tuple;
    CYTHON_UNUSED_VAR(context);
    if (unlikely(!result)) {
        if (op->defaults_getter) {
            if (unlikely(__Pyx_CyFunction_init_defaults(op) < 0)) return NULL;
            result = op->defaults_tuple;
        } else {
            result = Py_None;
        }
    }
    Py_INCREF(result);
    return result;
}

static int
__Pyx_CyFunction_set_kwdefaults(__pyx_CyFunctionObject *op, PyObject* value, void *context) {
    CYTHON_UNUSED_VAR(context);
    if (!value) {
        // del => explicit None to prevent rebuilding
        value = Py_None;
    } else if (unlikely(value != Py_None && !PyDict_Check(value))) {
        PyErr_SetString(PyExc_TypeError,
                        "__kwdefaults__ must be set to a dict object");
        return -1;
    }
    PyErr_WarnEx(PyExc_RuntimeWarning, "changes to cyfunction.__kwdefaults__ will not "
                 "currently affect the values used in function calls", 1);
    Py_INCREF(value);
    __Pyx_Py_XDECREF_SET(op->defaults_kwdict, value);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_kwdefaults(__pyx_CyFunctionObject *op, void *context) {
    PyObject* result = op->defaults_kwdict;
    CYTHON_UNUSED_VAR(context);
    if (unlikely(!result)) {
        if (op->defaults_getter) {
            if (unlikely(__Pyx_CyFunction_init_defaults(op) < 0)) return NULL;
            result = op->defaults_kwdict;
        } else {
            result = Py_None;
        }
    }
    Py_INCREF(result);
    return result;
}

static int
__Pyx_CyFunction_set_annotations(__pyx_CyFunctionObject *op, PyObject* value, void *context) {
    CYTHON_UNUSED_VAR(context);
    if (!value || value == Py_None) {
        value = NULL;
    } else if (unlikely(!PyDict_Check(value))) {
        PyErr_SetString(PyExc_TypeError,
                        "__annotations__ must be set to a dict object");
        return -1;
    }
    Py_XINCREF(value);
    __Pyx_Py_XDECREF_SET(op->func_annotations, value);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_annotations(__pyx_CyFunctionObject *op, void *context) {
    PyObject* result = op->func_annotations;
    CYTHON_UNUSED_VAR(context);
    if (unlikely(!result)) {
        result = PyDict_New();
        if (unlikely(!result)) return NULL;
        op->func_annotations = result;
    }
    Py_INCREF(result);
    return result;
}

static PyObject *
__Pyx_CyFunction_get_is_coroutine(__pyx_CyFunctionObject *op, void *context) {
    int is_coroutine;
    CYTHON_UNUSED_VAR(context);
    if (op->func_is_coroutine) {
        return __Pyx_NewRef(op->func_is_coroutine);
    }

    is_coroutine = op->flags & __Pyx_CYFUNCTION_COROUTINE;
#if PY_VERSION_HEX >= 0x03050000
    if (is_coroutine) {
        PyObject *module, *fromlist, *marker = PYIDENT("_is_coroutine");
        fromlist = PyList_New(1);
        if (unlikely(!fromlist)) return NULL;
        Py_INCREF(marker);
#if CYTHON_ASSUME_SAFE_MACROS
        PyList_SET_ITEM(fromlist, 0, marker);
#else
        if (unlikely(PyList_SetItem(fromlist, 0, marker) < 0)) {
            Py_DECREF(marker);
            Py_DECREF(fromlist);
            return NULL;
        }
#endif
        module = PyImport_ImportModuleLevelObject(PYIDENT("asyncio.coroutines"), NULL, NULL, fromlist, 0);
        Py_DECREF(fromlist);
        if (unlikely(!module)) goto ignore;
        op->func_is_coroutine = __Pyx_PyObject_GetAttrStr(module, marker);
        Py_DECREF(module);
        if (likely(op->func_is_coroutine)) {
            return __Pyx_NewRef(op->func_is_coroutine);
        }
ignore:
        PyErr_Clear();
    }
#endif

    op->func_is_coroutine = __Pyx_PyBool_FromLong(is_coroutine);
    return __Pyx_NewRef(op->func_is_coroutine);
}

//#if PY_VERSION_HEX >= 0x030400C1
//static PyObject *
//__Pyx_CyFunction_get_signature(__pyx_CyFunctionObject *op, void *context) {
//    PyObject *inspect_module, *signature_class, *signature;
//    CYTHON_UNUSED_VAR(context);
//    // from inspect import Signature
//    inspect_module = PyImport_ImportModuleLevelObject(PYIDENT("inspect"), NULL, NULL, NULL, 0);
//    if (unlikely(!inspect_module))
//        goto bad;
//    signature_class = __Pyx_PyObject_GetAttrStr(inspect_module, PYIDENT("Signature"));
//    Py_DECREF(inspect_module);
//    if (unlikely(!signature_class))
//        goto bad;
//    // return Signature.from_function(op)
//    signature = PyObject_CallMethodObjArgs(signature_class, PYIDENT("from_function"), op, NULL);
//    Py_DECREF(signature_class);
//    if (likely(signature))
//        return signature;
//bad:
//    // make sure we raise an AttributeError from this property on any errors
//    if (!PyErr_ExceptionMatches(PyExc_AttributeError))
//        PyErr_SetString(PyExc_AttributeError, "failed to calculate __signature__");
//    return NULL;
//}
//#endif

#if CYTHON_COMPILING_IN_LIMITED_API
static PyObject *
__Pyx_CyFunction_get_module(__pyx_CyFunctionObject *op, void *context) {
    CYTHON_UNUSED_VAR(context);
    return PyObject_GetAttrString(op->func, "__module__");
}

static int
__Pyx_CyFunction_set_module(__pyx_CyFunctionObject *op, PyObject* value, void *context) {
    CYTHON_UNUSED_VAR(context);
    return PyObject_SetAttrString(op->func, "__module__", value);
}
#endif

static PyGetSetDef __pyx_CyFunction_getsets[] = {
    {(char *) "func_doc", (getter)__Pyx_CyFunction_get_doc, (setter)__Pyx_CyFunction_set_doc, 0, 0},
    {(char *) "__doc__",  (getter)__Pyx_CyFunction_get_doc, (setter)__Pyx_CyFunction_set_doc, 0, 0},
    {(char *) "func_name", (getter)__Pyx_CyFunction_get_name, (setter)__Pyx_CyFunction_set_name, 0, 0},
    {(char *) "__name__", (getter)__Pyx_CyFunction_get_name, (setter)__Pyx_CyFunction_set_name, 0, 0},
    {(char *) "__qualname__", (getter)__Pyx_CyFunction_get_qualname, (setter)__Pyx_CyFunction_set_qualname, 0, 0},
    {(char *) "func_dict", (getter)__Pyx_CyFunction_get_dict, (setter)__Pyx_CyFunction_set_dict, 0, 0},
    {(char *) "__dict__", (getter)__Pyx_CyFunction_get_dict, (setter)__Pyx_CyFunction_set_dict, 0, 0},
    {(char *) "func_globals", (getter)__Pyx_CyFunction_get_globals, 0, 0, 0},
    {(char *) "__globals__", (getter)__Pyx_CyFunction_get_globals, 0, 0, 0},
    {(char *) "func_closure", (getter)__Pyx_CyFunction_get_closure, 0, 0, 0},
    {(char *) "__closure__", (getter)__Pyx_CyFunction_get_closure, 0, 0, 0},
    {(char *) "func_code", (getter)__Pyx_CyFunction_get_code, 0, 0, 0},
    {(char *) "__code__", (getter)__Pyx_CyFunction_get_code, 0, 0, 0},
    {(char *) "func_defaults", (getter)__Pyx_CyFunction_get_defaults, (setter)__Pyx_CyFunction_set_defaults, 0, 0},
    {(char *) "__defaults__", (getter)__Pyx_CyFunction_get_defaults, (setter)__Pyx_CyFunction_set_defaults, 0, 0},
    {(char *) "__kwdefaults__", (getter)__Pyx_CyFunction_get_kwdefaults, (setter)__Pyx_CyFunction_set_kwdefaults, 0, 0},
    {(char *) "__annotations__", (getter)__Pyx_CyFunction_get_annotations, (setter)__Pyx_CyFunction_set_annotations, 0, 0},
    {(char *) "_is_coroutine", (getter)__Pyx_CyFunction_get_is_coroutine, 0, 0, 0},
//#if PY_VERSION_HEX >= 0x030400C1
//    {(char *) "__signature__", (getter)__Pyx_CyFunction_get_signature, 0, 0, 0},
//#endif
#if CYTHON_COMPILING_IN_LIMITED_API
    {"__module__", (getter)__Pyx_CyFunction_get_module, (setter)__Pyx_CyFunction_set_module, 0, 0},
#endif
    {0, 0, 0, 0, 0}
};

static PyMemberDef __pyx_CyFunction_members[] = {
#if !CYTHON_COMPILING_IN_LIMITED_API
    {(char *) "__module__", T_OBJECT, offsetof(PyCFunctionObject, m_module), 0, 0},
#endif
#if CYTHON_USE_TYPE_SPECS
    {(char *) "__dictoffset__", T_PYSSIZET, offsetof(__pyx_CyFunctionObject, func_dict), READONLY, 0},
#if CYTHON_METH_FASTCALL
#if CYTHON_BACKPORT_VECTORCALL
    {(char *) "__vectorcalloffset__", T_PYSSIZET, offsetof(__pyx_CyFunctionObject, func_vectorcall), READONLY, 0},
#else
#if !CYTHON_COMPILING_IN_LIMITED_API
    {(char *) "__vectorcalloffset__", T_PYSSIZET, offsetof(PyCFunctionObject, vectorcall), READONLY, 0},
#endif
#endif
#endif
#if PY_VERSION_HEX < 0x030500A0 || CYTHON_COMPILING_IN_LIMITED_API
    {(char *) "__weaklistoffset__", T_PYSSIZET, offsetof(__pyx_CyFunctionObject, func_weakreflist), READONLY, 0},
#else
    {(char *) "__weaklistoffset__", T_PYSSIZET, offsetof(PyCFunctionObject, m_weakreflist), READONLY, 0},
#endif
#endif
    {0, 0, 0,  0, 0}
};

static PyObject *
__Pyx_CyFunction_reduce(__pyx_CyFunctionObject *m, PyObject *args)
{
    CYTHON_UNUSED_VAR(args);
#if PY_MAJOR_VERSION >= 3
    Py_INCREF(m->func_qualname);
    return m->func_qualname;
#else
    return PyString_FromString(((PyCFunctionObject*)m)->m_ml->ml_name);
#endif
}

static PyMethodDef __pyx_CyFunction_methods[] = {
    {"__reduce__", (PyCFunction)__Pyx_CyFunction_reduce, METH_VARARGS, 0},
    {0, 0, 0, 0}
};


#if PY_VERSION_HEX < 0x030500A0 || CYTHON_COMPILING_IN_LIMITED_API
#define __Pyx_CyFunction_weakreflist(cyfunc) ((cyfunc)->func_weakreflist)
#else
#define __Pyx_CyFunction_weakreflist(cyfunc) (((PyCFunctionObject*)cyfunc)->m_weakreflist)
#endif

static PyObject *__Pyx_CyFunction_Init(__pyx_CyFunctionObject *op, PyMethodDef *ml, int flags, PyObject* qualname,
                                       PyObject *closure, PyObject *module, PyObject* globals, PyObject* code) {
#if !CYTHON_COMPILING_IN_LIMITED_API
    PyCFunctionObject *cf = (PyCFunctionObject*) op;
#endif
    if (unlikely(op == NULL))
        return NULL;
#if CYTHON_COMPILING_IN_LIMITED_API
    // Note that we end up with a circular reference to op. This isn't
    // a disaster, but in an ideal world it'd be nice to avoid it.
    op->func = PyCFunction_NewEx(ml, (PyObject*)op, module);
    if (unlikely(!op->func)) return NULL;
#endif
    op->flags = flags;
    __Pyx_CyFunction_weakreflist(op) = NULL;
#if !CYTHON_COMPILING_IN_LIMITED_API
    cf->m_ml = ml;
    cf->m_self = (PyObject *) op;
#endif
    Py_XINCREF(closure);
    op->func_closure = closure;
#if !CYTHON_COMPILING_IN_LIMITED_API
    Py_XINCREF(module);
    cf->m_module = module;
#endif
    op->func_dict = NULL;
    op->func_name = NULL;
    Py_INCREF(qualname);
    op->func_qualname = qualname;
    op->func_doc = NULL;
#if PY_VERSION_HEX < 0x030900B1 || CYTHON_COMPILING_IN_LIMITED_API
    op->func_classobj = NULL;
#else
    ((PyCMethodObject*)op)->mm_class = NULL;
#endif
    op->func_globals = globals;
    Py_INCREF(op->func_globals);
    Py_XINCREF(code);
    op->func_code = code;
    // Dynamic Default args
    op->defaults_pyobjects = 0;
    op->defaults_size = 0;
    op->defaults = NULL;
    op->defaults_tuple = NULL;
    op->defaults_kwdict = NULL;
    op->defaults_getter = NULL;
    op->func_annotations = NULL;
    op->func_is_coroutine = NULL;
#if CYTHON_METH_FASTCALL
    switch (ml->ml_flags & (METH_VARARGS | METH_FASTCALL | METH_NOARGS | METH_O | METH_KEYWORDS | METH_METHOD)) {
    case METH_NOARGS:
        __Pyx_CyFunction_func_vectorcall(op) = __Pyx_CyFunction_Vectorcall_NOARGS;
        break;
    case METH_O:
        __Pyx_CyFunction_func_vectorcall(op) = __Pyx_CyFunction_Vectorcall_O;
        break;
    // case METH_FASTCALL is not used
    case METH_METHOD | METH_FASTCALL | METH_KEYWORDS:
        __Pyx_CyFunction_func_vectorcall(op) = __Pyx_CyFunction_Vectorcall_FASTCALL_KEYWORDS_METHOD;
        break;
    case METH_FASTCALL | METH_KEYWORDS:
        __Pyx_CyFunction_func_vectorcall(op) = __Pyx_CyFunction_Vectorcall_FASTCALL_KEYWORDS;
        break;
    // case METH_VARARGS is not used
    case METH_VARARGS | METH_KEYWORDS:
        __Pyx_CyFunction_func_vectorcall(op) = NULL;
        break;
    default:
        PyErr_SetString(PyExc_SystemError, "Bad call flags for CyFunction");
        Py_DECREF(op);
        return NULL;
    }
#endif
    return (PyObject *) op;
}

static int
__Pyx_CyFunction_clear(__pyx_CyFunctionObject *m)
{
    Py_CLEAR(m->func_closure);
#if CYTHON_COMPILING_IN_LIMITED_API
    Py_CLEAR(m->func);
#else
    Py_CLEAR(((PyCFunctionObject*)m)->m_module);
#endif
    Py_CLEAR(m->func_dict);
    Py_CLEAR(m->func_name);
    Py_CLEAR(m->func_qualname);
    Py_CLEAR(m->func_doc);
    Py_CLEAR(m->func_globals);
    Py_CLEAR(m->func_code);
#if !CYTHON_COMPILING_IN_LIMITED_API
#if PY_VERSION_HEX < 0x030900B1
    Py_CLEAR(__Pyx_CyFunction_GetClassObj(m));
#else
    {
        PyObject *cls = (PyObject*) ((PyCMethodObject *) (m))->mm_class;
        ((PyCMethodObject *) (m))->mm_class = NULL;
        Py_XDECREF(cls);
    }
#endif
#endif
    Py_CLEAR(m->defaults_tuple);
    Py_CLEAR(m->defaults_kwdict);
    Py_CLEAR(m->func_annotations);
    Py_CLEAR(m->func_is_coroutine);

    if (m->defaults) {
        PyObject **pydefaults = __Pyx_CyFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_XDECREF(pydefaults[i]);

        PyObject_Free(m->defaults);
        m->defaults = NULL;
    }

    return 0;
}

static void __Pyx__CyFunction_dealloc(__pyx_CyFunctionObject *m)
{
    if (__Pyx_CyFunction_weakreflist(m) != NULL)
        PyObject_ClearWeakRefs((PyObject *) m);
    __Pyx_CyFunction_clear(m);
    __Pyx_PyHeapTypeObject_GC_Del(m);
}

static void __Pyx_CyFunction_dealloc(__pyx_CyFunctionObject *m)
{
    PyObject_GC_UnTrack(m);
    __Pyx__CyFunction_dealloc(m);
}

static int __Pyx_CyFunction_traverse(__pyx_CyFunctionObject *m, visitproc visit, void *arg)
{
    Py_VISIT(m->func_closure);
#if CYTHON_COMPILING_IN_LIMITED_API
    Py_VISIT(m->func);
#else
    Py_VISIT(((PyCFunctionObject*)m)->m_module);
#endif
    Py_VISIT(m->func_dict);
    Py_VISIT(m->func_name);
    Py_VISIT(m->func_qualname);
    Py_VISIT(m->func_doc);
    Py_VISIT(m->func_globals);
    Py_VISIT(m->func_code);
#if !CYTHON_COMPILING_IN_LIMITED_API
    Py_VISIT(__Pyx_CyFunction_GetClassObj(m));
#endif
    Py_VISIT(m->defaults_tuple);
    Py_VISIT(m->defaults_kwdict);
    Py_VISIT(m->func_is_coroutine);

    if (m->defaults) {
        PyObject **pydefaults = __Pyx_CyFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_VISIT(pydefaults[i]);
    }

    return 0;
}

static PyObject*
__Pyx_CyFunction_repr(__pyx_CyFunctionObject *op)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("<cyfunction %U at %p>",
                                op->func_qualname, (void *)op);
#else
    return PyString_FromFormat("<cyfunction %s at %p>",
                               PyString_AsString(op->func_qualname), (void *)op);
#endif
}

static PyObject * __Pyx_CyFunction_CallMethod(PyObject *func, PyObject *self, PyObject *arg, PyObject *kw) {
    // originally copied from PyCFunction_Call() in CPython's Objects/methodobject.c
#if CYTHON_COMPILING_IN_LIMITED_API
    PyObject *f = ((__pyx_CyFunctionObject*)func)->func;
    PyObject *py_name = NULL;
    PyCFunction meth;
    int flags;
    meth = PyCFunction_GetFunction(f);
    if (unlikely(!meth)) return NULL;
    flags = PyCFunction_GetFlags(f);
    if (unlikely(flags < 0)) return NULL;
#else
    PyCFunctionObject* f = (PyCFunctionObject*)func;
    PyCFunction meth = f->m_ml->ml_meth;
    int flags = f->m_ml->ml_flags;
#endif

    Py_ssize_t size;

    switch (flags & (METH_VARARGS | METH_KEYWORDS | METH_NOARGS | METH_O)) {
    case METH_VARARGS:
        if (likely(kw == NULL || PyDict_Size(kw) == 0))
            return (*meth)(self, arg);
        break;
    case METH_VARARGS | METH_KEYWORDS:
        return (*(PyCFunctionWithKeywords)(void*)meth)(self, arg, kw);
    case METH_NOARGS:
        if (likely(kw == NULL || PyDict_Size(kw) == 0)) {
#if CYTHON_ASSUME_SAFE_MACROS
            size = PyTuple_GET_SIZE(arg);
#else
            size = PyTuple_Size(arg);
            if (unlikely(size < 0)) return NULL;
#endif
            if (likely(size == 0))
                return (*meth)(self, NULL);
#if CYTHON_COMPILING_IN_LIMITED_API
            py_name = __Pyx_CyFunction_get_name((__pyx_CyFunctionObject*)func, NULL);
            if (!py_name) return NULL;
            PyErr_Format(PyExc_TypeError,
                "%.200S() takes no arguments (%" CYTHON_FORMAT_SSIZE_T "d given)",
                py_name, size);
            Py_DECREF(py_name);
#else
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes no arguments (%" CYTHON_FORMAT_SSIZE_T "d given)",
                f->m_ml->ml_name, size);
#endif
            return NULL;
        }
        break;
    case METH_O:
        if (likely(kw == NULL || PyDict_Size(kw) == 0)) {
#if CYTHON_ASSUME_SAFE_MACROS
            size = PyTuple_GET_SIZE(arg);
#else
            size = PyTuple_Size(arg);
            if (unlikely(size < 0)) return NULL;
#endif
            if (likely(size == 1)) {
                PyObject *result, *arg0;
                #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
                arg0 = PyTuple_GET_ITEM(arg, 0);
                #else
                arg0 = __Pyx_PySequence_ITEM(arg, 0); if (unlikely(!arg0)) return NULL;
                #endif
                result = (*meth)(self, arg0);
                #if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
                Py_DECREF(arg0);
                #endif
                return result;
            }
#if CYTHON_COMPILING_IN_LIMITED_API
            py_name = __Pyx_CyFunction_get_name((__pyx_CyFunctionObject*)func, NULL);
            if (!py_name) return NULL;
            PyErr_Format(PyExc_TypeError,
                "%.200S() takes exactly one argument (%" CYTHON_FORMAT_SSIZE_T "d given)",
                py_name, size);
            Py_DECREF(py_name);
#else
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes exactly one argument (%" CYTHON_FORMAT_SSIZE_T "d given)",
                f->m_ml->ml_name, size);
#endif

            return NULL;
        }
        break;
    default:
        PyErr_SetString(PyExc_SystemError, "Bad call flags for CyFunction");
        return NULL;
    }
#if CYTHON_COMPILING_IN_LIMITED_API
    py_name = __Pyx_CyFunction_get_name((__pyx_CyFunctionObject*)func, NULL);
    if (!py_name) return NULL;
    PyErr_Format(PyExc_TypeError, "%.200S() takes no keyword arguments",
                 py_name);
    Py_DECREF(py_name);
#else
    PyErr_Format(PyExc_TypeError, "%.200s() takes no keyword arguments",
                 f->m_ml->ml_name);
#endif
    return NULL;
}

static CYTHON_INLINE PyObject *__Pyx_CyFunction_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    PyObject *self, *result;
#if CYTHON_COMPILING_IN_LIMITED_API
    // PyCFunction_GetSelf returns a borrowed reference
    self = PyCFunction_GetSelf(((__pyx_CyFunctionObject*)func)->func);
    if (unlikely(!self) && PyErr_Occurred()) return NULL;
#else
    self = ((PyCFunctionObject*)func)->m_self;
#endif
    result = __Pyx_CyFunction_CallMethod(func, self, arg, kw);
    return result;
}

static PyObject *__Pyx_CyFunction_CallAsMethod(PyObject *func, PyObject *args, PyObject *kw) {
    PyObject *result;
    __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *) func;

#if CYTHON_METH_FASTCALL
    // Prefer vectorcall if available. This is not the typical case, as
    // CPython would normally use vectorcall directly instead of tp_call.
     __pyx_vectorcallfunc vc = __Pyx_CyFunction_func_vectorcall(cyfunc);
    if (vc) {
#if CYTHON_ASSUME_SAFE_MACROS
        return __Pyx_PyVectorcall_FastCallDict(func, vc, &PyTuple_GET_ITEM(args, 0), (size_t)PyTuple_GET_SIZE(args), kw);
#else
        // avoid unused function warning
        (void) &__Pyx_PyVectorcall_FastCallDict;
        return PyVectorcall_Call(func, args, kw);
#endif
    }
#endif

    if ((cyfunc->flags & __Pyx_CYFUNCTION_CCLASS) && !(cyfunc->flags & __Pyx_CYFUNCTION_STATICMETHOD)) {
        Py_ssize_t argc;
        PyObject *new_args;
        PyObject *self;

#if CYTHON_ASSUME_SAFE_MACROS
        argc = PyTuple_GET_SIZE(args);
#else
        argc = PyTuple_Size(args);
        if (unlikely(!argc) < 0) return NULL;
#endif
        new_args = PyTuple_GetSlice(args, 1, argc);

        if (unlikely(!new_args))
            return NULL;

        self = PyTuple_GetItem(args, 0);
        if (unlikely(!self)) {
            Py_DECREF(new_args);
#if PY_MAJOR_VERSION > 2
            PyErr_Format(PyExc_TypeError,
                         "unbound method %.200S() needs an argument",
                         cyfunc->func_qualname);
#else
            // %S doesn't work in PyErr_Format on Py2 and replicating
            // the formatting seems more trouble than it's worth
            // (so produce a less useful error message).
            PyErr_SetString(PyExc_TypeError,
                            "unbound method needs an argument");
#endif
            return NULL;
        }

        result = __Pyx_CyFunction_CallMethod(func, self, new_args, kw);
        Py_DECREF(new_args);
    } else {
        result = __Pyx_CyFunction_Call(func, args, kw);
    }
    return result;
}

#if CYTHON_METH_FASTCALL
// Check that kwnames is empty (if you want to allow keyword arguments,
// simply pass kwnames=NULL) and figure out what to do with "self".
// Return value:
//  1: self = args[0]
//  0: self = cyfunc->func.m_self
// -1: error
static CYTHON_INLINE int __Pyx_CyFunction_Vectorcall_CheckArgs(__pyx_CyFunctionObject *cyfunc, Py_ssize_t nargs, PyObject *kwnames)
{
    int ret = 0;
    if ((cyfunc->flags & __Pyx_CYFUNCTION_CCLASS) && !(cyfunc->flags & __Pyx_CYFUNCTION_STATICMETHOD)) {
        if (unlikely(nargs < 1)) {
            PyErr_Format(PyExc_TypeError, "%.200s() needs an argument",
                         ((PyCFunctionObject*)cyfunc)->m_ml->ml_name);
            return -1;
        }
        ret = 1;
    }
    if (unlikely(kwnames) && unlikely(PyTuple_GET_SIZE(kwnames))) {
        PyErr_Format(PyExc_TypeError,
                     "%.200s() takes no keyword arguments", ((PyCFunctionObject*)cyfunc)->m_ml->ml_name);
        return -1;
    }
    return ret;
}

static PyObject * __Pyx_CyFunction_Vectorcall_NOARGS(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *)func;
    PyMethodDef* def = ((PyCFunctionObject*)cyfunc)->m_ml;
#if CYTHON_BACKPORT_VECTORCALL
    Py_ssize_t nargs = (Py_ssize_t)nargsf;
#else
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
#endif
    PyObject *self;
    switch (__Pyx_CyFunction_Vectorcall_CheckArgs(cyfunc, nargs, kwnames)) {
    case 1:
        self = args[0];
        args += 1;
        nargs -= 1;
        break;
    case 0:
        self = ((PyCFunctionObject*)cyfunc)->m_self;
        break;
    default:
        return NULL;
    }

    if (unlikely(nargs != 0)) {
        PyErr_Format(PyExc_TypeError,
            "%.200s() takes no arguments (%" CYTHON_FORMAT_SSIZE_T "d given)",
            def->ml_name, nargs);
        return NULL;
    }
    return def->ml_meth(self, NULL);
}

static PyObject * __Pyx_CyFunction_Vectorcall_O(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *)func;
    PyMethodDef* def = ((PyCFunctionObject*)cyfunc)->m_ml;
#if CYTHON_BACKPORT_VECTORCALL
    Py_ssize_t nargs = (Py_ssize_t)nargsf;
#else
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
#endif
    PyObject *self;
    switch (__Pyx_CyFunction_Vectorcall_CheckArgs(cyfunc, nargs, kwnames)) {
    case 1:
        self = args[0];
        args += 1;
        nargs -= 1;
        break;
    case 0:
        self = ((PyCFunctionObject*)cyfunc)->m_self;
        break;
    default:
        return NULL;
    }

    if (unlikely(nargs != 1)) {
        PyErr_Format(PyExc_TypeError,
            "%.200s() takes exactly one argument (%" CYTHON_FORMAT_SSIZE_T "d given)",
            def->ml_name, nargs);
        return NULL;
    }
    return def->ml_meth(self, args[0]);
}

static PyObject * __Pyx_CyFunction_Vectorcall_FASTCALL_KEYWORDS(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *)func;
    PyMethodDef* def = ((PyCFunctionObject*)cyfunc)->m_ml;
#if CYTHON_BACKPORT_VECTORCALL
    Py_ssize_t nargs = (Py_ssize_t)nargsf;
#else
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
#endif
    PyObject *self;
    switch (__Pyx_CyFunction_Vectorcall_CheckArgs(cyfunc, nargs, NULL)) {
    case 1:
        self = args[0];
        args += 1;
        nargs -= 1;
        break;
    case 0:
        self = ((PyCFunctionObject*)cyfunc)->m_self;
        break;
    default:
        return NULL;
    }

    return ((__Pyx_PyCFunctionFastWithKeywords)(void(*)(void))def->ml_meth)(self, args, nargs, kwnames);
}

static PyObject * __Pyx_CyFunction_Vectorcall_FASTCALL_KEYWORDS_METHOD(PyObject *func, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *)func;
    PyMethodDef* def = ((PyCFunctionObject*)cyfunc)->m_ml;
    PyTypeObject *cls = (PyTypeObject *) __Pyx_CyFunction_GetClassObj(cyfunc);
#if CYTHON_BACKPORT_VECTORCALL
    Py_ssize_t nargs = (Py_ssize_t)nargsf;
#else
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
#endif
    PyObject *self;
    switch (__Pyx_CyFunction_Vectorcall_CheckArgs(cyfunc, nargs, NULL)) {
    case 1:
        self = args[0];
        args += 1;
        nargs -= 1;
        break;
    case 0:
        self = ((PyCFunctionObject*)cyfunc)->m_self;
        break;
    default:
        return NULL;
    }

    return ((__Pyx_PyCMethod)(void(*)(void))def->ml_meth)(self, cls, args, (size_t)nargs, kwnames);
}
#endif

#if CYTHON_USE_TYPE_SPECS
static PyType_Slot __pyx_CyFunctionType_slots[] = {
    {Py_tp_dealloc, (void *)__Pyx_CyFunction_dealloc},
    {Py_tp_repr, (void *)__Pyx_CyFunction_repr},
    {Py_tp_call, (void *)__Pyx_CyFunction_CallAsMethod},
    {Py_tp_traverse, (void *)__Pyx_CyFunction_traverse},
    {Py_tp_clear, (void *)__Pyx_CyFunction_clear},
    {Py_tp_methods, (void *)__pyx_CyFunction_methods},
    {Py_tp_members, (void *)__pyx_CyFunction_members},
    {Py_tp_getset, (void *)__pyx_CyFunction_getsets},
    {Py_tp_descr_get, (void *)__Pyx_PyMethod_New},
    {0, 0},
};

static PyType_Spec __pyx_CyFunctionType_spec = {
    __PYX_TYPE_MODULE_PREFIX "cython_function_or_method",
    sizeof(__pyx_CyFunctionObject),
    0,
#ifdef Py_TPFLAGS_METHOD_DESCRIPTOR
    Py_TPFLAGS_METHOD_DESCRIPTOR |
#endif
#if (defined(_Py_TPFLAGS_HAVE_VECTORCALL) && CYTHON_METH_FASTCALL)
    _Py_TPFLAGS_HAVE_VECTORCALL |
#endif
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    __pyx_CyFunctionType_slots
};
#else /* CYTHON_USE_TYPE_SPECS */

static PyTypeObject __pyx_CyFunctionType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    __PYX_TYPE_MODULE_PREFIX "cython_function_or_method",  /*tp_name*/
    sizeof(__pyx_CyFunctionObject),   /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __Pyx_CyFunction_dealloc, /*tp_dealloc*/
#if !CYTHON_METH_FASTCALL
    0,                                  /*tp_print*/
#elif CYTHON_BACKPORT_VECTORCALL
    (printfunc)offsetof(__pyx_CyFunctionObject, func_vectorcall), /*tp_vectorcall_offset backported into tp_print*/
#else
    offsetof(PyCFunctionObject, vectorcall), /*tp_vectorcall_offset*/
#endif
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if PY_MAJOR_VERSION < 3
    0,                                  /*tp_compare*/
#else
    0,                                  /*tp_as_async*/
#endif
    (reprfunc) __Pyx_CyFunction_repr,   /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    __Pyx_CyFunction_CallAsMethod,      /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
#ifdef Py_TPFLAGS_METHOD_DESCRIPTOR
    Py_TPFLAGS_METHOD_DESCRIPTOR |
#endif
#if defined(_Py_TPFLAGS_HAVE_VECTORCALL) && CYTHON_METH_FASTCALL
    _Py_TPFLAGS_HAVE_VECTORCALL |
#endif
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) __Pyx_CyFunction_traverse,   /*tp_traverse*/
    (inquiry) __Pyx_CyFunction_clear,   /*tp_clear*/
    0,                                  /*tp_richcompare*/
#if PY_VERSION_HEX < 0x030500A0
    offsetof(__pyx_CyFunctionObject, func_weakreflist), /*tp_weaklistoffset*/
#else
    offsetof(PyCFunctionObject, m_weakreflist),         /*tp_weaklistoffset*/
#endif
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    __pyx_CyFunction_methods,           /*tp_methods*/
    __pyx_CyFunction_members,           /*tp_members*/
    __pyx_CyFunction_getsets,           /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    __Pyx_PyMethod_New,                 /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    offsetof(__pyx_CyFunctionObject, func_dict),/*tp_dictoffset*/
    0,                                  /*tp_init*/
    0,                                  /*tp_alloc*/
    0,                                  /*tp_new*/
    0,                                  /*tp_free*/
    0,                                  /*tp_is_gc*/
    0,                                  /*tp_bases*/
    0,                                  /*tp_mro*/
    0,                                  /*tp_cache*/
    0,                                  /*tp_subclasses*/
    0,                                  /*tp_weaklist*/
    0,                                  /*tp_del*/
    0,                                  /*tp_version_tag*/
#if PY_VERSION_HEX >= 0x030400a1
    0,                                  /*tp_finalize*/
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                  /*tp_vectorcall*/
#endif
#if __PYX_NEED_TP_PRINT_SLOT
    0,                                  /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030C0000
    0,                                  /*tp_watched*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000 && PY_VERSION_HEX < 0x030a0000
    0,                                          /*tp_pypy_flags*/
#endif
};
#endif  /* CYTHON_USE_TYPE_SPECS */


static int __pyx_CyFunction_init(PyObject *module) {
#if CYTHON_USE_TYPE_SPECS
    __pyx_CyFunctionType = __Pyx_FetchCommonTypeFromSpec(module, &__pyx_CyFunctionType_spec, NULL);
#else
    CYTHON_UNUSED_VAR(module);
    __pyx_CyFunctionType = __Pyx_FetchCommonType(&__pyx_CyFunctionType_type);
#endif
    if (unlikely(__pyx_CyFunctionType == NULL)) {
        return -1;
    }
    return 0;
}

static CYTHON_INLINE void *__Pyx_CyFunction_InitDefaults(PyObject *func, size_t size, int pyobjects) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;

    m->defaults = PyObject_Malloc(size);
    if (unlikely(!m->defaults))
        return PyErr_NoMemory();
    memset(m->defaults, 0, size);
    m->defaults_pyobjects = pyobjects;
    m->defaults_size = size;
    return m->defaults;
}

static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsTuple(PyObject *func, PyObject *tuple) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->defaults_tuple = tuple;
    Py_INCREF(tuple);
}

static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsKwDict(PyObject *func, PyObject *dict) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->defaults_kwdict = dict;
    Py_INCREF(dict);
}

static CYTHON_INLINE void __Pyx_CyFunction_SetAnnotationsDict(PyObject *func, PyObject *dict) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->func_annotations = dict;
    Py_INCREF(dict);
}


//////////////////// CythonFunction.proto ////////////////////

static PyObject *__Pyx_CyFunction_New(PyMethodDef *ml,
                                      int flags, PyObject* qualname,
                                      PyObject *closure,
                                      PyObject *module, PyObject *globals,
                                      PyObject* code);

//////////////////// CythonFunction ////////////////////
//@requires: CythonFunctionShared

static PyObject *__Pyx_CyFunction_New(PyMethodDef *ml, int flags, PyObject* qualname,
                                      PyObject *closure, PyObject *module, PyObject* globals, PyObject* code) {
    PyObject *op = __Pyx_CyFunction_Init(
        PyObject_GC_New(__pyx_CyFunctionObject, __pyx_CyFunctionType),
        ml, flags, qualname, closure, module, globals, code
    );
    if (likely(op)) {
        PyObject_GC_Track(op);
    }
    return op;
}

//////////////////// CyFunctionClassCell.proto ////////////////////
static int __Pyx_CyFunction_InitClassCell(PyObject *cyfunctions, PyObject *classobj);/*proto*/

//////////////////// CyFunctionClassCell ////////////////////
//@requires: CythonFunctionShared

static int __Pyx_CyFunction_InitClassCell(PyObject *cyfunctions, PyObject *classobj) {
    Py_ssize_t i, count = PyList_GET_SIZE(cyfunctions);

    for (i = 0; i < count; i++) {
        __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *)
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
            PyList_GET_ITEM(cyfunctions, i);
#else
            PySequence_ITEM(cyfunctions, i);
        if (unlikely(!m))
            return -1;
#endif
        __Pyx_CyFunction_SetClassObj(m, classobj);
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
        Py_DECREF((PyObject*)m);
#endif
    }
    return 0;
}


//////////////////// FusedFunction.proto ////////////////////

typedef struct {
    __pyx_CyFunctionObject func;
    PyObject *__signatures__;
    PyObject *self;
} __pyx_FusedFunctionObject;

static PyObject *__pyx_FusedFunction_New(PyMethodDef *ml, int flags,
                                         PyObject *qualname, PyObject *closure,
                                         PyObject *module, PyObject *globals,
                                         PyObject *code);

static int __pyx_FusedFunction_clear(__pyx_FusedFunctionObject *self);
static int __pyx_FusedFunction_init(PyObject *module);

#define __Pyx_FusedFunction_USED

//////////////////// FusedFunction ////////////////////
//@requires: CythonFunctionShared

static PyObject *
__pyx_FusedFunction_New(PyMethodDef *ml, int flags,
                        PyObject *qualname, PyObject *closure,
                        PyObject *module, PyObject *globals,
                        PyObject *code)
{
    PyObject *op = __Pyx_CyFunction_Init(
        // __pyx_CyFunctionObject is correct below since that's the cast that we want.
        PyObject_GC_New(__pyx_CyFunctionObject, __pyx_FusedFunctionType),
        ml, flags, qualname, closure, module, globals, code
    );
    if (likely(op)) {
        __pyx_FusedFunctionObject *fusedfunc = (__pyx_FusedFunctionObject *) op;
        fusedfunc->__signatures__ = NULL;
        fusedfunc->self = NULL;
        PyObject_GC_Track(op);
    }
    return op;
}

static void
__pyx_FusedFunction_dealloc(__pyx_FusedFunctionObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->self);
    Py_CLEAR(self->__signatures__);
    __Pyx__CyFunction_dealloc((__pyx_CyFunctionObject *) self);
}

static int
__pyx_FusedFunction_traverse(__pyx_FusedFunctionObject *self,
                             visitproc visit,
                             void *arg)
{
    Py_VISIT(self->self);
    Py_VISIT(self->__signatures__);
    return __Pyx_CyFunction_traverse((__pyx_CyFunctionObject *) self, visit, arg);
}

static int
__pyx_FusedFunction_clear(__pyx_FusedFunctionObject *self)
{
    Py_CLEAR(self->self);
    Py_CLEAR(self->__signatures__);
    return __Pyx_CyFunction_clear((__pyx_CyFunctionObject *) self);
}


static PyObject *
__pyx_FusedFunction_descr_get(PyObject *self, PyObject *obj, PyObject *type)
{
    __pyx_FusedFunctionObject *func, *meth;

    func = (__pyx_FusedFunctionObject *) self;

    if (func->self || func->func.flags & __Pyx_CYFUNCTION_STATICMETHOD) {
        // Do not allow rebinding and don't do anything for static methods
        Py_INCREF(self);
        return self;
    }

    if (obj == Py_None)
        obj = NULL;

    if (func->func.flags & __Pyx_CYFUNCTION_CLASSMETHOD)
        obj = type;

    if (obj == NULL) {
        // We aren't actually binding to anything, save the effort of rebinding
        Py_INCREF(self);
        return self;
    }

    meth = (__pyx_FusedFunctionObject *) __pyx_FusedFunction_New(
                    ((PyCFunctionObject *) func)->m_ml,
                    ((__pyx_CyFunctionObject *) func)->flags,
                    ((__pyx_CyFunctionObject *) func)->func_qualname,
                    ((__pyx_CyFunctionObject *) func)->func_closure,
                    ((PyCFunctionObject *) func)->m_module,
                    ((__pyx_CyFunctionObject *) func)->func_globals,
                    ((__pyx_CyFunctionObject *) func)->func_code);
    if (unlikely(!meth))
        return NULL;

    // defaults needs copying fully rather than just copying the pointer
    // since otherwise it will be freed on destruction of meth despite
    // belonging to func rather than meth
    if (func->func.defaults) {
        PyObject **pydefaults;
        int i;

        if (unlikely(!__Pyx_CyFunction_InitDefaults(
                (PyObject*)meth,
                func->func.defaults_size,
                func->func.defaults_pyobjects))) {
            Py_XDECREF((PyObject*)meth);
            return NULL;
        }
        memcpy(meth->func.defaults, func->func.defaults, func->func.defaults_size);

        pydefaults = __Pyx_CyFunction_Defaults(PyObject *, meth);
        for (i = 0; i < meth->func.defaults_pyobjects; i++)
            Py_XINCREF(pydefaults[i]);
    }

    __Pyx_CyFunction_SetClassObj(meth, __Pyx_CyFunction_GetClassObj(func));

    Py_XINCREF(func->__signatures__);
    meth->__signatures__ = func->__signatures__;

    Py_XINCREF(func->func.defaults_tuple);
    meth->func.defaults_tuple = func->func.defaults_tuple;

    Py_XINCREF(obj);
    meth->self = obj;

    return (PyObject *) meth;
}

static PyObject *
_obj_to_string(PyObject *obj)
{
    if (PyUnicode_CheckExact(obj))
        return __Pyx_NewRef(obj);
#if PY_MAJOR_VERSION == 2
    else if (PyString_Check(obj))
        return PyUnicode_FromEncodedObject(obj, NULL, "strict");
#endif
    else if (PyType_Check(obj))
        return PyObject_GetAttr(obj, PYIDENT("__name__"));
    else
        return PyObject_Unicode(obj);
}

static PyObject *
__pyx_FusedFunction_getitem(__pyx_FusedFunctionObject *self, PyObject *idx)
{
    PyObject *signature = NULL;
    PyObject *unbound_result_func;
    PyObject *result_func = NULL;

    if (unlikely(self->__signatures__ == NULL)) {
        PyErr_SetString(PyExc_TypeError, "Function is not fused");
        return NULL;
    }

    if (PyTuple_Check(idx)) {
        Py_ssize_t n = PyTuple_GET_SIZE(idx);
        PyObject *list = PyList_New(n);
        int i;

        if (unlikely(!list))
            return NULL;

        for (i = 0; i < n; i++) {
            PyObject *string;
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
            PyObject *item = PyTuple_GET_ITEM(idx, i);
#else
            PyObject *item = PySequence_ITEM(idx, i);  if (unlikely(!item)) goto __pyx_err;
#endif
            string = _obj_to_string(item);
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
            Py_DECREF(item);
#endif
            if (unlikely(!string)) goto __pyx_err;
            PyList_SET_ITEM(list, i, string);
        }

        signature = PyUnicode_Join(PYUNICODE("|"), list);
__pyx_err:;
        Py_DECREF(list);
    } else {
        signature = _obj_to_string(idx);
    }

    if (unlikely(!signature))
        return NULL;

    unbound_result_func = PyObject_GetItem(self->__signatures__, signature);

    if (likely(unbound_result_func)) {
        if (self->self) {
            __pyx_FusedFunctionObject *unbound = (__pyx_FusedFunctionObject *) unbound_result_func;

            // TODO: move this to InitClassCell
            __Pyx_CyFunction_SetClassObj(unbound, __Pyx_CyFunction_GetClassObj(self));

            result_func = __pyx_FusedFunction_descr_get(unbound_result_func,
                                                        self->self, self->self);
        } else {
            result_func = unbound_result_func;
            Py_INCREF(result_func);
        }
    }

    Py_DECREF(signature);
    Py_XDECREF(unbound_result_func);

    return result_func;
}

static PyObject *
__pyx_FusedFunction_callfunction(PyObject *func, PyObject *args, PyObject *kw)
{
     __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *) func;
    int static_specialized = (cyfunc->flags & __Pyx_CYFUNCTION_STATICMETHOD &&
                              !((__pyx_FusedFunctionObject *) func)->__signatures__);

    if ((cyfunc->flags & __Pyx_CYFUNCTION_CCLASS) && !static_specialized) {
        return __Pyx_CyFunction_CallAsMethod(func, args, kw);
    } else {
        return __Pyx_CyFunction_Call(func, args, kw);
    }
}

// Note: the 'self' from method binding is passed in in the args tuple,
//       whereas PyCFunctionObject's m_self is passed in as the first
//       argument to the C function. For extension methods we need
//       to pass 'self' as 'm_self' and not as the first element of the
//       args tuple.

static PyObject *
__pyx_FusedFunction_call(PyObject *func, PyObject *args, PyObject *kw)
{
    __pyx_FusedFunctionObject *binding_func = (__pyx_FusedFunctionObject *) func;
    Py_ssize_t argc = PyTuple_GET_SIZE(args);
    PyObject *new_args = NULL;
    __pyx_FusedFunctionObject *new_func = NULL;
    PyObject *result = NULL;
    int is_staticmethod = binding_func->func.flags & __Pyx_CYFUNCTION_STATICMETHOD;

    if (binding_func->self) {
        // Bound method call, put 'self' in the args tuple
        PyObject *self;
        Py_ssize_t i;
        new_args = PyTuple_New(argc + 1);
        if (unlikely(!new_args))
            return NULL;

        self = binding_func->self;

        Py_INCREF(self);
        PyTuple_SET_ITEM(new_args, 0, self);
        self = NULL;

        for (i = 0; i < argc; i++) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
            PyObject *item = PyTuple_GET_ITEM(args, i);
            Py_INCREF(item);
#else
            PyObject *item = PySequence_ITEM(args, i);  if (unlikely(!item)) goto bad;
#endif
            PyTuple_SET_ITEM(new_args, i + 1, item);
        }

        args = new_args;
    }

    if (binding_func->__signatures__) {
        PyObject *tup;
        if (is_staticmethod && binding_func->func.flags & __Pyx_CYFUNCTION_CCLASS) {
            // FIXME: this seems wrong, but we must currently pass the signatures dict as 'self' argument
            tup = PyTuple_Pack(3, args,
                               kw == NULL ? Py_None : kw,
                               binding_func->func.defaults_tuple);
            if (unlikely(!tup)) goto bad;
            new_func = (__pyx_FusedFunctionObject *) __Pyx_CyFunction_CallMethod(
                func, binding_func->__signatures__, tup, NULL);
        } else {
            tup = PyTuple_Pack(4, binding_func->__signatures__, args,
                               kw == NULL ? Py_None : kw,
                               binding_func->func.defaults_tuple);
            if (unlikely(!tup)) goto bad;
            new_func = (__pyx_FusedFunctionObject *) __pyx_FusedFunction_callfunction(func, tup, NULL);
        }
        Py_DECREF(tup);

        if (unlikely(!new_func))
            goto bad;

        __Pyx_CyFunction_SetClassObj(new_func, __Pyx_CyFunction_GetClassObj(binding_func));

        func = (PyObject *) new_func;
    }

    result = __pyx_FusedFunction_callfunction(func, args, kw);
bad:
    Py_XDECREF(new_args);
    Py_XDECREF((PyObject *) new_func);
    return result;
}

static PyMemberDef __pyx_FusedFunction_members[] = {
    {(char *) "__signatures__",
     T_OBJECT,
     offsetof(__pyx_FusedFunctionObject, __signatures__),
     READONLY,
     0},
    {(char *) "__self__", T_OBJECT_EX, offsetof(__pyx_FusedFunctionObject, self), READONLY, 0},
    {0, 0, 0, 0, 0},
};

static PyGetSetDef __pyx_FusedFunction_getsets[] = {
    // __doc__ is None for the fused function type, but we need it to be
    // a descriptor for the instance's __doc__, so rebuild the descriptor in our subclass
    // (all other descriptors are inherited)
    {(char *) "__doc__",  (getter)__Pyx_CyFunction_get_doc, (setter)__Pyx_CyFunction_set_doc, 0, 0},
    {0, 0, 0, 0, 0}
};

#if CYTHON_USE_TYPE_SPECS
static PyType_Slot __pyx_FusedFunctionType_slots[] = {
    {Py_tp_dealloc, (void *)__pyx_FusedFunction_dealloc},
    {Py_tp_call, (void *)__pyx_FusedFunction_call},
    {Py_tp_traverse, (void *)__pyx_FusedFunction_traverse},
    {Py_tp_clear, (void *)__pyx_FusedFunction_clear},
    {Py_tp_members, (void *)__pyx_FusedFunction_members},
    {Py_tp_getset, (void *)__pyx_FusedFunction_getsets},
    {Py_tp_descr_get, (void *)__pyx_FusedFunction_descr_get},
    {Py_mp_subscript, (void *)__pyx_FusedFunction_getitem},
    {0, 0},
};

static PyType_Spec __pyx_FusedFunctionType_spec = {
    __PYX_TYPE_MODULE_PREFIX "fused_cython_function",
    sizeof(__pyx_FusedFunctionObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    __pyx_FusedFunctionType_slots
};

#else  /* !CYTHON_USE_TYPE_SPECS */

static PyMappingMethods __pyx_FusedFunction_mapping_methods = {
    0,
    (binaryfunc) __pyx_FusedFunction_getitem,
    0,
};

static PyTypeObject __pyx_FusedFunctionType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    __PYX_TYPE_MODULE_PREFIX "fused_cython_function",  /*tp_name*/
    sizeof(__pyx_FusedFunctionObject), /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __pyx_FusedFunction_dealloc, /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if PY_MAJOR_VERSION < 3
    0,                                  /*tp_compare*/
#else
    0,                                  /*tp_as_async*/
#endif
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    &__pyx_FusedFunction_mapping_methods, /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    (ternaryfunc) __pyx_FusedFunction_call, /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) __pyx_FusedFunction_traverse,   /*tp_traverse*/
    (inquiry) __pyx_FusedFunction_clear,/*tp_clear*/
    0,                                  /*tp_richcompare*/
    0,                                  /*tp_weaklistoffset*/
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    0,                                  /*tp_methods*/
    __pyx_FusedFunction_members,        /*tp_members*/
    __pyx_FusedFunction_getsets,           /*tp_getset*/
    // NOTE: tp_base may be changed later during module initialisation when importing CyFunction across modules.
    &__pyx_CyFunctionType_type,         /*tp_base*/
    0,                                  /*tp_dict*/
    __pyx_FusedFunction_descr_get,      /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    0,                                  /*tp_init*/
    0,                                  /*tp_alloc*/
    0,                                  /*tp_new*/
    0,                                  /*tp_free*/
    0,                                  /*tp_is_gc*/
    0,                                  /*tp_bases*/
    0,                                  /*tp_mro*/
    0,                                  /*tp_cache*/
    0,                                  /*tp_subclasses*/
    0,                                  /*tp_weaklist*/
    0,                                  /*tp_del*/
    0,                                  /*tp_version_tag*/
#if PY_VERSION_HEX >= 0x030400a1
    0,                                  /*tp_finalize*/
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                  /*tp_vectorcall*/
#endif
#if __PYX_NEED_TP_PRINT_SLOT
    0,                                  /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030C0000
    0,                                  /*tp_watched*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000 && PY_VERSION_HEX < 0x030a0000
    0,                                          /*tp_pypy_flags*/
#endif
};
#endif

static int __pyx_FusedFunction_init(PyObject *module) {
#if CYTHON_USE_TYPE_SPECS
    PyObject *bases = PyTuple_Pack(1, __pyx_CyFunctionType);
    if (unlikely(!bases)) {
        return -1;
    }
    __pyx_FusedFunctionType = __Pyx_FetchCommonTypeFromSpec(module, &__pyx_FusedFunctionType_spec, bases);
    Py_DECREF(bases);
#else
    CYTHON_UNUSED_VAR(module);
    // Set base from __Pyx_FetchCommonTypeFromSpec, in case it's different from the local static value.
    __pyx_FusedFunctionType_type.tp_base = __pyx_CyFunctionType;
    __pyx_FusedFunctionType = __Pyx_FetchCommonType(&__pyx_FusedFunctionType_type);
#endif
    if (unlikely(__pyx_FusedFunctionType == NULL)) {
        return -1;
    }
    return 0;
}

//////////////////// ClassMethod.proto ////////////////////

#include "descrobject.h"
CYTHON_UNUSED static PyObject* __Pyx_Method_ClassMethod(PyObject *method); /*proto*/

//////////////////// ClassMethod ////////////////////

static PyObject* __Pyx_Method_ClassMethod(PyObject *method) {
#if CYTHON_COMPILING_IN_PYPY && PYPY_VERSION_NUM <= 0x05080000
    if (PyObject_TypeCheck(method, &PyWrapperDescr_Type)) {
        // cdef classes
        return PyClassMethod_New(method);
    }
#else
#if CYTHON_COMPILING_IN_PYPY
    // special C-API function only in PyPy >= 5.9
    if (PyMethodDescr_Check(method))
#else
    #if PY_MAJOR_VERSION == 2
    // PyMethodDescr_Type is not exposed in the CPython C-API in Py2.
    static PyTypeObject *methoddescr_type = NULL;
    if (unlikely(methoddescr_type == NULL)) {
       PyObject *meth = PyObject_GetAttrString((PyObject*)&PyList_Type, "append");
       if (unlikely(!meth)) return NULL;
       methoddescr_type = Py_TYPE(meth);
       Py_DECREF(meth);
    }
    #else
    PyTypeObject *methoddescr_type = &PyMethodDescr_Type;
    #endif
    if (__Pyx_TypeCheck(method, methoddescr_type))
#endif
    {
        // cdef classes
        PyMethodDescrObject *descr = (PyMethodDescrObject *)method;
        #if PY_VERSION_HEX < 0x03020000
        PyTypeObject *d_type = descr->d_type;
        #else
        PyTypeObject *d_type = descr->d_common.d_type;
        #endif
        return PyDescr_NewClassMethod(d_type, descr->d_method);
    }
#endif
    else if (PyMethod_Check(method)) {
        // python classes
        return PyClassMethod_New(PyMethod_GET_FUNCTION(method));
    }
    else {
        return PyClassMethod_New(method);
    }
}
