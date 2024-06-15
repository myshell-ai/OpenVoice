/////////////// ImportDottedModule.proto ///////////////

static PyObject *__Pyx_ImportDottedModule(PyObject *name, PyObject *parts_tuple); /*proto*/
#if PY_MAJOR_VERSION >= 3
static PyObject *__Pyx_ImportDottedModule_WalkParts(PyObject *module, PyObject *name, PyObject *parts_tuple); /*proto*/
#endif

/////////////// ImportDottedModule ///////////////
//@requires: Import

#if PY_MAJOR_VERSION >= 3
static PyObject *__Pyx__ImportDottedModule_Error(PyObject *name, PyObject *parts_tuple, Py_ssize_t count) {
    PyObject *partial_name = NULL, *slice = NULL, *sep = NULL;
    if (unlikely(PyErr_Occurred())) {
        PyErr_Clear();
    }
    if (likely(PyTuple_GET_SIZE(parts_tuple) == count)) {
        partial_name = name;
    } else {
        slice = PySequence_GetSlice(parts_tuple, 0, count);
        if (unlikely(!slice))
            goto bad;
        sep = PyUnicode_FromStringAndSize(".", 1);
        if (unlikely(!sep))
            goto bad;
        partial_name = PyUnicode_Join(sep, slice);
    }

    PyErr_Format(
#if PY_MAJOR_VERSION < 3
        PyExc_ImportError,
        "No module named '%s'", PyString_AS_STRING(partial_name));
#else
#if PY_VERSION_HEX >= 0x030600B1
        PyExc_ModuleNotFoundError,
#else
        PyExc_ImportError,
#endif
        "No module named '%U'", partial_name);
#endif

bad:
    Py_XDECREF(sep);
    Py_XDECREF(slice);
    Py_XDECREF(partial_name);
    return NULL;
}
#endif

#if PY_MAJOR_VERSION >= 3
static PyObject *__Pyx__ImportDottedModule_Lookup(PyObject *name) {
    PyObject *imported_module;
#if PY_VERSION_HEX < 0x030700A1 || (CYTHON_COMPILING_IN_PYPY && PYPY_VERSION_NUM  < 0x07030400)
    PyObject *modules = PyImport_GetModuleDict();
    if (unlikely(!modules))
        return NULL;
    imported_module = __Pyx_PyDict_GetItemStr(modules, name);
    Py_XINCREF(imported_module);
#else
    imported_module = PyImport_GetModule(name);
#endif
    return imported_module;
}
#endif

#if PY_MAJOR_VERSION >= 3
static PyObject *__Pyx_ImportDottedModule_WalkParts(PyObject *module, PyObject *name, PyObject *parts_tuple) {
    Py_ssize_t i, nparts;
    nparts = PyTuple_GET_SIZE(parts_tuple);
    for (i=1; i < nparts && module; i++) {
        PyObject *part, *submodule;
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        part = PyTuple_GET_ITEM(parts_tuple, i);
#else
        part = PySequence_ITEM(parts_tuple, i);
#endif
        submodule = __Pyx_PyObject_GetAttrStrNoError(module, part);
        // We stop if the attribute isn't found, i.e. if submodule is NULL here.
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
        Py_DECREF(part);
#endif
        Py_DECREF(module);
        module = submodule;
    }
    if (unlikely(!module)) {
        return __Pyx__ImportDottedModule_Error(name, parts_tuple, i);
    }
    return module;
}
#endif

static PyObject *__Pyx__ImportDottedModule(PyObject *name, PyObject *parts_tuple) {
#if PY_MAJOR_VERSION < 3
    PyObject *module, *from_list, *star = PYIDENT("*");
    CYTHON_UNUSED_VAR(parts_tuple);
    from_list = PyList_New(1);
    if (unlikely(!from_list))
        return NULL;
    Py_INCREF(star);
    PyList_SET_ITEM(from_list, 0, star);
    module = __Pyx_Import(name, from_list, 0);
    Py_DECREF(from_list);
    return module;
#else
    PyObject *imported_module;
    PyObject *module = __Pyx_Import(name, NULL, 0);
    if (!parts_tuple || unlikely(!module))
        return module;

    // Look up module in sys.modules, which is safer than the attribute lookups below.
    imported_module = __Pyx__ImportDottedModule_Lookup(name);
    if (likely(imported_module)) {
        Py_DECREF(module);
        return imported_module;
    }
    PyErr_Clear();
    return __Pyx_ImportDottedModule_WalkParts(module, name, parts_tuple);
#endif
}

static PyObject *__Pyx_ImportDottedModule(PyObject *name, PyObject *parts_tuple) {
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030400B1
    PyObject *module = __Pyx__ImportDottedModule_Lookup(name);
    if (likely(module)) {
        // CPython guards against thread-concurrent initialisation in importlib.
        // In this case, we let PyImport_ImportModuleLevelObject() handle the locking.
        PyObject *spec = __Pyx_PyObject_GetAttrStrNoError(module, PYIDENT("__spec__"));
        if (likely(spec)) {
            PyObject *unsafe = __Pyx_PyObject_GetAttrStrNoError(spec, PYIDENT("_initializing"));
            if (likely(!unsafe || !__Pyx_PyObject_IsTrue(unsafe))) {
                Py_DECREF(spec);
                spec = NULL;
            }
            Py_XDECREF(unsafe);
        }
        if (likely(!spec)) {
            // Not in initialisation phase => use modules as is.
            PyErr_Clear();
            return module;
        }
        Py_DECREF(spec);
        Py_DECREF(module);
    } else if (PyErr_Occurred()) {
        PyErr_Clear();
    }
#endif

    return __Pyx__ImportDottedModule(name, parts_tuple);
}


/////////////// ImportDottedModuleRelFirst.proto ///////////////

static PyObject *__Pyx_ImportDottedModuleRelFirst(PyObject *name, PyObject *parts_tuple); /*proto*/

/////////////// ImportDottedModuleRelFirst ///////////////
//@requires: ImportDottedModule
//@requires: Import

static PyObject *__Pyx_ImportDottedModuleRelFirst(PyObject *name, PyObject *parts_tuple) {
    PyObject *module;
    PyObject *from_list = NULL;
#if PY_MAJOR_VERSION < 3
    PyObject *star = PYIDENT("*");
    from_list = PyList_New(1);
    if (unlikely(!from_list))
        return NULL;
    Py_INCREF(star);
    PyList_SET_ITEM(from_list, 0, star);
#endif
    module = __Pyx_Import(name, from_list, -1);
    Py_XDECREF(from_list);
    if (module) {
        #if PY_MAJOR_VERSION >= 3
        if (parts_tuple) {
            module = __Pyx_ImportDottedModule_WalkParts(module, name, parts_tuple);
        }
        #endif
        return module;
    }
    if (unlikely(!PyErr_ExceptionMatches(PyExc_ImportError)))
        return NULL;
    PyErr_Clear();
    // try absolute import
    return __Pyx_ImportDottedModule(name, parts_tuple);
}


/////////////// Import.proto ///////////////

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level); /*proto*/

/////////////// Import ///////////////
//@requires: ObjectHandling.c::PyObjectGetAttrStr
//@requires:StringTools.c::IncludeStringH
//@substitute: naming

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level) {
    PyObject *module = 0;
    PyObject *empty_dict = 0;
    PyObject *empty_list = 0;
    #if PY_MAJOR_VERSION < 3
    PyObject *py_import;
    py_import = __Pyx_PyObject_GetAttrStr($builtins_cname, PYIDENT("__import__"));
    if (unlikely(!py_import))
        goto bad;
    if (!from_list) {
        empty_list = PyList_New(0);
        if (unlikely(!empty_list))
            goto bad;
        from_list = empty_list;
    }
    #endif
    empty_dict = PyDict_New();
    if (unlikely(!empty_dict))
        goto bad;
    {
        #if PY_MAJOR_VERSION >= 3
        if (level == -1) {
            if (strchr(__Pyx_MODULE_NAME, '.') != NULL) {
                /* try package relative import first */
                module = PyImport_ImportModuleLevelObject(
                    name, $moddict_cname, empty_dict, from_list, 1);
                if (unlikely(!module)) {
                    if (unlikely(!PyErr_ExceptionMatches(PyExc_ImportError)))
                        goto bad;
                    PyErr_Clear();
                }
            }
            level = 0; /* try absolute import on failure */
        }
        #endif
        if (!module) {
            #if PY_MAJOR_VERSION < 3
            PyObject *py_level = PyInt_FromLong(level);
            if (unlikely(!py_level))
                goto bad;
            module = PyObject_CallFunctionObjArgs(py_import,
                name, $moddict_cname, empty_dict, from_list, py_level, (PyObject *)NULL);
            Py_DECREF(py_level);
            #else
            module = PyImport_ImportModuleLevelObject(
                name, $moddict_cname, empty_dict, from_list, level);
            #endif
        }
    }
bad:
    Py_XDECREF(empty_dict);
    Py_XDECREF(empty_list);
    #if PY_MAJOR_VERSION < 3
    Py_XDECREF(py_import);
    #endif
    return module;
}


/////////////// ImportFrom.proto ///////////////

static PyObject* __Pyx_ImportFrom(PyObject* module, PyObject* name); /*proto*/

/////////////// ImportFrom ///////////////
//@requires: ObjectHandling.c::PyObjectGetAttrStr

static PyObject* __Pyx_ImportFrom(PyObject* module, PyObject* name) {
    PyObject* value = __Pyx_PyObject_GetAttrStr(module, name);
    if (unlikely(!value) && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        // 'name' may refer to a (sub-)module which has not finished initialization
        // yet, and may not be assigned as an attribute to its parent, so try
        // finding it by full name.
        const char* module_name_str = 0;
        PyObject* module_name = 0;
        PyObject* module_dot = 0;
        PyObject* full_name = 0;
        PyErr_Clear();
        module_name_str = PyModule_GetName(module);
        if (unlikely(!module_name_str)) { goto modbad; }
        module_name = PyUnicode_FromString(module_name_str);
        if (unlikely(!module_name)) { goto modbad; }
        module_dot = PyUnicode_Concat(module_name, PYUNICODE("."));
        if (unlikely(!module_dot)) { goto modbad; }
        full_name = PyUnicode_Concat(module_dot, name);
        if (unlikely(!full_name)) { goto modbad; }
        #if PY_VERSION_HEX < 0x030700A1 || (CYTHON_COMPILING_IN_PYPY && PYPY_VERSION_NUM  < 0x07030400)
        {
            PyObject *modules = PyImport_GetModuleDict();
            if (unlikely(!modules))
                goto modbad;
            value = PyObject_GetItem(modules, full_name);
        }
        #else
        value = PyImport_GetModule(full_name);
        #endif

      modbad:
        Py_XDECREF(full_name);
        Py_XDECREF(module_dot);
        Py_XDECREF(module_name);
    }
    if (unlikely(!value)) {
        PyErr_Format(PyExc_ImportError,
        #if PY_MAJOR_VERSION < 3
            "cannot import name %.230s", PyString_AS_STRING(name));
        #else
            "cannot import name %S", name);
        #endif
    }
    return value;
}


/////////////// ImportStar ///////////////
//@substitute: naming

/* import_all_from is an unexposed function from ceval.c */

static int
__Pyx_import_all_from(PyObject *locals, PyObject *v)
{
    PyObject *all = PyObject_GetAttrString(v, "__all__");
    PyObject *dict, *name, *value;
    int skip_leading_underscores = 0;
    int pos, err;

    if (all == NULL) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
            return -1; /* Unexpected error */
        PyErr_Clear();
        dict = PyObject_GetAttrString(v, "__dict__");
        if (dict == NULL) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError))
                return -1;
            PyErr_SetString(PyExc_ImportError,
            "from-import-* object has no __dict__ and no __all__");
            return -1;
        }
#if PY_MAJOR_VERSION < 3
        all = PyObject_CallMethod(dict, (char *)"keys", NULL);
#else
        all = PyMapping_Keys(dict);
#endif
        Py_DECREF(dict);
        if (all == NULL)
            return -1;
        skip_leading_underscores = 1;
    }

    for (pos = 0, err = 0; ; pos++) {
        name = PySequence_GetItem(all, pos);
        if (name == NULL) {
            if (!PyErr_ExceptionMatches(PyExc_IndexError))
                err = -1;
            else
                PyErr_Clear();
            break;
        }
        if (skip_leading_underscores &&
#if PY_MAJOR_VERSION < 3
            likely(PyString_Check(name)) &&
            PyString_AS_STRING(name)[0] == '_')
#else
            likely(PyUnicode_Check(name)) &&
            likely(__Pyx_PyUnicode_GET_LENGTH(name)) &&
            __Pyx_PyUnicode_READ_CHAR(name, 0) == '_')
#endif
        {
            Py_DECREF(name);
            continue;
        }
        value = PyObject_GetAttr(v, name);
        if (value == NULL)
            err = -1;
        else if (PyDict_CheckExact(locals))
            err = PyDict_SetItem(locals, name, value);
        else
            err = PyObject_SetItem(locals, name, value);
        Py_DECREF(name);
        Py_XDECREF(value);
        if (err != 0)
            break;
    }
    Py_DECREF(all);
    return err;
}


static int ${import_star}(PyObject* m) {

    int i;
    int ret = -1;
    char* s;
    PyObject *locals = 0;
    PyObject *list = 0;
#if PY_MAJOR_VERSION >= 3
    PyObject *utf8_name = 0;
#endif
    PyObject *name;
    PyObject *item;

    locals = PyDict_New();              if (!locals) goto bad;
    if (__Pyx_import_all_from(locals, m) < 0) goto bad;
    list = PyDict_Items(locals);        if (!list) goto bad;

    for(i=0; i<PyList_GET_SIZE(list); i++) {
        name = PyTuple_GET_ITEM(PyList_GET_ITEM(list, i), 0);
        item = PyTuple_GET_ITEM(PyList_GET_ITEM(list, i), 1);
#if PY_MAJOR_VERSION >= 3
        utf8_name = PyUnicode_AsUTF8String(name);
        if (!utf8_name) goto bad;
        s = PyBytes_AS_STRING(utf8_name);
        if (${import_star_set}(item, name, s) < 0) goto bad;
        Py_DECREF(utf8_name); utf8_name = 0;
#else
        s = PyString_AsString(name);
        if (!s) goto bad;
        if (${import_star_set}(item, name, s) < 0) goto bad;
#endif
    }
    ret = 0;

bad:
    Py_XDECREF(locals);
    Py_XDECREF(list);
#if PY_MAJOR_VERSION >= 3
    Py_XDECREF(utf8_name);
#endif
    return ret;
}


/////////////// SetPackagePathFromImportLib.proto ///////////////

// PY_VERSION_HEX >= 0x03030000
#if PY_MAJOR_VERSION >= 3 && !CYTHON_PEP489_MULTI_PHASE_INIT
static int __Pyx_SetPackagePathFromImportLib(PyObject *module_name);
#else
#define __Pyx_SetPackagePathFromImportLib(a) 0
#endif

/////////////// SetPackagePathFromImportLib ///////////////
//@substitute: naming

// PY_VERSION_HEX >= 0x03030000
#if PY_MAJOR_VERSION >= 3 && !CYTHON_PEP489_MULTI_PHASE_INIT
static int __Pyx_SetPackagePathFromImportLib(PyObject *module_name) {
    PyObject *importlib, *osmod, *ossep, *parts, *package_path;
    PyObject *file_path = NULL;
    int result;
    PyObject *spec;
    // package_path = [importlib.util.find_spec(module_name).origin.rsplit(os.sep, 1)[0]]
    importlib = PyImport_ImportModule("importlib.util");
    if (unlikely(!importlib))
        goto bad;
    spec = PyObject_CallMethod(importlib, "find_spec", "(O)", module_name);
    Py_DECREF(importlib);
    if (unlikely(!spec))
        goto bad;
    file_path = PyObject_GetAttrString(spec, "origin");
    Py_DECREF(spec);
    if (unlikely(!file_path))
        goto bad;

    if (unlikely(PyObject_SetAttrString($module_cname, "__file__", file_path) < 0))
        goto bad;

    osmod = PyImport_ImportModule("os");
    if (unlikely(!osmod))
        goto bad;
    ossep = PyObject_GetAttrString(osmod, "sep");
    Py_DECREF(osmod);
    if (unlikely(!ossep))
        goto bad;
    parts = PyObject_CallMethod(file_path, "rsplit", "(Oi)", ossep, 1);
    Py_DECREF(file_path); file_path = NULL;
    Py_DECREF(ossep);
    if (unlikely(!parts))
        goto bad;
    package_path = Py_BuildValue("[O]", PyList_GET_ITEM(parts, 0));
    Py_DECREF(parts);
    if (unlikely(!package_path))
        goto bad;
    goto set_path;

bad:
    PyErr_WriteUnraisable(module_name);
    Py_XDECREF(file_path);

    // set an empty path list on failure
    PyErr_Clear();
    package_path = PyList_New(0);
    if (unlikely(!package_path))
        return -1;

set_path:
    result = PyObject_SetAttrString($module_cname, "__path__", package_path);
    Py_DECREF(package_path);
    return result;
}
#endif


/////////////// TypeImport.proto ///////////////
//@substitute: naming

#ifndef __PYX_HAVE_RT_ImportType_proto_$cyversion
#define __PYX_HAVE_RT_ImportType_proto_$cyversion

#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdalign.h>
#endif

#if (defined (__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || __cplusplus >= 201103L
#define __PYX_GET_STRUCT_ALIGNMENT_$cyversion(s) alignof(s)
#else
// best guess at what the alignment could be since we can't measure it
#define __PYX_GET_STRUCT_ALIGNMENT_$cyversion(s) sizeof(void*)
#endif

enum __Pyx_ImportType_CheckSize_$cyversion {
   __Pyx_ImportType_CheckSize_Error_$cyversion = 0,
   __Pyx_ImportType_CheckSize_Warn_$cyversion = 1,
   __Pyx_ImportType_CheckSize_Ignore_$cyversion = 2
};

static PyTypeObject *__Pyx_ImportType_$cyversion(PyObject* module, const char *module_name, const char *class_name, size_t size, size_t alignment, enum __Pyx_ImportType_CheckSize_$cyversion check_size);  /*proto*/

#endif

/////////////// TypeImport ///////////////
//@substitute: naming

#ifndef __PYX_HAVE_RT_ImportType_$cyversion
#define __PYX_HAVE_RT_ImportType_$cyversion
static PyTypeObject *__Pyx_ImportType_$cyversion(PyObject *module, const char *module_name, const char *class_name,
    size_t size, size_t alignment, enum __Pyx_ImportType_CheckSize_$cyversion check_size)
{
    PyObject *result = 0;
    char warning[200];
    Py_ssize_t basicsize;
    Py_ssize_t itemsize;
#if CYTHON_COMPILING_IN_LIMITED_API
    PyObject *py_basicsize;
    PyObject *py_itemsize;
#endif

    result = PyObject_GetAttrString(module, class_name);
    if (!result)
        goto bad;
    if (!PyType_Check(result)) {
        PyErr_Format(PyExc_TypeError,
            "%.200s.%.200s is not a type object",
            module_name, class_name);
        goto bad;
    }
#if !CYTHON_COMPILING_IN_LIMITED_API
    basicsize = ((PyTypeObject *)result)->tp_basicsize;
    itemsize = ((PyTypeObject *)result)->tp_itemsize;
#else
    py_basicsize = PyObject_GetAttrString(result, "__basicsize__");
    if (!py_basicsize)
        goto bad;
    basicsize = PyLong_AsSsize_t(py_basicsize);
    Py_DECREF(py_basicsize);
    py_basicsize = 0;
    if (basicsize == (Py_ssize_t)-1 && PyErr_Occurred())
        goto bad;
    py_itemsize = PyObject_GetAttrString(result, "__itemsize__");
    if (!py_itemsize)
        goto bad;
    itemsize = PyLong_AsSsize_t(py_itemsize);
    Py_DECREF(py_itemsize);
    py_itemsize = 0;
    if (itemsize == (Py_ssize_t)-1 && PyErr_Occurred())
        goto bad;
#endif
    if (itemsize) {
        // If itemsize is smaller than the alignment the struct can end up with some extra
        // padding at the end. In this case we need to work out the maximum size that
        // the padding could be when calculating the range of valid struct sizes.
        if (size % alignment) {
            // if this is true we've probably calculated the alignment wrongly
            // (most likely because alignof isn't available)
            alignment = size % alignment;
        }
        if (itemsize < (Py_ssize_t)alignment)
            itemsize = (Py_ssize_t)alignment;
    }
    if ((size_t)(basicsize + itemsize) < size) {
        PyErr_Format(PyExc_ValueError,
            "%.200s.%.200s size changed, may indicate binary incompatibility. "
            "Expected %zd from C header, got %zd from PyObject",
            module_name, class_name, size, basicsize+itemsize);
        goto bad;
    }
    // varobjects almost have structs  between basicsize and basicsize + itemsize
    // but the struct isn't always one of the two limiting values
    if (check_size == __Pyx_ImportType_CheckSize_Error_$cyversion &&
            ((size_t)basicsize > size || (size_t)(basicsize + itemsize) < size)) {
        PyErr_Format(PyExc_ValueError,
            "%.200s.%.200s size changed, may indicate binary incompatibility. "
            "Expected %zd from C header, got %zd-%zd from PyObject",
            module_name, class_name, size, basicsize, basicsize+itemsize);
        goto bad;
    }
    else if (check_size == __Pyx_ImportType_CheckSize_Warn_$cyversion && (size_t)basicsize > size) {
        PyOS_snprintf(warning, sizeof(warning),
            "%s.%s size changed, may indicate binary incompatibility. "
            "Expected %zd from C header, got %zd from PyObject",
            module_name, class_name, size, basicsize);
        if (PyErr_WarnEx(NULL, warning, 0) < 0) goto bad;
    }
    /* check_size == __Pyx_ImportType_CheckSize_Ignore does not warn nor error */
    return (PyTypeObject *)result;
bad:
    Py_XDECREF(result);
    return NULL;
}
#endif

/////////////// FunctionImport.proto ///////////////
//@substitute: naming

static int __Pyx_ImportFunction_$cyversion(PyObject *module, const char *funcname, void (**f)(void), const char *sig); /*proto*/

/////////////// FunctionImport ///////////////
//@substitute: naming

#ifndef __PYX_HAVE_RT_ImportFunction_$cyversion
#define __PYX_HAVE_RT_ImportFunction_$cyversion
static int __Pyx_ImportFunction_$cyversion(PyObject *module, const char *funcname, void (**f)(void), const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;
    union {
        void (*fp)(void);
        void *p;
    } tmp;

    d = PyObject_GetAttrString(module, (char *)"$api_name");
    if (!d)
        goto bad;
    cobj = PyDict_GetItemString(d, funcname);
    if (!cobj) {
        PyErr_Format(PyExc_ImportError,
            "%.200s does not export expected C function %.200s",
                PyModule_GetName(module), funcname);
        goto bad;
    }
    if (!PyCapsule_IsValid(cobj, sig)) {
        PyErr_Format(PyExc_TypeError,
            "C function %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), funcname, sig, PyCapsule_GetName(cobj));
        goto bad;
    }
    tmp.p = PyCapsule_GetPointer(cobj, sig);
    *f = tmp.fp;
    if (!(*f))
        goto bad;
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(d);
    return -1;
}
#endif

/////////////// FunctionExport.proto ///////////////

static int __Pyx_ExportFunction(const char *name, void (*f)(void), const char *sig); /*proto*/

/////////////// FunctionExport ///////////////
//@substitute: naming

static int __Pyx_ExportFunction(const char *name, void (*f)(void), const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;
    union {
        void (*fp)(void);
        void *p;
    } tmp;

    d = PyObject_GetAttrString($module_cname, (char *)"$api_name");
    if (!d) {
        PyErr_Clear();
        d = PyDict_New();
        if (!d)
            goto bad;
        Py_INCREF(d);
        if (PyModule_AddObject($module_cname, (char *)"$api_name", d) < 0)
            goto bad;
    }
    tmp.fp = f;
    cobj = PyCapsule_New(tmp.p, sig, 0);
    if (!cobj)
        goto bad;
    if (PyDict_SetItemString(d, name, cobj) < 0)
        goto bad;
    Py_DECREF(cobj);
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(cobj);
    Py_XDECREF(d);
    return -1;
}

/////////////// VoidPtrImport.proto ///////////////
//@substitute: naming

static int __Pyx_ImportVoidPtr_$cyversion(PyObject *module, const char *name, void **p, const char *sig); /*proto*/

/////////////// VoidPtrImport ///////////////
//@substitute: naming

#ifndef __PYX_HAVE_RT_ImportVoidPtr_$cyversion
#define __PYX_HAVE_RT_ImportVoidPtr_$cyversion
static int __Pyx_ImportVoidPtr_$cyversion(PyObject *module, const char *name, void **p, const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;

    d = PyObject_GetAttrString(module, (char *)"$api_name");
    if (!d)
        goto bad;
    cobj = PyDict_GetItemString(d, name);
    if (!cobj) {
        PyErr_Format(PyExc_ImportError,
            "%.200s does not export expected C variable %.200s",
                PyModule_GetName(module), name);
        goto bad;
    }
    if (!PyCapsule_IsValid(cobj, sig)) {
        PyErr_Format(PyExc_TypeError,
            "C variable %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), name, sig, PyCapsule_GetName(cobj));
        goto bad;
    }
    *p = PyCapsule_GetPointer(cobj, sig);
    if (!(*p))
        goto bad;
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(d);
    return -1;
}
#endif

/////////////// VoidPtrExport.proto ///////////////

static int __Pyx_ExportVoidPtr(PyObject *name, void *p, const char *sig); /*proto*/

/////////////// VoidPtrExport ///////////////
//@substitute: naming
//@requires: ObjectHandling.c::PyObjectSetAttrStr

static int __Pyx_ExportVoidPtr(PyObject *name, void *p, const char *sig) {
    PyObject *d;
    PyObject *cobj = 0;

    d = PyDict_GetItem($moddict_cname, PYIDENT("$api_name"));
    Py_XINCREF(d);
    if (!d) {
        d = PyDict_New();
        if (!d)
            goto bad;
        if (__Pyx_PyObject_SetAttrStr($module_cname, PYIDENT("$api_name"), d) < 0)
            goto bad;
    }
    cobj = PyCapsule_New(p, sig, 0);
    if (!cobj)
        goto bad;
    if (PyDict_SetItem(d, name, cobj) < 0)
        goto bad;
    Py_DECREF(cobj);
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(cobj);
    Py_XDECREF(d);
    return -1;
}


/////////////// SetVTable.proto ///////////////

static int __Pyx_SetVtable(PyTypeObject* typeptr , void* vtable); /*proto*/

/////////////// SetVTable ///////////////

static int __Pyx_SetVtable(PyTypeObject *type, void *vtable) {
    PyObject *ob = PyCapsule_New(vtable, 0, 0);
    if (unlikely(!ob))
        goto bad;
#if CYTHON_COMPILING_IN_LIMITED_API
    if (unlikely(PyObject_SetAttr((PyObject *) type, PYIDENT("__pyx_vtable__"), ob) < 0))
#else
    if (unlikely(PyDict_SetItem(type->tp_dict, PYIDENT("__pyx_vtable__"), ob) < 0))
#endif
        goto bad;
    Py_DECREF(ob);
    return 0;
bad:
    Py_XDECREF(ob);
    return -1;
}


/////////////// GetVTable.proto ///////////////

static void* __Pyx_GetVtable(PyTypeObject *type); /*proto*/

/////////////// GetVTable ///////////////

static void* __Pyx_GetVtable(PyTypeObject *type) {
    void* ptr;
#if CYTHON_COMPILING_IN_LIMITED_API
    PyObject *ob = PyObject_GetAttr((PyObject *)type, PYIDENT("__pyx_vtable__"));
#else
    PyObject *ob = PyObject_GetItem(type->tp_dict, PYIDENT("__pyx_vtable__"));
#endif
    if (!ob)
        goto bad;
    ptr = PyCapsule_GetPointer(ob, 0);
    if (!ptr && !PyErr_Occurred())
        PyErr_SetString(PyExc_RuntimeError, "invalid vtable found for imported type");
    Py_DECREF(ob);
    return ptr;
bad:
    Py_XDECREF(ob);
    return NULL;
}


/////////////// MergeVTables.proto ///////////////
//@requires: GetVTable

// TODO: find a way to make this work with the Limited API!
#if !CYTHON_COMPILING_IN_LIMITED_API
static int __Pyx_MergeVtables(PyTypeObject *type); /*proto*/
#endif

/////////////// MergeVTables ///////////////

#if !CYTHON_COMPILING_IN_LIMITED_API
static int __Pyx_MergeVtables(PyTypeObject *type) {
    int i;
    void** base_vtables;
    __Pyx_TypeName tp_base_name;
    __Pyx_TypeName base_name;
    void* unknown = (void*)-1;
    PyObject* bases = type->tp_bases;
    int base_depth = 0;
    {
        PyTypeObject* base = type->tp_base;
        while (base) {
            base_depth += 1;
            base = base->tp_base;
        }
    }
    base_vtables = (void**) malloc(sizeof(void*) * (size_t)(base_depth + 1));
    base_vtables[0] = unknown;
    // Could do MRO resolution of individual methods in the future, assuming
    // compatible vtables, but for now simply require a common vtable base.
    // Note that if the vtables of various bases are extended separately,
    // resolution isn't possible and we must reject it just as when the
    // instance struct is so extended.  (It would be good to also do this
    // check when a multiple-base class is created in pure Python as well.)
    for (i = 1; i < PyTuple_GET_SIZE(bases); i++) {
        void* base_vtable = __Pyx_GetVtable(((PyTypeObject*)PyTuple_GET_ITEM(bases, i)));
        if (base_vtable != NULL) {
            int j;
            PyTypeObject* base = type->tp_base;
            for (j = 0; j < base_depth; j++) {
                if (base_vtables[j] == unknown) {
                    base_vtables[j] = __Pyx_GetVtable(base);
                    base_vtables[j + 1] = unknown;
                }
                if (base_vtables[j] == base_vtable) {
                    break;
                } else if (base_vtables[j] == NULL) {
                    // No more potential matching bases (with vtables).
                    goto bad;
                }
                base = base->tp_base;
            }
        }
    }
    PyErr_Clear();
    free(base_vtables);
    return 0;
bad:
    tp_base_name = __Pyx_PyType_GetName(type->tp_base);
    base_name = __Pyx_PyType_GetName((PyTypeObject*)PyTuple_GET_ITEM(bases, i));
    PyErr_Format(PyExc_TypeError,
        "multiple bases have vtable conflict: '" __Pyx_FMT_TYPENAME "' and '" __Pyx_FMT_TYPENAME "'", tp_base_name, base_name);
    __Pyx_DECREF_TypeName(tp_base_name);
    __Pyx_DECREF_TypeName(base_name);
    free(base_vtables);
    return -1;
}
#endif


/////////////// ImportNumPyArray.proto ///////////////

static PyObject *__pyx_numpy_ndarray = NULL;

static PyObject* __Pyx_ImportNumPyArrayTypeIfAvailable(void); /*proto*/

/////////////// ImportNumPyArray.cleanup ///////////////
Py_CLEAR(__pyx_numpy_ndarray);

/////////////// ImportNumPyArray ///////////////
//@requires: ImportExport.c::Import

static PyObject* __Pyx__ImportNumPyArray(void) {
    PyObject *numpy_module, *ndarray_object = NULL;
    numpy_module = __Pyx_Import(PYIDENT("numpy"), NULL, 0);
    if (likely(numpy_module)) {
        ndarray_object = PyObject_GetAttrString(numpy_module, "ndarray");
        Py_DECREF(numpy_module);
    }
    if (unlikely(!ndarray_object)) {
        // ImportError, AttributeError, ...
        PyErr_Clear();
    }
    if (unlikely(!ndarray_object || !PyObject_TypeCheck(ndarray_object, &PyType_Type))) {
        Py_XDECREF(ndarray_object);
        Py_INCREF(Py_None);
        ndarray_object = Py_None;
    }
    return ndarray_object;
}

static CYTHON_INLINE PyObject* __Pyx_ImportNumPyArrayTypeIfAvailable(void) {
    if (unlikely(!__pyx_numpy_ndarray)) {
        __pyx_numpy_ndarray = __Pyx__ImportNumPyArray();
    }
    Py_INCREF(__pyx_numpy_ndarray);
    return __pyx_numpy_ndarray;
}
