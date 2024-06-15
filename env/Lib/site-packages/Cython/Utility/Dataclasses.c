///////////////////// ModuleLoader.proto //////////////////////////

static PyObject* __Pyx_LoadInternalModule(const char* name, const char* fallback_code); /* proto */

//////////////////// ModuleLoader ///////////////////////
//@requires: CommonStructures.c::FetchSharedCythonModule

static PyObject* __Pyx_LoadInternalModule(const char* name, const char* fallback_code) {
    // We want to be able to use the contents of the standard library dataclasses module where available.
    // If those objects aren't available (due to Python version) then a simple fallback is substituted
    // instead, which largely just fails with a not-implemented error.
    //
    // The fallbacks are placed in the "shared abi module" as a convenient internal place to
    // store them

    PyObject *shared_abi_module = 0, *module = 0;
#if __PYX_LIMITED_VERSION_HEX >= 0x030d00A1
    PyObject *result;
#endif

    shared_abi_module = __Pyx_FetchSharedCythonABIModule();
    if (!shared_abi_module) return NULL;

#if __PYX_LIMITED_VERSION_HEX >= 0x030d00A1
     if (PyObject_GetOptionalAttrString(shared_abi_module, name, &result) != 0) {
        Py_DECREF(shared_abi_module);
        return result;
     }
#else
    if (PyObject_HasAttrString(shared_abi_module, name)) {
        PyObject* result = PyObject_GetAttrString(shared_abi_module, name);
        Py_DECREF(shared_abi_module);
        return result;
    }
#endif

    // the best and simplest case is simply to defer to the standard library (if available)
    module = PyImport_ImportModule(name);
    if (!module) {
        PyObject *localDict, *runValue, *builtins, *modulename;
        if (!PyErr_ExceptionMatches(PyExc_ImportError)) goto bad;
        PyErr_Clear();  /* this is reasonably likely (especially on older versions of Python) */
#if PY_MAJOR_VERSION < 3
        modulename = PyBytes_FromFormat("_cython_" CYTHON_ABI ".%s", name);
#else
        modulename = PyUnicode_FromFormat("_cython_" CYTHON_ABI ".%s", name);
#endif
        if (!modulename) goto bad;
#if PY_MAJOR_VERSION >= 3 && CYTHON_COMPILING_IN_CPYTHON
        module = PyImport_AddModuleObject(modulename);  /* borrowed */
#else
        module = PyImport_AddModule(PyBytes_AsString(modulename));  /* borrowed */
#endif
        Py_DECREF(modulename);
        if (!module) goto bad;
        Py_INCREF(module);
        if (PyObject_SetAttrString(shared_abi_module, name, module) < 0) goto bad;
        localDict = PyModule_GetDict(module);  /* borrowed */
        if (!localDict) goto bad;
        builtins = PyEval_GetBuiltins();  /* borrowed */
        if (!builtins) goto bad;
        if (PyDict_SetItemString(localDict, "__builtins__", builtins) <0) goto bad;

        runValue = PyRun_String(fallback_code, Py_file_input, localDict, localDict);
        if (!runValue) goto bad;
        Py_DECREF(runValue);
    }
    goto shared_cleanup;

    bad:
        Py_CLEAR(module);
    shared_cleanup:
        Py_XDECREF(shared_abi_module);
    return module;
}

///////////////////// SpecificModuleLoader.proto //////////////////////
//@substitute: tempita

static PyObject* __Pyx_Load_{{cname}}_Module(void); /* proto */


//////////////////// SpecificModuleLoader ///////////////////////
//@requires: ModuleLoader

static PyObject* __Pyx_Load_{{cname}}_Module(void) {
    return __Pyx_LoadInternalModule("{{cname}}", {{py_code}});
}

//////////////////// DataclassesCallHelper.proto ////////////////////////

static PyObject* __Pyx_DataclassesCallHelper(PyObject *callable, PyObject *kwds); /* proto */

//////////////////// DataclassesCallHelper ////////////////////////
//@substitute: naming

// The signature of a few of the dataclasses module functions has
// been expanded over the years. Cython always passes the full set
// of arguments from the most recent version we know of, so needs
// to remove any arguments that don't exist on earlier versions.

#if PY_MAJOR_VERSION >= 3
static int __Pyx_DataclassesCallHelper_FilterToDict(PyObject *callable, PyObject *kwds, PyObject *new_kwds, PyObject *args_list, int is_kwonly) {
    Py_ssize_t size, i;
    size = PySequence_Size(args_list);
    if (size == -1) return -1;

    for (i=0; i<size; ++i) {
        PyObject *key, *value;
        int setitem_result;
        key = PySequence_GetItem(args_list, i);
        if (!key) return -1;

        if (PyUnicode_Check(key) && (
                PyUnicode_CompareWithASCIIString(key, "self") == 0 ||
                // namedtuple constructor in fallback code
                PyUnicode_CompareWithASCIIString(key, "_cls") == 0)) {
            Py_DECREF(key);
            continue;
        }

        value = PyDict_GetItem(kwds, key);
        if (!value) {
            if (is_kwonly) {
                Py_DECREF(key);
                continue;
            } else {
                // The most likely reason for this is that Cython
                // hasn't kept up to date with the Python dataclasses module.
                // To be nice to our users, try not to fail, but ask them
                // to report a bug so we can keep up to date.
                value = Py_None;
                if (PyErr_WarnFormat(
                        PyExc_RuntimeWarning, 1,
                        "Argument %S not passed to %R. This is likely a bug in Cython so please report it.",
                        key, callable) == -1) {
                    Py_DECREF(key);
                    return -1;
                }
            }
        }
        Py_INCREF(value);
        setitem_result = PyDict_SetItem(new_kwds, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
        if (setitem_result == -1) return -1;
    }
    return 0;
}
#endif

static PyObject* __Pyx_DataclassesCallHelper(PyObject *callable, PyObject *kwds) {
#if PY_MAJOR_VERSION < 3
    // We're falling back to our full replacement anyway
    return PyObject_Call(callable, $empty_tuple, kwds);
#else
    PyObject *new_kwds=NULL, *result=NULL;
    PyObject *inspect;
    PyObject *args_list=NULL, *kwonly_args_list=NULL, *getfullargspec_result=NULL;

    // Going via inspect to work out what arguments to pass is unlikely to be the
    // fastest thing ever. However, it is compatible, and only happens once
    // at module-import time.
    inspect = PyImport_ImportModule("inspect");
    if (!inspect) goto bad;
    getfullargspec_result = PyObject_CallMethodObjArgs(inspect, PYUNICODE("getfullargspec"), callable, NULL);
    Py_DECREF(inspect);
    if (!getfullargspec_result) goto bad;
    args_list = PyObject_GetAttrString(getfullargspec_result, "args");
    if (!args_list) goto bad;
    kwonly_args_list = PyObject_GetAttrString(getfullargspec_result, "kwonlyargs");
    if (!kwonly_args_list) goto bad;

    new_kwds = PyDict_New();
    if (!new_kwds) goto bad;

    // copy over only those arguments that are in the specification
    if (__Pyx_DataclassesCallHelper_FilterToDict(callable, kwds, new_kwds, args_list, 0) == -1) goto bad;
    if (__Pyx_DataclassesCallHelper_FilterToDict(callable, kwds, new_kwds, kwonly_args_list, 1) == -1) goto bad;
    result = PyObject_Call(callable, $empty_tuple, new_kwds);
bad:
    Py_XDECREF(getfullargspec_result);
    Py_XDECREF(args_list);
    Py_XDECREF(kwonly_args_list);
    Py_XDECREF(new_kwds);
    return result;
#endif
}
