//////////////////// ArgTypeTest.proto ////////////////////


#define __Pyx_ArgTypeTest(obj, type, none_allowed, name, exact) \
    ((likely(__Pyx_IS_TYPE(obj, type) | (none_allowed && (obj == Py_None)))) ? 1 : \
        __Pyx__ArgTypeTest(obj, type, name, exact))

static int __Pyx__ArgTypeTest(PyObject *obj, PyTypeObject *type, const char *name, int exact); /*proto*/

//////////////////// ArgTypeTest ////////////////////

static int __Pyx__ArgTypeTest(PyObject *obj, PyTypeObject *type, const char *name, int exact)
{
    __Pyx_TypeName type_name;
    __Pyx_TypeName obj_type_name;
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    else if (exact) {
        #if PY_MAJOR_VERSION == 2
        if ((type == &PyBaseString_Type) && likely(__Pyx_PyBaseString_CheckExact(obj))) return 1;
        #endif
    }
    else {
        if (likely(__Pyx_TypeCheck(obj, type))) return 1;
    }
    type_name = __Pyx_PyType_GetName(type);
    obj_type_name = __Pyx_PyType_GetName(Py_TYPE(obj));
    PyErr_Format(PyExc_TypeError,
        "Argument '%.200s' has incorrect type (expected " __Pyx_FMT_TYPENAME
        ", got " __Pyx_FMT_TYPENAME ")", name, type_name, obj_type_name);
    __Pyx_DECREF_TypeName(type_name);
    __Pyx_DECREF_TypeName(obj_type_name);
    return 0;
}

//////////////////// RaiseArgTupleInvalid.proto ////////////////////

static void __Pyx_RaiseArgtupleInvalid(const char* func_name, int exact,
    Py_ssize_t num_min, Py_ssize_t num_max, Py_ssize_t num_found); /*proto*/

//////////////////// RaiseArgTupleInvalid ////////////////////

//  __Pyx_RaiseArgtupleInvalid raises the correct exception when too
//  many or too few positional arguments were found.  This handles
//  Py_ssize_t formatting correctly.

static void __Pyx_RaiseArgtupleInvalid(
    const char* func_name,
    int exact,
    Py_ssize_t num_min,
    Py_ssize_t num_max,
    Py_ssize_t num_found)
{
    Py_ssize_t num_expected;
    const char *more_or_less;

    if (num_found < num_min) {
        num_expected = num_min;
        more_or_less = "at least";
    } else {
        num_expected = num_max;
        more_or_less = "at most";
    }
    if (exact) {
        more_or_less = "exactly";
    }
    PyErr_Format(PyExc_TypeError,
                 "%.200s() takes %.8s %" CYTHON_FORMAT_SSIZE_T "d positional argument%.1s (%" CYTHON_FORMAT_SSIZE_T "d given)",
                 func_name, more_or_less, num_expected,
                 (num_expected == 1) ? "" : "s", num_found);
}


//////////////////// RaiseKeywordRequired.proto ////////////////////

static void __Pyx_RaiseKeywordRequired(const char* func_name, PyObject* kw_name); /*proto*/

//////////////////// RaiseKeywordRequired ////////////////////

static void __Pyx_RaiseKeywordRequired(const char* func_name, PyObject* kw_name) {
    PyErr_Format(PyExc_TypeError,
        #if PY_MAJOR_VERSION >= 3
        "%s() needs keyword-only argument %U", func_name, kw_name);
        #else
        "%s() needs keyword-only argument %s", func_name,
        PyString_AS_STRING(kw_name));
        #endif
}


//////////////////// RaiseDoubleKeywords.proto ////////////////////

static void __Pyx_RaiseDoubleKeywordsError(const char* func_name, PyObject* kw_name); /*proto*/

//////////////////// RaiseDoubleKeywords ////////////////////

static void __Pyx_RaiseDoubleKeywordsError(
    const char* func_name,
    PyObject* kw_name)
{
    PyErr_Format(PyExc_TypeError,
        #if PY_MAJOR_VERSION >= 3
        "%s() got multiple values for keyword argument '%U'", func_name, kw_name);
        #else
        "%s() got multiple values for keyword argument '%s'", func_name,
        PyString_AsString(kw_name));
        #endif
}


//////////////////// RaiseMappingExpected.proto ////////////////////

static void __Pyx_RaiseMappingExpectedError(PyObject* arg); /*proto*/

//////////////////// RaiseMappingExpected ////////////////////

static void __Pyx_RaiseMappingExpectedError(PyObject* arg) {
    __Pyx_TypeName arg_type_name = __Pyx_PyType_GetName(Py_TYPE(arg));
    PyErr_Format(PyExc_TypeError,
        "'" __Pyx_FMT_TYPENAME "' object is not a mapping", arg_type_name);
    __Pyx_DECREF_TypeName(arg_type_name);
}


//////////////////// KeywordStringCheck.proto ////////////////////

static int __Pyx_CheckKeywordStrings(PyObject *kw, const char* function_name, int kw_allowed); /*proto*/

//////////////////// KeywordStringCheck ////////////////////

// __Pyx_CheckKeywordStrings raises an error if non-string keywords
// were passed to a function, or if any keywords were passed to a
// function that does not accept them.
//
// The "kw" argument is either a dict (for METH_VARARGS) or a tuple
// (for METH_FASTCALL).

static int __Pyx_CheckKeywordStrings(
    PyObject *kw,
    const char* function_name,
    int kw_allowed)
{
    PyObject* key = 0;
    Py_ssize_t pos = 0;
#if CYTHON_COMPILING_IN_PYPY
    /* PyPy appears to check keywords at call time, not at unpacking time => not much to do here */
    if (!kw_allowed && PyDict_Next(kw, &pos, &key, 0))
        goto invalid_keyword;
    return 1;
#else
    if (CYTHON_METH_FASTCALL && likely(PyTuple_Check(kw))) {
        Py_ssize_t kwsize;
#if CYTHON_ASSUME_SAFE_MACROS
        kwsize = PyTuple_GET_SIZE(kw);
#else
        kwsize = PyTuple_Size(kw);
        if (kwsize < 0) return 0;
#endif
        if (unlikely(kwsize == 0))
            return 1;
        if (!kw_allowed) {
#if CYTHON_ASSUME_SAFE_MACROS
            key = PyTuple_GET_ITEM(kw, 0);
#else
            key = PyTuple_GetItem(kw, pos);
            if (!key) return 0;
#endif
            goto invalid_keyword;
        }
#if PY_VERSION_HEX < 0x03090000
        // On CPython >= 3.9, the FASTCALL protocol guarantees that keyword
        // names are strings (see https://bugs.python.org/issue37540)
        for (pos = 0; pos < kwsize; pos++) {
#if CYTHON_ASSUME_SAFE_MACROS
            key = PyTuple_GET_ITEM(kw, pos);
#else
            key = PyTuple_GetItem(kw, pos);
            if (!key) return 0;
#endif
            if (unlikely(!PyUnicode_Check(key)))
                goto invalid_keyword_type;
        }
#endif
        return 1;
    }

    while (PyDict_Next(kw, &pos, &key, 0)) {
        #if PY_MAJOR_VERSION < 3
        if (unlikely(!PyString_Check(key)))
        #endif
            if (unlikely(!PyUnicode_Check(key)))
                goto invalid_keyword_type;
    }
    if (!kw_allowed && unlikely(key))
        goto invalid_keyword;
    return 1;
invalid_keyword_type:
    PyErr_Format(PyExc_TypeError,
        "%.200s() keywords must be strings", function_name);
    return 0;
#endif
invalid_keyword:
    #if PY_MAJOR_VERSION < 3
    PyErr_Format(PyExc_TypeError,
        "%.200s() got an unexpected keyword argument '%.200s'",
        function_name, PyString_AsString(key));
    #else
    PyErr_Format(PyExc_TypeError,
        "%s() got an unexpected keyword argument '%U'",
        function_name, key);
    #endif
    return 0;
}


//////////////////// ParseKeywords.proto ////////////////////

static int __Pyx_ParseOptionalKeywords(PyObject *kwds, PyObject *const *kwvalues,
    PyObject **argnames[],
    PyObject *kwds2, PyObject *values[], Py_ssize_t num_pos_args,
    const char* function_name); /*proto*/

//////////////////// ParseKeywords ////////////////////
//@requires: RaiseDoubleKeywords

//  __Pyx_ParseOptionalKeywords copies the optional/unknown keyword
//  arguments from kwds into the dict kwds2.  If kwds2 is NULL, unknown
//  keywords will raise an invalid keyword error.
//
//  When not using METH_FASTCALL, kwds is a dict and kwvalues is NULL.
//  Otherwise, kwds is a tuple with keyword names and kwvalues is a C
//  array with the corresponding values.
//
//  Three kinds of errors are checked: 1) non-string keywords, 2)
//  unexpected keywords and 3) overlap with positional arguments.
//
//  If num_posargs is greater 0, it denotes the number of positional
//  arguments that were passed and that must therefore not appear
//  amongst the keywords as well.
//
//  This method does not check for required keyword arguments.

static int __Pyx_ParseOptionalKeywords(
    PyObject *kwds,
    PyObject *const *kwvalues,
    PyObject **argnames[],
    PyObject *kwds2,
    PyObject *values[],
    Py_ssize_t num_pos_args,
    const char* function_name)
{
    PyObject *key = 0, *value = 0;
    Py_ssize_t pos = 0;
    PyObject*** name;
    PyObject*** first_kw_arg = argnames + num_pos_args;
    int kwds_is_tuple = CYTHON_METH_FASTCALL && likely(PyTuple_Check(kwds));

    while (1) {
        // clean up key and value when the loop is "continued"
        Py_XDECREF(key); key = NULL;
        Py_XDECREF(value); value = NULL;

        if (kwds_is_tuple) {
            Py_ssize_t size;
#if CYTHON_ASSUME_SAFE_MACROS
            size = PyTuple_GET_SIZE(kwds);
#else
            size = PyTuple_Size(kwds);
            if (size < 0) goto bad;
#endif
            if (pos >= size) break;

#if CYTHON_AVOID_BORROWED_REFS
            // Get an owned reference to key.
            key = __Pyx_PySequence_ITEM(kwds, pos);
            if (!key) goto bad;
#elif CYTHON_ASSUME_SAFE_MACROS
            key = PyTuple_GET_ITEM(kwds, pos);
#else
            key = PyTuple_GetItem(kwds, pos);
            if (!key) goto bad;
#endif

            value = kwvalues[pos];
            pos++;
        }
        else
        {
            if (!PyDict_Next(kwds, &pos, &key, &value)) break;
            // It's unfortunately hard to avoid borrowed references (briefly) with PyDict_Next
#if CYTHON_AVOID_BORROWED_REFS
            // Own the reference to match the behaviour above.
            Py_INCREF(key);
#endif
        }

        name = first_kw_arg;
        while (*name && (**name != key)) name++;
        if (*name) {
            values[name-argnames] = value;
#if CYTHON_AVOID_BORROWED_REFS
            Py_INCREF(value);  /* transfer ownership of value to values */
            Py_DECREF(key);
#endif
            key = NULL;
            value = NULL;
            continue;
        }

        // Now make sure we own both references since we're doing non-trivial Python operations.
#if !CYTHON_AVOID_BORROWED_REFS
        Py_INCREF(key);
#endif
        Py_INCREF(value);

        name = first_kw_arg;
        #if PY_MAJOR_VERSION < 3
        if (likely(PyString_Check(key))) {
            while (*name) {
                if ((CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**name) == PyString_GET_SIZE(key))
                        && _PyString_Eq(**name, key)) {
                    values[name-argnames] = value;
#if CYTHON_AVOID_BORROWED_REFS
                    value = NULL;  /* ownership transferred to values */
#endif
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                // not found after positional args, check for duplicate
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    if ((**argname == key) || (
                            (CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**argname) == PyString_GET_SIZE(key))
                             && _PyString_Eq(**argname, key))) {
                        goto arg_passed_twice;
                    }
                    argname++;
                }
            }
        } else
        #endif
        if (likely(PyUnicode_Check(key))) {
            while (*name) {
                int cmp = (
                #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                    (__Pyx_PyUnicode_GET_LENGTH(**name) != __Pyx_PyUnicode_GET_LENGTH(key)) ? 1 :
                #endif
                    // In Py2, we may need to convert the argument name from str to unicode for comparison.
                    PyUnicode_Compare(**name, key)
                );
                if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                if (cmp == 0) {
                    values[name-argnames] = value;
#if CYTHON_AVOID_BORROWED_REFS
                    value = NULL;  /* ownership transferred to values */
#endif
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                // not found after positional args, check for duplicate
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    int cmp = (**argname == key) ? 0 :
                    #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                        (__Pyx_PyUnicode_GET_LENGTH(**argname) != __Pyx_PyUnicode_GET_LENGTH(key)) ? 1 :
                    #endif
                        // need to convert argument name from bytes to unicode for comparison
                        PyUnicode_Compare(**argname, key);
                    if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                    if (cmp == 0) goto arg_passed_twice;
                    argname++;
                }
            }
        } else
            goto invalid_keyword_type;

        if (kwds2) {
            if (unlikely(PyDict_SetItem(kwds2, key, value))) goto bad;
        } else {
            goto invalid_keyword;
        }
    }
    Py_XDECREF(key);
    Py_XDECREF(value);
    return 0;
arg_passed_twice:
    __Pyx_RaiseDoubleKeywordsError(function_name, key);
    goto bad;
invalid_keyword_type:
    PyErr_Format(PyExc_TypeError,
        "%.200s() keywords must be strings", function_name);
    goto bad;
invalid_keyword:
    #if PY_MAJOR_VERSION < 3
    PyErr_Format(PyExc_TypeError,
        "%.200s() got an unexpected keyword argument '%.200s'",
        function_name, PyString_AsString(key));
    #else
    PyErr_Format(PyExc_TypeError,
        "%s() got an unexpected keyword argument '%U'",
        function_name, key);
    #endif
bad:
    Py_XDECREF(key);
    Py_XDECREF(value);
    return -1;
}


//////////////////// MergeKeywords.proto ////////////////////

static int __Pyx_MergeKeywords(PyObject *kwdict, PyObject *source_mapping); /*proto*/

//////////////////// MergeKeywords ////////////////////
//@requires: RaiseDoubleKeywords
//@requires: Optimize.c::dict_iter

static int __Pyx_MergeKeywords(PyObject *kwdict, PyObject *source_mapping) {
    PyObject *iter, *key = NULL, *value = NULL;
    int source_is_dict, result;
    Py_ssize_t orig_length, ppos = 0;

    iter = __Pyx_dict_iterator(source_mapping, 0, PYIDENT("items"), &orig_length, &source_is_dict);
    if (unlikely(!iter)) {
        // slow fallback: try converting to dict, then iterate
        PyObject *args;
        if (unlikely(!PyErr_ExceptionMatches(PyExc_AttributeError))) goto bad;
        PyErr_Clear();
        args = PyTuple_Pack(1, source_mapping);
        if (likely(args)) {
            PyObject *fallback = PyObject_Call((PyObject*)&PyDict_Type, args, NULL);
            Py_DECREF(args);
            if (likely(fallback)) {
                iter = __Pyx_dict_iterator(fallback, 1, PYIDENT("items"), &orig_length, &source_is_dict);
                Py_DECREF(fallback);
            }
        }
        if (unlikely(!iter)) goto bad;
    }

    while (1) {
        result = __Pyx_dict_iter_next(iter, orig_length, &ppos, &key, &value, NULL, source_is_dict);
        if (unlikely(result < 0)) goto bad;
        if (!result) break;

        if (unlikely(PyDict_Contains(kwdict, key))) {
            __Pyx_RaiseDoubleKeywordsError("function", key);
            result = -1;
        } else {
            result = PyDict_SetItem(kwdict, key, value);
        }
        Py_DECREF(key);
        Py_DECREF(value);
        if (unlikely(result < 0)) goto bad;
    }
    Py_XDECREF(iter);
    return 0;

bad:
    Py_XDECREF(iter);
    return -1;
}


/////////////// fastcall.proto ///////////////

// We define various functions and macros with two variants:
//..._FASTCALL and ..._VARARGS

// The first is used when METH_FASTCALL is enabled and the second is used
// otherwise. If the Python implementation does not support METH_FASTCALL
// (because it's an old version of CPython or it's not CPython at all),
// then the ..._FASTCALL macros simply alias ..._VARARGS

#if CYTHON_AVOID_BORROWED_REFS
    // This is the only case where we request an owned reference.
    #define __Pyx_Arg_VARARGS(args, i) PySequence_GetItem(args, i)
#elif CYTHON_ASSUME_SAFE_MACROS
    #define __Pyx_Arg_VARARGS(args, i) PyTuple_GET_ITEM(args, i)
#else
    #define __Pyx_Arg_VARARGS(args, i) PyTuple_GetItem(args, i)
#endif
#if CYTHON_AVOID_BORROWED_REFS
    #define __Pyx_Arg_NewRef_VARARGS(arg) __Pyx_NewRef(arg)
    #define __Pyx_Arg_XDECREF_VARARGS(arg) Py_XDECREF(arg)
#else
    #define __Pyx_Arg_NewRef_VARARGS(arg) arg  /* no-op */
    #define __Pyx_Arg_XDECREF_VARARGS(arg)     /* no-op - arg is borrowed */
#endif
#define __Pyx_NumKwargs_VARARGS(kwds) PyDict_Size(kwds)
#define __Pyx_KwValues_VARARGS(args, nargs) NULL
#define __Pyx_GetKwValue_VARARGS(kw, kwvalues, s) __Pyx_PyDict_GetItemStrWithError(kw, s)
#define __Pyx_KwargsAsDict_VARARGS(kw, kwvalues) PyDict_Copy(kw)
#if CYTHON_METH_FASTCALL
    #define __Pyx_Arg_FASTCALL(args, i) args[i]
    #define __Pyx_NumKwargs_FASTCALL(kwds) PyTuple_GET_SIZE(kwds)
    #define __Pyx_KwValues_FASTCALL(args, nargs) ((args) + (nargs))
    static CYTHON_INLINE PyObject * __Pyx_GetKwValue_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues, PyObject *s);
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030d0000
    CYTHON_UNUSED static PyObject *__Pyx_KwargsAsDict_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues);/*proto*/
  #else
    #define __Pyx_KwargsAsDict_FASTCALL(kw, kwvalues) _PyStack_AsDict(kwvalues, kw)
  #endif
    #define __Pyx_Arg_NewRef_FASTCALL(arg) arg  /* no-op, __Pyx_Arg_FASTCALL is direct and this needs
                                                   to have the same reference counting */
    #define __Pyx_Arg_XDECREF_FASTCALL(arg)     /* no-op - arg was returned from array */
#else
    #define __Pyx_Arg_FASTCALL __Pyx_Arg_VARARGS
    #define __Pyx_NumKwargs_FASTCALL __Pyx_NumKwargs_VARARGS
    #define __Pyx_KwValues_FASTCALL __Pyx_KwValues_VARARGS
    #define __Pyx_GetKwValue_FASTCALL __Pyx_GetKwValue_VARARGS
    #define __Pyx_KwargsAsDict_FASTCALL __Pyx_KwargsAsDict_VARARGS
    #define __Pyx_Arg_NewRef_FASTCALL(arg) __Pyx_Arg_NewRef_VARARGS(arg)
    #define __Pyx_Arg_XDECREF_FASTCALL(arg) __Pyx_Arg_XDECREF_VARARGS(arg)
#endif

#if CYTHON_COMPILING_IN_CPYTHON && CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
#define __Pyx_ArgsSlice_VARARGS(args, start, stop) __Pyx_PyTuple_FromArray(&__Pyx_Arg_VARARGS(args, start), stop - start)
#define __Pyx_ArgsSlice_FASTCALL(args, start, stop) __Pyx_PyTuple_FromArray(&__Pyx_Arg_FASTCALL(args, start), stop - start)
#else
/* Not CPython, so certainly no METH_FASTCALL support */
#define __Pyx_ArgsSlice_VARARGS(args, start, stop) PyTuple_GetSlice(args, start, stop)
#define __Pyx_ArgsSlice_FASTCALL(args, start, stop) PyTuple_GetSlice(args, start, stop)
#endif


/////////////// fastcall ///////////////
//@requires: ObjectHandling.c::TupleAndListFromArray
//@requires: StringTools.c::UnicodeEquals

#if CYTHON_METH_FASTCALL
// kwnames: tuple with names of keyword arguments
// kwvalues: C array with values of keyword arguments
// s: str with the keyword name to look for
static CYTHON_INLINE PyObject * __Pyx_GetKwValue_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues, PyObject *s)
{
    // Search the kwnames array for s and return the corresponding value.
    // We do two loops: a first one to compare pointers (which will find a
    // match if the name in kwnames is interned, given that s is interned
    // by Cython). A second loop compares the actual strings.
    Py_ssize_t i, n = PyTuple_GET_SIZE(kwnames);
    for (i = 0; i < n; i++)
    {
        if (s == PyTuple_GET_ITEM(kwnames, i)) return kwvalues[i];
    }
    for (i = 0; i < n; i++)
    {
        int eq = __Pyx_PyUnicode_Equals(s, PyTuple_GET_ITEM(kwnames, i), Py_EQ);
        if (unlikely(eq != 0)) {
            if (unlikely(eq < 0)) return NULL;  /* error */
            return kwvalues[i];
        }
    }
    return NULL;  /* not found (no exception set) */
}

#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030d0000
CYTHON_UNUSED static PyObject *__Pyx_KwargsAsDict_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues) {
    Py_ssize_t i, nkwargs = PyTuple_GET_SIZE(kwnames);
    PyObject *dict;

    dict = PyDict_New();
    if (unlikely(!dict))
        return NULL;

    for (i=0; i<nkwargs; i++) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        if (unlikely(PyDict_SetItem(dict, key, kwvalues[i]) < 0))
            goto bad;
    }
    return dict;

bad:
    Py_DECREF(dict);
    return NULL;
}
#endif

#endif
