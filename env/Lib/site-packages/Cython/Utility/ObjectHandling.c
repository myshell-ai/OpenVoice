/*
 * General object operations and protocol implementations,
 * including their specialisations for certain builtins.
 *
 * Optional optimisations for builtins are in Optimize.c.
 *
 * Required replacements of builtins are in Builtins.c.
 */

/////////////// RaiseNoneIterError.proto ///////////////

static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void);

/////////////// RaiseNoneIterError ///////////////

static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
}

/////////////// RaiseTooManyValuesToUnpack.proto ///////////////

static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected);

/////////////// RaiseTooManyValuesToUnpack ///////////////

static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected) {
    PyErr_Format(PyExc_ValueError,
                 "too many values to unpack (expected %" CYTHON_FORMAT_SSIZE_T "d)", expected);
}

/////////////// RaiseNeedMoreValuesToUnpack.proto ///////////////

static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index);

/////////////// RaiseNeedMoreValuesToUnpack ///////////////

static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index) {
    PyErr_Format(PyExc_ValueError,
                 "need more than %" CYTHON_FORMAT_SSIZE_T "d value%.1s to unpack",
                 index, (index == 1) ? "" : "s");
}

/////////////// UnpackTupleError.proto ///////////////

static void __Pyx_UnpackTupleError(PyObject *, Py_ssize_t index); /*proto*/

/////////////// UnpackTupleError ///////////////
//@requires: RaiseNoneIterError
//@requires: RaiseNeedMoreValuesToUnpack
//@requires: RaiseTooManyValuesToUnpack

static void __Pyx_UnpackTupleError(PyObject *t, Py_ssize_t index) {
    if (t == Py_None) {
      __Pyx_RaiseNoneNotIterableError();
    } else if (PyTuple_GET_SIZE(t) < index) {
      __Pyx_RaiseNeedMoreValuesError(PyTuple_GET_SIZE(t));
    } else {
      __Pyx_RaiseTooManyValuesError(index);
    }
}

/////////////// UnpackItemEndCheck.proto ///////////////

static int __Pyx_IternextUnpackEndCheck(PyObject *retval, Py_ssize_t expected); /*proto*/

/////////////// UnpackItemEndCheck ///////////////
//@requires: RaiseTooManyValuesToUnpack
//@requires: IterFinish

static int __Pyx_IternextUnpackEndCheck(PyObject *retval, Py_ssize_t expected) {
    if (unlikely(retval)) {
        Py_DECREF(retval);
        __Pyx_RaiseTooManyValuesError(expected);
        return -1;
    }

    return __Pyx_IterFinish();
}

/////////////// UnpackTuple2.proto ///////////////

#define __Pyx_unpack_tuple2(tuple, value1, value2, is_tuple, has_known_size, decref_tuple) \
    (likely(is_tuple || PyTuple_Check(tuple)) ? \
        (likely(has_known_size || PyTuple_GET_SIZE(tuple) == 2) ? \
            __Pyx_unpack_tuple2_exact(tuple, value1, value2, decref_tuple) : \
            (__Pyx_UnpackTupleError(tuple, 2), -1)) : \
        __Pyx_unpack_tuple2_generic(tuple, value1, value2, has_known_size, decref_tuple))

static CYTHON_INLINE int __Pyx_unpack_tuple2_exact(
    PyObject* tuple, PyObject** value1, PyObject** value2, int decref_tuple);
static int __Pyx_unpack_tuple2_generic(
    PyObject* tuple, PyObject** value1, PyObject** value2, int has_known_size, int decref_tuple);

/////////////// UnpackTuple2 ///////////////
//@requires: UnpackItemEndCheck
//@requires: UnpackTupleError
//@requires: RaiseNeedMoreValuesToUnpack

static CYTHON_INLINE int __Pyx_unpack_tuple2_exact(
        PyObject* tuple, PyObject** pvalue1, PyObject** pvalue2, int decref_tuple) {
    PyObject *value1 = NULL, *value2 = NULL;
#if CYTHON_COMPILING_IN_PYPY
    value1 = PySequence_ITEM(tuple, 0);  if (unlikely(!value1)) goto bad;
    value2 = PySequence_ITEM(tuple, 1);  if (unlikely(!value2)) goto bad;
#else
    value1 = PyTuple_GET_ITEM(tuple, 0);  Py_INCREF(value1);
    value2 = PyTuple_GET_ITEM(tuple, 1);  Py_INCREF(value2);
#endif
    if (decref_tuple) {
        Py_DECREF(tuple);
    }

    *pvalue1 = value1;
    *pvalue2 = value2;
    return 0;
#if CYTHON_COMPILING_IN_PYPY
bad:
    Py_XDECREF(value1);
    Py_XDECREF(value2);
    if (decref_tuple) { Py_XDECREF(tuple); }
    return -1;
#endif
}

static int __Pyx_unpack_tuple2_generic(PyObject* tuple, PyObject** pvalue1, PyObject** pvalue2,
                                       int has_known_size, int decref_tuple) {
    Py_ssize_t index;
    PyObject *value1 = NULL, *value2 = NULL, *iter = NULL;
    iternextfunc iternext;

    iter = PyObject_GetIter(tuple);
    if (unlikely(!iter)) goto bad;
    if (decref_tuple) { Py_DECREF(tuple); tuple = NULL; }

    iternext = __Pyx_PyObject_GetIterNextFunc(iter);
    value1 = iternext(iter); if (unlikely(!value1)) { index = 0; goto unpacking_failed; }
    value2 = iternext(iter); if (unlikely(!value2)) { index = 1; goto unpacking_failed; }
    if (!has_known_size && unlikely(__Pyx_IternextUnpackEndCheck(iternext(iter), 2))) goto bad;

    Py_DECREF(iter);
    *pvalue1 = value1;
    *pvalue2 = value2;
    return 0;

unpacking_failed:
    if (!has_known_size && __Pyx_IterFinish() == 0)
        __Pyx_RaiseNeedMoreValuesError(index);
bad:
    Py_XDECREF(iter);
    Py_XDECREF(value1);
    Py_XDECREF(value2);
    if (decref_tuple) { Py_XDECREF(tuple); }
    return -1;
}


/////////////// IterNext.proto ///////////////

#define __Pyx_PyIter_Next(obj) __Pyx_PyIter_Next2(obj, NULL)
static CYTHON_INLINE PyObject *__Pyx_PyIter_Next2(PyObject *, PyObject *); /*proto*/

/////////////// IterNext ///////////////
//@requires: Exceptions.c::PyThreadStateGet
//@requires: Exceptions.c::PyErrFetchRestore

static PyObject *__Pyx_PyIter_Next2Default(PyObject* defval) {
    PyObject* exc_type;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    exc_type = __Pyx_PyErr_CurrentExceptionType();
    if (unlikely(exc_type)) {
        if (!defval || unlikely(!__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration)))
            return NULL;
        __Pyx_PyErr_Clear();
        Py_INCREF(defval);
        return defval;
    }
    if (defval) {
        Py_INCREF(defval);
        return defval;
    }
    __Pyx_PyErr_SetNone(PyExc_StopIteration);
    return NULL;
}

static void __Pyx_PyIter_Next_ErrorNoIterator(PyObject *iterator) {
    __Pyx_TypeName iterator_type_name = __Pyx_PyType_GetName(Py_TYPE(iterator));
    PyErr_Format(PyExc_TypeError,
        __Pyx_FMT_TYPENAME " object is not an iterator", iterator_type_name);
    __Pyx_DECREF_TypeName(iterator_type_name);
}

// originally copied from Py3's builtin_next()
static CYTHON_INLINE PyObject *__Pyx_PyIter_Next2(PyObject* iterator, PyObject* defval) {
    PyObject* next;
    // We always do a quick slot check because calling PyIter_Check() is so wasteful.
    iternextfunc iternext = Py_TYPE(iterator)->tp_iternext;
    if (likely(iternext)) {
#if CYTHON_USE_TYPE_SLOTS || CYTHON_COMPILING_IN_PYPY
        next = iternext(iterator);
        if (likely(next))
            return next;
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX < 0x030d0000
        if (unlikely(iternext == &_PyObject_NextNotImplemented))
            return NULL;
#endif
#else
        // Since the slot was set, assume that PyIter_Next() will likely succeed, and properly fail otherwise.
        // Note: PyIter_Next() crashes in CPython if "tp_iternext" is NULL.
        next = PyIter_Next(iterator);
        if (likely(next))
            return next;
#endif
    } else if (CYTHON_USE_TYPE_SLOTS || unlikely(!PyIter_Check(iterator))) {
        // If CYTHON_USE_TYPE_SLOTS, then the slot was not set and we don't have an iterable.
        // Otherwise, don't trust "tp_iternext" and rely on PyIter_Check().
        __Pyx_PyIter_Next_ErrorNoIterator(iterator);
        return NULL;
    }
#if !CYTHON_USE_TYPE_SLOTS
    else {
        // We have an iterator with an empty "tp_iternext", but didn't call next() on it yet.
        next = PyIter_Next(iterator);
        if (likely(next))
            return next;
    }
#endif
    return __Pyx_PyIter_Next2Default(defval);
}

/////////////// IterFinish.proto ///////////////

static CYTHON_INLINE int __Pyx_IterFinish(void); /*proto*/

/////////////// IterFinish ///////////////
//@requires: Exceptions.c::PyThreadStateGet
//@requires: Exceptions.c::PyErrFetchRestore

// When PyIter_Next(iter) has returned NULL in order to signal termination,
// this function does the right cleanup and returns 0 on success.  If it
// detects an error that occurred in the iterator, it returns -1.

static CYTHON_INLINE int __Pyx_IterFinish(void) {
    PyObject* exc_type;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    exc_type = __Pyx_PyErr_CurrentExceptionType();
    if (unlikely(exc_type)) {
        if (unlikely(!__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration)))
            return -1;
        __Pyx_PyErr_Clear();
        return 0;
    }
    return 0;
}


/////////////// ObjectGetItem.proto ///////////////

#if CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PyObject *__Pyx_PyObject_GetItem(PyObject *obj, PyObject *key);/*proto*/
#else
#define __Pyx_PyObject_GetItem(obj, key)  PyObject_GetItem(obj, key)
#endif

/////////////// ObjectGetItem ///////////////
// //@requires: GetItemInt - added in IndexNode as it uses templating.
//@requires: PyObjectGetAttrStrNoError
//@requires: PyObjectCallOneArg

#if CYTHON_USE_TYPE_SLOTS
static PyObject *__Pyx_PyObject_GetIndex(PyObject *obj, PyObject *index) {
    // Get element from sequence object `obj` at index `index`.
    PyObject *runerr = NULL;
    Py_ssize_t key_value;
    key_value = __Pyx_PyIndex_AsSsize_t(index);
    if (likely(key_value != -1 || !(runerr = PyErr_Occurred()))) {
        return __Pyx_GetItemInt_Fast(obj, key_value, 0, 1, 1);
    }

    // Error handling code -- only manage OverflowError differently.
    if (PyErr_GivenExceptionMatches(runerr, PyExc_OverflowError)) {
        __Pyx_TypeName index_type_name = __Pyx_PyType_GetName(Py_TYPE(index));
        PyErr_Clear();
        PyErr_Format(PyExc_IndexError,
            "cannot fit '" __Pyx_FMT_TYPENAME "' into an index-sized integer", index_type_name);
        __Pyx_DECREF_TypeName(index_type_name);
    }
    return NULL;
}

static PyObject *__Pyx_PyObject_GetItem_Slow(PyObject *obj, PyObject *key) {
    __Pyx_TypeName obj_type_name;
    // Handles less common slow-path checks for GetItem
    if (likely(PyType_Check(obj))) {
        PyObject *meth = __Pyx_PyObject_GetAttrStrNoError(obj, PYIDENT("__class_getitem__"));
        if (!meth) {
            PyErr_Clear();
        } else {
            PyObject *result = __Pyx_PyObject_CallOneArg(meth, key);
            Py_DECREF(meth);
            return result;
        }
    }

    obj_type_name = __Pyx_PyType_GetName(Py_TYPE(obj));
    PyErr_Format(PyExc_TypeError,
        "'" __Pyx_FMT_TYPENAME "' object is not subscriptable", obj_type_name);
    __Pyx_DECREF_TypeName(obj_type_name);
    return NULL;
}

static PyObject *__Pyx_PyObject_GetItem(PyObject *obj, PyObject *key) {
    PyTypeObject *tp = Py_TYPE(obj);
    PyMappingMethods *mm = tp->tp_as_mapping;
    PySequenceMethods *sm = tp->tp_as_sequence;

    if (likely(mm && mm->mp_subscript)) {
        return mm->mp_subscript(obj, key);
    }
    if (likely(sm && sm->sq_item)) {
        return __Pyx_PyObject_GetIndex(obj, key);
    }
    return __Pyx_PyObject_GetItem_Slow(obj, key);
}
#endif


/////////////// DictGetItem.proto ///////////////

#if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
static PyObject *__Pyx_PyDict_GetItem(PyObject *d, PyObject* key);/*proto*/

#define __Pyx_PyObject_Dict_GetItem(obj, name) \
    (likely(PyDict_CheckExact(obj)) ? \
     __Pyx_PyDict_GetItem(obj, name) : PyObject_GetItem(obj, name))

#else
#define __Pyx_PyDict_GetItem(d, key) PyObject_GetItem(d, key)
#define __Pyx_PyObject_Dict_GetItem(obj, name)  PyObject_GetItem(obj, name)
#endif

/////////////// DictGetItem ///////////////

#if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
static PyObject *__Pyx_PyDict_GetItem(PyObject *d, PyObject* key) {
    PyObject *value;
    value = PyDict_GetItemWithError(d, key);
    if (unlikely(!value)) {
        if (!PyErr_Occurred()) {
            if (unlikely(PyTuple_Check(key))) {
                // CPython interprets tuples as separate arguments => must wrap them in another tuple.
                PyObject* args = PyTuple_Pack(1, key);
                if (likely(args)) {
                    PyErr_SetObject(PyExc_KeyError, args);
                    Py_DECREF(args);
                }
            } else {
                // Avoid tuple packing if possible.
                PyErr_SetObject(PyExc_KeyError, key);
            }
        }
        return NULL;
    }
    Py_INCREF(value);
    return value;
}
#endif

/////////////// GetItemInt.proto ///////////////
//@substitute: tempita

#define __Pyx_GetItemInt(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck) \
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ? \
    __Pyx_GetItemInt_Fast(o, (Py_ssize_t)i, is_list, wraparound, boundscheck) : \
    (is_list ? (PyErr_SetString(PyExc_IndexError, "list index out of range"), (PyObject*)NULL) : \
               __Pyx_GetItemInt_Generic(o, to_py_func(i))))

{{for type in ['List', 'Tuple']}}
#define __Pyx_GetItemInt_{{type}}(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck) \
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ? \
    __Pyx_GetItemInt_{{type}}_Fast(o, (Py_ssize_t)i, wraparound, boundscheck) : \
    (PyErr_SetString(PyExc_IndexError, "{{ type.lower() }} index out of range"), (PyObject*)NULL))

static CYTHON_INLINE PyObject *__Pyx_GetItemInt_{{type}}_Fast(PyObject *o, Py_ssize_t i,
                                                              int wraparound, int boundscheck);
{{endfor}}

static PyObject *__Pyx_GetItemInt_Generic(PyObject *o, PyObject* j);
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_Fast(PyObject *o, Py_ssize_t i,
                                                     int is_list, int wraparound, int boundscheck);

/////////////// GetItemInt ///////////////
//@substitute: tempita

static PyObject *__Pyx_GetItemInt_Generic(PyObject *o, PyObject* j) {
    PyObject *r;
    if (unlikely(!j)) return NULL;
    r = PyObject_GetItem(o, j);
    Py_DECREF(j);
    return r;
}

{{for type in ['List', 'Tuple']}}
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_{{type}}_Fast(PyObject *o, Py_ssize_t i,
                                                              CYTHON_NCP_UNUSED int wraparound,
                                                              CYTHON_NCP_UNUSED int boundscheck) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    Py_ssize_t wrapped_i = i;
    if (wraparound & unlikely(i < 0)) {
        wrapped_i += Py{{type}}_GET_SIZE(o);
    }
    if ((!boundscheck) || likely(__Pyx_is_valid_index(wrapped_i, Py{{type}}_GET_SIZE(o)))) {
        PyObject *r = Py{{type}}_GET_ITEM(o, wrapped_i);
        Py_INCREF(r);
        return r;
    }
    return __Pyx_GetItemInt_Generic(o, PyInt_FromSsize_t(i));
#else
    return PySequence_GetItem(o, i);
#endif
}
{{endfor}}

static CYTHON_INLINE PyObject *__Pyx_GetItemInt_Fast(PyObject *o, Py_ssize_t i, int is_list,
                                                     CYTHON_NCP_UNUSED int wraparound,
                                                     CYTHON_NCP_UNUSED int boundscheck) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS && CYTHON_USE_TYPE_SLOTS
    if (is_list || PyList_CheckExact(o)) {
        Py_ssize_t n = ((!wraparound) | likely(i >= 0)) ? i : i + PyList_GET_SIZE(o);
        if ((!boundscheck) || (likely(__Pyx_is_valid_index(n, PyList_GET_SIZE(o))))) {
            PyObject *r = PyList_GET_ITEM(o, n);
            Py_INCREF(r);
            return r;
        }
    }
    else if (PyTuple_CheckExact(o)) {
        Py_ssize_t n = ((!wraparound) | likely(i >= 0)) ? i : i + PyTuple_GET_SIZE(o);
        if ((!boundscheck) || likely(__Pyx_is_valid_index(n, PyTuple_GET_SIZE(o)))) {
            PyObject *r = PyTuple_GET_ITEM(o, n);
            Py_INCREF(r);
            return r;
        }
    } else {
        // inlined PySequence_GetItem() + special cased length overflow
        PyMappingMethods *mm = Py_TYPE(o)->tp_as_mapping;
        PySequenceMethods *sm = Py_TYPE(o)->tp_as_sequence;
        if (mm && mm->mp_subscript) {
            PyObject *r, *key = PyInt_FromSsize_t(i);
            if (unlikely(!key)) return NULL;
            r = mm->mp_subscript(o, key);
            Py_DECREF(key);
            return r;
        }
        if (likely(sm && sm->sq_item)) {
            if (wraparound && unlikely(i < 0) && likely(sm->sq_length)) {
                Py_ssize_t l = sm->sq_length(o);
                if (likely(l >= 0)) {
                    i += l;
                } else {
                    // if length > max(Py_ssize_t), maybe the object can wrap around itself?
                    if (!PyErr_ExceptionMatches(PyExc_OverflowError))
                        return NULL;
                    PyErr_Clear();
                }
            }
            return sm->sq_item(o, i);
        }
    }
#else
    // PySequence_GetItem behaves differently to PyObject_GetItem for i<0
    // and possibly some other cases so can't generally be substituted
    if (is_list || !PyMapping_Check(o)) {
        return PySequence_GetItem(o, i);
    }
#endif
    return __Pyx_GetItemInt_Generic(o, PyInt_FromSsize_t(i));
}

/////////////// SetItemInt.proto ///////////////

#define __Pyx_SetItemInt(o, i, v, type, is_signed, to_py_func, is_list, wraparound, boundscheck) \
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ? \
    __Pyx_SetItemInt_Fast(o, (Py_ssize_t)i, v, is_list, wraparound, boundscheck) : \
    (is_list ? (PyErr_SetString(PyExc_IndexError, "list assignment index out of range"), -1) : \
               __Pyx_SetItemInt_Generic(o, to_py_func(i), v)))

static int __Pyx_SetItemInt_Generic(PyObject *o, PyObject *j, PyObject *v);
static CYTHON_INLINE int __Pyx_SetItemInt_Fast(PyObject *o, Py_ssize_t i, PyObject *v,
                                               int is_list, int wraparound, int boundscheck);

/////////////// SetItemInt ///////////////

static int __Pyx_SetItemInt_Generic(PyObject *o, PyObject *j, PyObject *v) {
    int r;
    if (unlikely(!j)) return -1;
    r = PyObject_SetItem(o, j, v);
    Py_DECREF(j);
    return r;
}

static CYTHON_INLINE int __Pyx_SetItemInt_Fast(PyObject *o, Py_ssize_t i, PyObject *v, int is_list,
                                               CYTHON_NCP_UNUSED int wraparound, CYTHON_NCP_UNUSED int boundscheck) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS && CYTHON_USE_TYPE_SLOTS
    if (is_list || PyList_CheckExact(o)) {
        Py_ssize_t n = (!wraparound) ? i : ((likely(i >= 0)) ? i : i + PyList_GET_SIZE(o));
        if ((!boundscheck) || likely(__Pyx_is_valid_index(n, PyList_GET_SIZE(o)))) {
            PyObject* old = PyList_GET_ITEM(o, n);
            Py_INCREF(v);
            PyList_SET_ITEM(o, n, v);
            Py_DECREF(old);
            return 1;
        }
    } else {
        // inlined PySequence_SetItem() + special cased length overflow
        PyMappingMethods *mm = Py_TYPE(o)->tp_as_mapping;
        PySequenceMethods *sm = Py_TYPE(o)->tp_as_sequence;
        if (mm && mm->mp_ass_subscript) {
            int r;
            PyObject *key = PyInt_FromSsize_t(i);
            if (unlikely(!key)) return -1;
            r = mm->mp_ass_subscript(o, key, v);
            Py_DECREF(key);
            return r;
        }
        if (likely(sm && sm->sq_ass_item)) {
            if (wraparound && unlikely(i < 0) && likely(sm->sq_length)) {
                Py_ssize_t l = sm->sq_length(o);
                if (likely(l >= 0)) {
                    i += l;
                } else {
                    // if length > max(Py_ssize_t), maybe the object can wrap around itself?
                    if (!PyErr_ExceptionMatches(PyExc_OverflowError))
                        return -1;
                    PyErr_Clear();
                }
            }
            return sm->sq_ass_item(o, i, v);
        }
    }
#else
    // PySequence_SetItem behaves differently to PyObject_SetItem for i<0
    // and possibly some other cases so can't generally be substituted
    if (is_list || !PyMapping_Check(o))
    {
        return PySequence_SetItem(o, i, v);
    }
#endif
    return __Pyx_SetItemInt_Generic(o, PyInt_FromSsize_t(i), v);
}


/////////////// DelItemInt.proto ///////////////

#define __Pyx_DelItemInt(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck) \
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ? \
    __Pyx_DelItemInt_Fast(o, (Py_ssize_t)i, is_list, wraparound) : \
    (is_list ? (PyErr_SetString(PyExc_IndexError, "list assignment index out of range"), -1) : \
               __Pyx_DelItem_Generic(o, to_py_func(i))))

static int __Pyx_DelItem_Generic(PyObject *o, PyObject *j);
static CYTHON_INLINE int __Pyx_DelItemInt_Fast(PyObject *o, Py_ssize_t i,
                                               int is_list, int wraparound);

/////////////// DelItemInt ///////////////

static int __Pyx_DelItem_Generic(PyObject *o, PyObject *j) {
    int r;
    if (unlikely(!j)) return -1;
    r = PyObject_DelItem(o, j);
    Py_DECREF(j);
    return r;
}

static CYTHON_INLINE int __Pyx_DelItemInt_Fast(PyObject *o, Py_ssize_t i,
                                               int is_list, CYTHON_NCP_UNUSED int wraparound) {
#if !CYTHON_USE_TYPE_SLOTS
    // PySequence_DelItem behaves differently to PyObject_DelItem for i<0
    // and possibly some other cases so can't generally be substituted
    if (is_list || !PyMapping_Check(o)) {
        return PySequence_DelItem(o, i);
    }
#else
    // inlined PySequence_DelItem() + special cased length overflow
    PyMappingMethods *mm = Py_TYPE(o)->tp_as_mapping;
    PySequenceMethods *sm = Py_TYPE(o)->tp_as_sequence;
    if ((!is_list) && mm && mm->mp_ass_subscript) {
        PyObject *key = PyInt_FromSsize_t(i);
        return likely(key) ? mm->mp_ass_subscript(o, key, (PyObject *)NULL) : -1;
    }
    if (likely(sm && sm->sq_ass_item)) {
        if (wraparound && unlikely(i < 0) && likely(sm->sq_length)) {
            Py_ssize_t l = sm->sq_length(o);
            if (likely(l >= 0)) {
                i += l;
            } else {
                // if length > max(Py_ssize_t), maybe the object can wrap around itself?
                if (!PyErr_ExceptionMatches(PyExc_OverflowError))
                    return -1;
                PyErr_Clear();
            }
        }
        return sm->sq_ass_item(o, i, (PyObject *)NULL);
    }
#endif
    return __Pyx_DelItem_Generic(o, PyInt_FromSsize_t(i));
}


/////////////// SliceObject.proto ///////////////

// we pass pointer addresses to show the C compiler what is NULL and what isn't
{{if access == 'Get'}}
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetSlice(
        PyObject* obj, Py_ssize_t cstart, Py_ssize_t cstop,
        PyObject** py_start, PyObject** py_stop, PyObject** py_slice,
        int has_cstart, int has_cstop, int wraparound);
{{else}}
#define __Pyx_PyObject_DelSlice(obj, cstart, cstop, py_start, py_stop, py_slice, has_cstart, has_cstop, wraparound) \
    __Pyx_PyObject_SetSlice(obj, (PyObject*)NULL, cstart, cstop, py_start, py_stop, py_slice, has_cstart, has_cstop, wraparound)

// we pass pointer addresses to show the C compiler what is NULL and what isn't
static CYTHON_INLINE int __Pyx_PyObject_SetSlice(
        PyObject* obj, PyObject* value, Py_ssize_t cstart, Py_ssize_t cstop,
        PyObject** py_start, PyObject** py_stop, PyObject** py_slice,
        int has_cstart, int has_cstop, int wraparound);
{{endif}}

/////////////// SliceObject ///////////////

{{if access == 'Get'}}
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetSlice(PyObject* obj,
{{else}}
static CYTHON_INLINE int __Pyx_PyObject_SetSlice(PyObject* obj, PyObject* value,
{{endif}}
        Py_ssize_t cstart, Py_ssize_t cstop,
        PyObject** _py_start, PyObject** _py_stop, PyObject** _py_slice,
        int has_cstart, int has_cstop, int wraparound) {
    __Pyx_TypeName obj_type_name;
#if CYTHON_USE_TYPE_SLOTS
    PyMappingMethods* mp;
#if PY_MAJOR_VERSION < 3
    PySequenceMethods* ms = Py_TYPE(obj)->tp_as_sequence;
    if (likely(ms && ms->sq_{{if access == 'Set'}}ass_{{endif}}slice)) {
        if (!has_cstart) {
            if (_py_start && (*_py_start != Py_None)) {
                cstart = __Pyx_PyIndex_AsSsize_t(*_py_start);
                if ((cstart == (Py_ssize_t)-1) && PyErr_Occurred()) goto bad;
            } else
                cstart = 0;
        }
        if (!has_cstop) {
            if (_py_stop && (*_py_stop != Py_None)) {
                cstop = __Pyx_PyIndex_AsSsize_t(*_py_stop);
                if ((cstop == (Py_ssize_t)-1) && PyErr_Occurred()) goto bad;
            } else
                cstop = PY_SSIZE_T_MAX;
        }
        if (wraparound && unlikely((cstart < 0) | (cstop < 0)) && likely(ms->sq_length)) {
            Py_ssize_t l = ms->sq_length(obj);
            if (likely(l >= 0)) {
                if (cstop < 0) {
                    cstop += l;
                    if (cstop < 0) cstop = 0;
                }
                if (cstart < 0) {
                    cstart += l;
                    if (cstart < 0) cstart = 0;
                }
            } else {
                // if length > max(Py_ssize_t), maybe the object can wrap around itself?
                if (!PyErr_ExceptionMatches(PyExc_OverflowError))
                    goto bad;
                PyErr_Clear();
            }
        }
{{if access == 'Get'}}
        return ms->sq_slice(obj, cstart, cstop);
{{else}}
        return ms->sq_ass_slice(obj, cstart, cstop, value);
{{endif}}
    }
#else
    CYTHON_UNUSED_VAR(wraparound);
#endif

    mp = Py_TYPE(obj)->tp_as_mapping;
{{if access == 'Get'}}
    if (likely(mp && mp->mp_subscript))
{{else}}
    if (likely(mp && mp->mp_ass_subscript))
{{endif}}
#else
    CYTHON_UNUSED_VAR(wraparound);
#endif
    {
        {{if access == 'Get'}}PyObject*{{else}}int{{endif}} result;
        PyObject *py_slice, *py_start, *py_stop;
        if (_py_slice) {
            py_slice = *_py_slice;
        } else {
            PyObject* owned_start = NULL;
            PyObject* owned_stop = NULL;
            if (_py_start) {
                py_start = *_py_start;
            } else {
                if (has_cstart) {
                    owned_start = py_start = PyInt_FromSsize_t(cstart);
                    if (unlikely(!py_start)) goto bad;
                } else
                    py_start = Py_None;
            }
            if (_py_stop) {
                py_stop = *_py_stop;
            } else {
                if (has_cstop) {
                    owned_stop = py_stop = PyInt_FromSsize_t(cstop);
                    if (unlikely(!py_stop)) {
                        Py_XDECREF(owned_start);
                        goto bad;
                    }
                } else
                    py_stop = Py_None;
            }
            py_slice = PySlice_New(py_start, py_stop, Py_None);
            Py_XDECREF(owned_start);
            Py_XDECREF(owned_stop);
            if (unlikely(!py_slice)) goto bad;
        }
#if CYTHON_USE_TYPE_SLOTS
{{if access == 'Get'}}
        result = mp->mp_subscript(obj, py_slice);
#else
        result = PyObject_GetItem(obj, py_slice);
{{else}}
        result = mp->mp_ass_subscript(obj, py_slice, value);
#else
        result = value ? PyObject_SetItem(obj, py_slice, value) : PyObject_DelItem(obj, py_slice);
{{endif}}
#endif
        if (!_py_slice) {
            Py_DECREF(py_slice);
        }
        return result;
    }
    obj_type_name = __Pyx_PyType_GetName(Py_TYPE(obj));
    PyErr_Format(PyExc_TypeError,
{{if access == 'Get'}}
        "'" __Pyx_FMT_TYPENAME "' object is unsliceable", obj_type_name);
{{else}}
        "'" __Pyx_FMT_TYPENAME "' object does not support slice %.10s",
        obj_type_name, value ? "assignment" : "deletion");
{{endif}}
    __Pyx_DECREF_TypeName(obj_type_name);

bad:
    return {{if access == 'Get'}}NULL{{else}}-1{{endif}};
}


/////////////// TupleAndListFromArray.proto ///////////////

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyList_FromArray(PyObject *const *src, Py_ssize_t n);
static CYTHON_INLINE PyObject* __Pyx_PyTuple_FromArray(PyObject *const *src, Py_ssize_t n);
#endif

/////////////// TupleAndListFromArray ///////////////
//@substitute: naming

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE void __Pyx_copy_object_array(PyObject *const *CYTHON_RESTRICT src, PyObject** CYTHON_RESTRICT dest, Py_ssize_t length) {
    PyObject *v;
    Py_ssize_t i;
    for (i = 0; i < length; i++) {
        v = dest[i] = src[i];
        Py_INCREF(v);
    }
}

static CYTHON_INLINE PyObject *
__Pyx_PyTuple_FromArray(PyObject *const *src, Py_ssize_t n)
{
    PyObject *res;
    if (n <= 0) {
        Py_INCREF($empty_tuple);
        return $empty_tuple;
    }
    res = PyTuple_New(n);
    if (unlikely(res == NULL)) return NULL;
    __Pyx_copy_object_array(src, ((PyTupleObject*)res)->ob_item, n);
    return res;
}

static CYTHON_INLINE PyObject *
__Pyx_PyList_FromArray(PyObject *const *src, Py_ssize_t n)
{
    PyObject *res;
    if (n <= 0) {
        return PyList_New(0);
    }
    res = PyList_New(n);
    if (unlikely(res == NULL)) return NULL;
    __Pyx_copy_object_array(src, ((PyListObject*)res)->ob_item, n);
    return res;
}
#endif


/////////////// SliceTupleAndList.proto ///////////////

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyList_GetSlice(PyObject* src, Py_ssize_t start, Py_ssize_t stop);
static CYTHON_INLINE PyObject* __Pyx_PyTuple_GetSlice(PyObject* src, Py_ssize_t start, Py_ssize_t stop);
#else
#define __Pyx_PyList_GetSlice(seq, start, stop)   PySequence_GetSlice(seq, start, stop)
#define __Pyx_PyTuple_GetSlice(seq, start, stop)  PySequence_GetSlice(seq, start, stop)
#endif

/////////////// SliceTupleAndList ///////////////
//@requires: TupleAndListFromArray
//@substitute: tempita

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE void __Pyx_crop_slice(Py_ssize_t* _start, Py_ssize_t* _stop, Py_ssize_t* _length) {
    Py_ssize_t start = *_start, stop = *_stop, length = *_length;
    if (start < 0) {
        start += length;
        if (start < 0)
            start = 0;
    }

    if (stop < 0)
        stop += length;
    else if (stop > length)
        stop = length;

    *_length = stop - start;
    *_start = start;
    *_stop = stop;
}

{{for type in ['List', 'Tuple']}}
static CYTHON_INLINE PyObject* __Pyx_Py{{type}}_GetSlice(
            PyObject* src, Py_ssize_t start, Py_ssize_t stop) {
    Py_ssize_t length = Py{{type}}_GET_SIZE(src);
    __Pyx_crop_slice(&start, &stop, &length);
{{if type=='List'}}
    if (length <= 0) {
        // Avoid undefined behaviour when accessing `ob_item` of an empty list.
        return PyList_New(0);
    }
{{endif}}
    return __Pyx_Py{{type}}_FromArray(((Py{{type}}Object*)src)->ob_item + start, length);
}
{{endfor}}
#endif


/////////////// CalculateMetaclass.proto ///////////////

static PyObject *__Pyx_CalculateMetaclass(PyTypeObject *metaclass, PyObject *bases);

/////////////// CalculateMetaclass ///////////////

static PyObject *__Pyx_CalculateMetaclass(PyTypeObject *metaclass, PyObject *bases) {
    Py_ssize_t i, nbases;
#if CYTHON_ASSUME_SAFE_MACROS
    nbases = PyTuple_GET_SIZE(bases);
#else
    nbases = PyTuple_Size(bases);
    if (nbases < 0) return NULL;
#endif
    for (i=0; i < nbases; i++) {
        PyTypeObject *tmptype;
#if CYTHON_ASSUME_SAFE_MACROS
        PyObject *tmp = PyTuple_GET_ITEM(bases, i);
#else
        PyObject *tmp = PyTuple_GetItem(bases, i);
        if (!tmp) return NULL;
#endif
        tmptype = Py_TYPE(tmp);
#if PY_MAJOR_VERSION < 3
        if (tmptype == &PyClass_Type)
            continue;
#endif
        if (!metaclass) {
            metaclass = tmptype;
            continue;
        }
        if (PyType_IsSubtype(metaclass, tmptype))
            continue;
        if (PyType_IsSubtype(tmptype, metaclass)) {
            metaclass = tmptype;
            continue;
        }
        // else:
        PyErr_SetString(PyExc_TypeError,
                        "metaclass conflict: "
                        "the metaclass of a derived class "
                        "must be a (non-strict) subclass "
                        "of the metaclasses of all its bases");
        return NULL;
    }
    if (!metaclass) {
#if PY_MAJOR_VERSION < 3
        metaclass = &PyClass_Type;
#else
        metaclass = &PyType_Type;
#endif
    }
    // make owned reference
    Py_INCREF((PyObject*) metaclass);
    return (PyObject*) metaclass;
}


/////////////// FindInheritedMetaclass.proto ///////////////

static PyObject *__Pyx_FindInheritedMetaclass(PyObject *bases); /*proto*/

/////////////// FindInheritedMetaclass ///////////////
//@requires: PyObjectGetAttrStr
//@requires: CalculateMetaclass

static PyObject *__Pyx_FindInheritedMetaclass(PyObject *bases) {
    PyObject *metaclass;
    if (PyTuple_Check(bases) && PyTuple_GET_SIZE(bases) > 0) {
        PyTypeObject *metatype;
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        PyObject *base = PyTuple_GET_ITEM(bases, 0);
#else
        PyObject *base = PySequence_ITEM(bases, 0);
#endif
#if PY_MAJOR_VERSION < 3
        PyObject* basetype = __Pyx_PyObject_GetAttrStr(base, PYIDENT("__class__"));
        if (basetype) {
            metatype = (PyType_Check(basetype)) ? ((PyTypeObject*) basetype) : NULL;
        } else {
            PyErr_Clear();
            metatype = Py_TYPE(base);
            basetype = (PyObject*) metatype;
            Py_INCREF(basetype);
        }
#else
        metatype = Py_TYPE(base);
#endif
        metaclass = __Pyx_CalculateMetaclass(metatype, bases);
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
        Py_DECREF(base);
#endif
#if PY_MAJOR_VERSION < 3
        Py_DECREF(basetype);
#endif
    } else {
        // no bases => use default metaclass
#if PY_MAJOR_VERSION < 3
        metaclass = (PyObject *) &PyClass_Type;
#else
        metaclass = (PyObject *) &PyType_Type;
#endif
        Py_INCREF(metaclass);
    }
    return metaclass;
}

/////////////// Py3MetaclassGet.proto ///////////////

static PyObject *__Pyx_Py3MetaclassGet(PyObject *bases, PyObject *mkw); /*proto*/

/////////////// Py3MetaclassGet ///////////////
//@requires: FindInheritedMetaclass
//@requires: CalculateMetaclass

static PyObject *__Pyx_Py3MetaclassGet(PyObject *bases, PyObject *mkw) {
    PyObject *metaclass = mkw ? __Pyx_PyDict_GetItemStr(mkw, PYIDENT("metaclass")) : NULL;
    if (metaclass) {
        Py_INCREF(metaclass);
        if (PyDict_DelItem(mkw, PYIDENT("metaclass")) < 0) {
            Py_DECREF(metaclass);
            return NULL;
        }
        if (PyType_Check(metaclass)) {
            PyObject* orig = metaclass;
            metaclass = __Pyx_CalculateMetaclass((PyTypeObject*) metaclass, bases);
            Py_DECREF(orig);
        }
        return metaclass;
    }
    return __Pyx_FindInheritedMetaclass(bases);
}

/////////////// CreateClass.proto ///////////////

static PyObject *__Pyx_CreateClass(PyObject *bases, PyObject *dict, PyObject *name,
                                   PyObject *qualname, PyObject *modname); /*proto*/

/////////////// CreateClass ///////////////
//@requires: FindInheritedMetaclass
//@requires: CalculateMetaclass

static PyObject *__Pyx_CreateClass(PyObject *bases, PyObject *dict, PyObject *name,
                                   PyObject *qualname, PyObject *modname) {
    PyObject *result;
    PyObject *metaclass;

    if (unlikely(PyDict_SetItem(dict, PYIDENT("__module__"), modname) < 0))
        return NULL;
#if PY_VERSION_HEX >= 0x03030000
    if (unlikely(PyDict_SetItem(dict, PYIDENT("__qualname__"), qualname) < 0))
        return NULL;
#else
    CYTHON_MAYBE_UNUSED_VAR(qualname);
#endif

    /* Python2 __metaclass__ */
    metaclass = __Pyx_PyDict_GetItemStr(dict, PYIDENT("__metaclass__"));
    if (metaclass) {
        Py_INCREF(metaclass);
        if (PyType_Check(metaclass)) {
            PyObject* orig = metaclass;
            metaclass = __Pyx_CalculateMetaclass((PyTypeObject*) metaclass, bases);
            Py_DECREF(orig);
        }
    } else {
        metaclass = __Pyx_FindInheritedMetaclass(bases);
    }
    if (unlikely(!metaclass))
        return NULL;
    result = PyObject_CallFunctionObjArgs(metaclass, name, bases, dict, NULL);
    Py_DECREF(metaclass);
    return result;
}

/////////////// Py3UpdateBases.proto ///////////////

static PyObject* __Pyx_PEP560_update_bases(PyObject *bases); /* proto */

/////////////// Py3UpdateBases /////////////////////
//@requires: PyObjectCallOneArg
//@requires: PyObjectGetAttrStrNoError

/* Shamelessly adapted from cpython/bltinmodule.c update_bases */
static PyObject*
__Pyx_PEP560_update_bases(PyObject *bases)
{
    Py_ssize_t i, j, size_bases;
    PyObject *base, *meth, *new_base, *result, *new_bases = NULL;
    /*assert(PyTuple_Check(bases));*/

    size_bases = PyTuple_GET_SIZE(bases);
    for (i = 0; i < size_bases; i++) {
        // original code in CPython: base  = args[i];
        base  = PyTuple_GET_ITEM(bases, i);
        if (PyType_Check(base)) {
            if (new_bases) {
                // If we already have made a replacement, then we append every normal base,
                // otherwise just skip it.
                if (PyList_Append(new_bases, base) < 0) {
                    goto error;
                }
            }
            continue;
        }
        // original code in CPython:
        // if (_PyObject_LookupAttrId(base, &PyId___mro_entries__, &meth) < 0) {
        meth = __Pyx_PyObject_GetAttrStrNoError(base, PYIDENT("__mro_entries__"));
        if (!meth && PyErr_Occurred()) {
            goto error;
        }
        if (!meth) {
            if (new_bases) {
                if (PyList_Append(new_bases, base) < 0) {
                    goto error;
                }
            }
            continue;
        }
        new_base = __Pyx_PyObject_CallOneArg(meth, bases);
        Py_DECREF(meth);
        if (!new_base) {
            goto error;
        }
        if (!PyTuple_Check(new_base)) {
            PyErr_SetString(PyExc_TypeError,
                            "__mro_entries__ must return a tuple");
            Py_DECREF(new_base);
            goto error;
        }
        if (!new_bases) {
            // If this is a first successful replacement, create new_bases list and
            // copy previously encountered bases.
            if (!(new_bases = PyList_New(i))) {
                goto error;
            }
            for (j = 0; j < i; j++) {
                // original code in CPython: base = args[j];
                base = PyTuple_GET_ITEM(bases, j);
                PyList_SET_ITEM(new_bases, j, base);
                Py_INCREF(base);
            }
        }
        j = PyList_GET_SIZE(new_bases);
        if (PyList_SetSlice(new_bases, j, j, new_base) < 0) {
            goto error;
        }
        Py_DECREF(new_base);
    }
    if (!new_bases) {
        // unlike the CPython implementation, always return a new reference
        Py_INCREF(bases);
        return bases;
    }
    result = PyList_AsTuple(new_bases);
    Py_DECREF(new_bases);
    return result;

error:
    Py_XDECREF(new_bases);
    return NULL;
}

/////////////// Py3ClassCreate.proto ///////////////

static PyObject *__Pyx_Py3MetaclassPrepare(PyObject *metaclass, PyObject *bases, PyObject *name, PyObject *qualname,
                                           PyObject *mkw, PyObject *modname, PyObject *doc); /*proto*/
static PyObject *__Pyx_Py3ClassCreate(PyObject *metaclass, PyObject *name, PyObject *bases, PyObject *dict,
                                      PyObject *mkw, int calculate_metaclass, int allow_py2_metaclass); /*proto*/

/////////////// Py3ClassCreate ///////////////
//@substitute: naming
//@requires: PyObjectGetAttrStrNoError
//@requires: CalculateMetaclass
//@requires: PyObjectFastCall
//@requires: PyObjectCall2Args
//@requires: PyObjectLookupSpecial
// only in fallback code:
//@requires: GetBuiltinName

static PyObject *__Pyx_Py3MetaclassPrepare(PyObject *metaclass, PyObject *bases, PyObject *name,
                                           PyObject *qualname, PyObject *mkw, PyObject *modname, PyObject *doc) {
    PyObject *ns;
    if (metaclass) {
        PyObject *prep = __Pyx_PyObject_GetAttrStrNoError(metaclass, PYIDENT("__prepare__"));
        if (prep) {
            PyObject *pargs[3] = {NULL, name, bases};
            ns = __Pyx_PyObject_FastCallDict(prep, pargs+1, 2 | __Pyx_PY_VECTORCALL_ARGUMENTS_OFFSET, mkw);
            Py_DECREF(prep);
        } else {
            if (unlikely(PyErr_Occurred()))
                return NULL;
            ns = PyDict_New();
        }
    } else {
        ns = PyDict_New();
    }

    if (unlikely(!ns))
        return NULL;

    /* Required here to emulate assignment order */
    if (unlikely(PyObject_SetItem(ns, PYIDENT("__module__"), modname) < 0)) goto bad;
#if PY_VERSION_HEX >= 0x03030000
    if (unlikely(PyObject_SetItem(ns, PYIDENT("__qualname__"), qualname) < 0)) goto bad;
#else
    CYTHON_MAYBE_UNUSED_VAR(qualname);
#endif
    if (unlikely(doc && PyObject_SetItem(ns, PYIDENT("__doc__"), doc) < 0)) goto bad;
    return ns;
bad:
    Py_DECREF(ns);
    return NULL;
}

#if PY_VERSION_HEX < 0x030600A4 && CYTHON_PEP487_INIT_SUBCLASS
// https://www.python.org/dev/peps/pep-0487/
static int __Pyx_SetNamesPEP487(PyObject *type_obj) {
    PyTypeObject *type = (PyTypeObject*) type_obj;
    PyObject *names_to_set, *key, *value, *set_name, *tmp;
    Py_ssize_t i = 0;

#if CYTHON_USE_TYPE_SLOTS
    names_to_set = PyDict_Copy(type->tp_dict);
#else
    {
        PyObject *d = PyObject_GetAttr(type_obj, PYIDENT("__dict__"));
        names_to_set = NULL;
        if (likely(d)) {
            // d may not be a dict, e.g. PyDictProxy in PyPy2.
            PyObject *names_to_set = PyDict_New();
            int ret = likely(names_to_set) ? PyDict_Update(names_to_set, d) : -1;
            Py_DECREF(d);
            if (unlikely(ret < 0))
                Py_CLEAR(names_to_set);
        }
    }
#endif
    if (unlikely(names_to_set == NULL))
        goto bad;

    while (PyDict_Next(names_to_set, &i, &key, &value)) {
        set_name = __Pyx_PyObject_LookupSpecialNoError(value, PYIDENT("__set_name__"));
        if (unlikely(set_name != NULL)) {
            tmp = __Pyx_PyObject_Call2Args(set_name, type_obj, key);
            Py_DECREF(set_name);
            if (unlikely(tmp == NULL)) {
                __Pyx_TypeName value_type_name =
                    __Pyx_PyType_GetName(Py_TYPE(value));
                __Pyx_TypeName type_name = __Pyx_PyType_GetName(type);
                PyErr_Format(PyExc_RuntimeError,
#if PY_MAJOR_VERSION >= 3
                    "Error calling __set_name__ on '" __Pyx_FMT_TYPENAME "' instance %R " "in '" __Pyx_FMT_TYPENAME "'",
                    value_type_name, key, type_name);
#else
                    "Error calling __set_name__ on '" __Pyx_FMT_TYPENAME "' instance %.100s in '" __Pyx_FMT_TYPENAME "'",
                    value_type_name,
                    PyString_Check(key) ? PyString_AS_STRING(key) : "?",
                    type_name);
#endif
                goto bad;
            } else {
                Py_DECREF(tmp);
            }
        }
        else if (unlikely(PyErr_Occurred())) {
            goto bad;
        }
    }

    Py_DECREF(names_to_set);
    return 0;
bad:
    Py_XDECREF(names_to_set);
    return -1;
}

static PyObject *__Pyx_InitSubclassPEP487(PyObject *type_obj, PyObject *mkw) {
#if CYTHON_USE_TYPE_SLOTS && CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
// Stripped-down version of "super(type_obj, type_obj).__init_subclass__(**mkw)" in CPython 3.8.
    PyTypeObject *type = (PyTypeObject*) type_obj;
    PyObject *mro = type->tp_mro;
    Py_ssize_t i, nbases;
    if (unlikely(!mro)) goto done;

    // avoid "unused" warning
    (void) &__Pyx_GetBuiltinName;

    Py_INCREF(mro);
    nbases = PyTuple_GET_SIZE(mro);

    // Skip over the type itself and 'object'.
    assert(PyTuple_GET_ITEM(mro, 0) == type_obj);
    for (i = 1; i < nbases-1; i++) {
        PyObject *base, *dict, *meth;
        base = PyTuple_GET_ITEM(mro, i);
        dict = ((PyTypeObject *)base)->tp_dict;
        meth = __Pyx_PyDict_GetItemStrWithError(dict, PYIDENT("__init_subclass__"));
        if (unlikely(meth)) {
            descrgetfunc f = Py_TYPE(meth)->tp_descr_get;
            PyObject *res;
            Py_INCREF(meth);
            if (likely(f)) {
                res = f(meth, NULL, type_obj);
                Py_DECREF(meth);
                if (unlikely(!res)) goto bad;
                meth = res;
            }
            res = __Pyx_PyObject_FastCallDict(meth, NULL, 0, mkw);
            Py_DECREF(meth);
            if (unlikely(!res)) goto bad;
            Py_DECREF(res);
            goto done;
        } else if (unlikely(PyErr_Occurred())) {
            goto bad;
        }
    }

done:
    Py_XDECREF(mro);
    return type_obj;

bad:
    Py_XDECREF(mro);
    Py_DECREF(type_obj);
    return NULL;

// CYTHON_USE_TYPE_SLOTS && CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
#else
// Generic fallback: "super(type_obj, type_obj).__init_subclass__(**mkw)", as used in CPython 3.8.
    PyObject *super_type, *super, *func, *res;

#if CYTHON_COMPILING_IN_PYPY && !defined(PySuper_Type)
    super_type = __Pyx_GetBuiltinName(PYIDENT("super"));
#else
    super_type = (PyObject*) &PySuper_Type;
    // avoid "unused" warning
    (void) &__Pyx_GetBuiltinName;
#endif
    super = likely(super_type) ? __Pyx_PyObject_Call2Args(super_type, type_obj, type_obj) : NULL;
#if CYTHON_COMPILING_IN_PYPY && !defined(PySuper_Type)
    Py_XDECREF(super_type);
#endif
    if (unlikely(!super)) {
        Py_CLEAR(type_obj);
        goto done;
    }
    func = __Pyx_PyObject_GetAttrStrNoError(super, PYIDENT("__init_subclass__"));
    Py_DECREF(super);
    if (likely(!func)) {
        if (unlikely(PyErr_Occurred()))
            Py_CLEAR(type_obj);
        goto done;
    }
    res = __Pyx_PyObject_FastCallDict(func, NULL, 0, mkw);
    Py_DECREF(func);
    if (unlikely(!res))
        Py_CLEAR(type_obj);
    Py_XDECREF(res);
done:
    return type_obj;
#endif
}

// PY_VERSION_HEX < 0x030600A4 && CYTHON_PEP487_INIT_SUBCLASS
#endif

static PyObject *__Pyx_Py3ClassCreate(PyObject *metaclass, PyObject *name, PyObject *bases,
                                      PyObject *dict, PyObject *mkw,
                                      int calculate_metaclass, int allow_py2_metaclass) {
    PyObject *result;
    PyObject *owned_metaclass = NULL;
    PyObject *margs[4] = {NULL, name, bases, dict};
    if (allow_py2_metaclass) {
        /* honour Python2 __metaclass__ for backward compatibility */
        owned_metaclass = PyObject_GetItem(dict, PYIDENT("__metaclass__"));
        if (owned_metaclass) {
            metaclass = owned_metaclass;
        } else if (likely(PyErr_ExceptionMatches(PyExc_KeyError))) {
            PyErr_Clear();
        } else {
            return NULL;
        }
    }
    if (calculate_metaclass && (!metaclass || PyType_Check(metaclass))) {
        metaclass = __Pyx_CalculateMetaclass((PyTypeObject*) metaclass, bases);
        Py_XDECREF(owned_metaclass);
        if (unlikely(!metaclass))
            return NULL;
        owned_metaclass = metaclass;
    }
    result = __Pyx_PyObject_FastCallDict(metaclass, margs+1, 3 | __Pyx_PY_VECTORCALL_ARGUMENTS_OFFSET,
#if PY_VERSION_HEX < 0x030600A4
        // Before PEP-487, type(a,b,c) did not accept any keyword arguments, so guard at least against that case.
        (metaclass == (PyObject*)&PyType_Type) ? NULL : mkw
#else
        mkw
#endif
    );
    Py_XDECREF(owned_metaclass);

#if PY_VERSION_HEX < 0x030600A4 && CYTHON_PEP487_INIT_SUBCLASS
    if (likely(result) && likely(PyType_Check(result))) {
        if (unlikely(__Pyx_SetNamesPEP487(result) < 0)) {
            Py_CLEAR(result);
        } else {
            result = __Pyx_InitSubclassPEP487(result, mkw);
        }
    }
#else
    // avoid "unused" warning
    (void) &__Pyx_GetBuiltinName;
#endif
    return result;
}

/////////////// ExtTypeTest.proto ///////////////

static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type); /*proto*/

/////////////// ExtTypeTest ///////////////

static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type) {
    __Pyx_TypeName obj_type_name;
    __Pyx_TypeName type_name;
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (likely(__Pyx_TypeCheck(obj, type)))
        return 1;
    obj_type_name = __Pyx_PyType_GetName(Py_TYPE(obj));
    type_name = __Pyx_PyType_GetName(type);
    PyErr_Format(PyExc_TypeError,
                 "Cannot convert " __Pyx_FMT_TYPENAME " to " __Pyx_FMT_TYPENAME,
                 obj_type_name, type_name);
    __Pyx_DECREF_TypeName(obj_type_name);
    __Pyx_DECREF_TypeName(type_name);
    return 0;
}

/////////////// CallableCheck.proto ///////////////

#if CYTHON_USE_TYPE_SLOTS && PY_MAJOR_VERSION >= 3
#define __Pyx_PyCallable_Check(obj)   (Py_TYPE(obj)->tp_call != NULL)
#else
#define __Pyx_PyCallable_Check(obj)   PyCallable_Check(obj)
#endif

/////////////// PyDictContains.proto ///////////////

static CYTHON_INLINE int __Pyx_PyDict_ContainsTF(PyObject* item, PyObject* dict, int eq) {
    int result = PyDict_Contains(dict, item);
    return unlikely(result < 0) ? result : (result == (eq == Py_EQ));
}

/////////////// PySetContains.proto ///////////////

static CYTHON_INLINE int __Pyx_PySet_ContainsTF(PyObject* key, PyObject* set, int eq); /* proto */

/////////////// PySetContains ///////////////
//@requires: Builtins.c::pyfrozenset_new

static int __Pyx_PySet_ContainsUnhashable(PyObject *set, PyObject *key) {
    int result = -1;
    if (PySet_Check(key) && PyErr_ExceptionMatches(PyExc_TypeError)) {
        /* Convert key to frozenset */
        PyObject *tmpkey;
        PyErr_Clear();
        tmpkey = __Pyx_PyFrozenSet_New(key);
        if (tmpkey != NULL) {
            result = PySet_Contains(set, tmpkey);
            Py_DECREF(tmpkey);
        }
    }
    return result;
}

static CYTHON_INLINE int __Pyx_PySet_ContainsTF(PyObject* key, PyObject* set, int eq) {
    int result = PySet_Contains(set, key);

    if (unlikely(result < 0)) {
        result = __Pyx_PySet_ContainsUnhashable(set, key);
    }
    return unlikely(result < 0) ? result : (result == (eq == Py_EQ));
}

/////////////// PySequenceContains.proto ///////////////

static CYTHON_INLINE int __Pyx_PySequence_ContainsTF(PyObject* item, PyObject* seq, int eq) {
    int result = PySequence_Contains(seq, item);
    return unlikely(result < 0) ? result : (result == (eq == Py_EQ));
}

/////////////// PyBoolOrNullFromLong.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_PyBoolOrNull_FromLong(long b) {
    return unlikely(b < 0) ? NULL : __Pyx_PyBool_FromLong(b);
}

/////////////// GetBuiltinName.proto ///////////////

static PyObject *__Pyx_GetBuiltinName(PyObject *name); /*proto*/

/////////////// GetBuiltinName ///////////////
//@requires: PyObjectGetAttrStrNoError
//@substitute: naming

static PyObject *__Pyx_GetBuiltinName(PyObject *name) {
    PyObject* result = __Pyx_PyObject_GetAttrStrNoError($builtins_cname, name);
    if (unlikely(!result) && !PyErr_Occurred()) {
        PyErr_Format(PyExc_NameError,
#if PY_MAJOR_VERSION >= 3
            "name '%U' is not defined", name);
#else
            "name '%.200s' is not defined", PyString_AS_STRING(name));
#endif
    }
    return result;
}

/////////////// GetNameInClass.proto ///////////////

#define __Pyx_GetNameInClass(var, nmspace, name)  (var) = __Pyx__GetNameInClass(nmspace, name)
static PyObject *__Pyx__GetNameInClass(PyObject *nmspace, PyObject *name); /*proto*/

/////////////// GetNameInClass ///////////////
//@requires: GetModuleGlobalName

static PyObject *__Pyx__GetNameInClass(PyObject *nmspace, PyObject *name) {
    PyObject *result;
    PyObject *dict;
    assert(PyType_Check(nmspace));
#if CYTHON_USE_TYPE_SLOTS
    dict = ((PyTypeObject*)nmspace)->tp_dict;
    Py_XINCREF(dict);
#else
    dict = PyObject_GetAttr(nmspace, PYIDENT("__dict__"));
#endif
    if (likely(dict)) {
        result = PyObject_GetItem(dict, name);
        Py_DECREF(dict);
        if (result) {
            return result;
        }
    }
    PyErr_Clear();
    __Pyx_GetModuleGlobalNameUncached(result, name);
    return result;
}


/////////////// SetNameInClass.proto ///////////////

#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030500A1 && PY_VERSION_HEX < 0x030d0000
// Identifier names are always interned and have a pre-calculated hash value.
#define __Pyx_SetNameInClass(ns, name, value) \
    (likely(PyDict_CheckExact(ns)) ? _PyDict_SetItem_KnownHash(ns, name, value, ((PyASCIIObject *) name)->hash) : PyObject_SetItem(ns, name, value))
#elif CYTHON_COMPILING_IN_CPYTHON
#define __Pyx_SetNameInClass(ns, name, value) \
    (likely(PyDict_CheckExact(ns)) ? PyDict_SetItem(ns, name, value) : PyObject_SetItem(ns, name, value))
#else
#define __Pyx_SetNameInClass(ns, name, value)  PyObject_SetItem(ns, name, value)
#endif

/////////////// SetNewInClass.proto ///////////////

static int __Pyx_SetNewInClass(PyObject *ns, PyObject *name, PyObject *value);

/////////////// SetNewInClass ///////////////
//@requires: SetNameInClass

// Special-case setting __new__: if it's a Cython function, wrap it in a
// staticmethod. This is similar to what Python does for a Python function
// called __new__.
static int __Pyx_SetNewInClass(PyObject *ns, PyObject *name, PyObject *value) {
#ifdef __Pyx_CyFunction_USED
    int ret;
    if (__Pyx_CyFunction_Check(value)) {
        PyObject *staticnew = PyStaticMethod_New(value);
        if (unlikely(!staticnew)) return -1;
        ret = __Pyx_SetNameInClass(ns, name, staticnew);
        Py_DECREF(staticnew);
        return ret;
    }
#endif
    return __Pyx_SetNameInClass(ns, name, value);
}


/////////////// GetModuleGlobalName.proto ///////////////
//@requires: PyDictVersioning
//@substitute: naming

#if CYTHON_USE_DICT_VERSIONS
#define __Pyx_GetModuleGlobalName(var, name)  do { \
    static PY_UINT64_T __pyx_dict_version = 0; \
    static PyObject *__pyx_dict_cached_value = NULL; \
    (var) = (likely(__pyx_dict_version == __PYX_GET_DICT_VERSION($moddict_cname))) ? \
        (likely(__pyx_dict_cached_value) ? __Pyx_NewRef(__pyx_dict_cached_value) : __Pyx_GetBuiltinName(name)) : \
        __Pyx__GetModuleGlobalName(name, &__pyx_dict_version, &__pyx_dict_cached_value); \
} while(0)
#define __Pyx_GetModuleGlobalNameUncached(var, name)  do { \
    PY_UINT64_T __pyx_dict_version; \
    PyObject *__pyx_dict_cached_value; \
    (var) = __Pyx__GetModuleGlobalName(name, &__pyx_dict_version, &__pyx_dict_cached_value); \
} while(0)
static PyObject *__Pyx__GetModuleGlobalName(PyObject *name, PY_UINT64_T *dict_version, PyObject **dict_cached_value); /*proto*/
#else
#define __Pyx_GetModuleGlobalName(var, name)  (var) = __Pyx__GetModuleGlobalName(name)
#define __Pyx_GetModuleGlobalNameUncached(var, name)  (var) = __Pyx__GetModuleGlobalName(name)
static CYTHON_INLINE PyObject *__Pyx__GetModuleGlobalName(PyObject *name); /*proto*/
#endif


/////////////// GetModuleGlobalName ///////////////
//@requires: GetBuiltinName
//@substitute: naming

#if CYTHON_USE_DICT_VERSIONS
static PyObject *__Pyx__GetModuleGlobalName(PyObject *name, PY_UINT64_T *dict_version, PyObject **dict_cached_value)
#else
static CYTHON_INLINE PyObject *__Pyx__GetModuleGlobalName(PyObject *name)
#endif
{
    PyObject *result;
// FIXME: clean up the macro guard order here: limited API first, then borrowed refs, then cpython
#if !CYTHON_AVOID_BORROWED_REFS
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030500A1 && PY_VERSION_HEX < 0x030d0000
    // Identifier names are always interned and have a pre-calculated hash value.
    result = _PyDict_GetItem_KnownHash($moddict_cname, name, ((PyASCIIObject *) name)->hash);
    __PYX_UPDATE_DICT_CACHE($moddict_cname, result, *dict_cached_value, *dict_version)
    if (likely(result)) {
        return __Pyx_NewRef(result);
    } else if (unlikely(PyErr_Occurred())) {
        return NULL;
    }
#elif CYTHON_COMPILING_IN_LIMITED_API
    if (unlikely(!$module_cname)) {
        return NULL;
    }
    result = PyObject_GetAttr($module_cname, name);
    if (likely(result)) {
        return result;
    }
#else
    result = PyDict_GetItem($moddict_cname, name);
    __PYX_UPDATE_DICT_CACHE($moddict_cname, result, *dict_cached_value, *dict_version)
    if (likely(result)) {
        return __Pyx_NewRef(result);
    }
#endif
#else
    result = PyObject_GetItem($moddict_cname, name);
    __PYX_UPDATE_DICT_CACHE($moddict_cname, result, *dict_cached_value, *dict_version)
    if (likely(result)) {
        return __Pyx_NewRef(result);
    }
    PyErr_Clear();
#endif
    return __Pyx_GetBuiltinName(name);
}

//////////////////// GetAttr.proto ////////////////////

static CYTHON_INLINE PyObject *__Pyx_GetAttr(PyObject *, PyObject *); /*proto*/

//////////////////// GetAttr ////////////////////
//@requires: PyObjectGetAttrStr

static CYTHON_INLINE PyObject *__Pyx_GetAttr(PyObject *o, PyObject *n) {
#if CYTHON_USE_TYPE_SLOTS
#if PY_MAJOR_VERSION >= 3
    if (likely(PyUnicode_Check(n)))
#else
    if (likely(PyString_Check(n)))
#endif
        return __Pyx_PyObject_GetAttrStr(o, n);
#endif
    return PyObject_GetAttr(o, n);
}


/////////////// PyObjectLookupSpecial.proto ///////////////

#if CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS
#define __Pyx_PyObject_LookupSpecialNoError(obj, attr_name)  __Pyx__PyObject_LookupSpecial(obj, attr_name, 0)
#define __Pyx_PyObject_LookupSpecial(obj, attr_name)  __Pyx__PyObject_LookupSpecial(obj, attr_name, 1)

static CYTHON_INLINE PyObject* __Pyx__PyObject_LookupSpecial(PyObject* obj, PyObject* attr_name, int with_error); /*proto*/

#else
#define __Pyx_PyObject_LookupSpecialNoError(o,n) __Pyx_PyObject_GetAttrStrNoError(o,n)
#define __Pyx_PyObject_LookupSpecial(o,n) __Pyx_PyObject_GetAttrStr(o,n)
#endif

/////////////// PyObjectLookupSpecial ///////////////
//@requires: PyObjectGetAttrStr
//@requires: PyObjectGetAttrStrNoError

#if CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PyObject* __Pyx__PyObject_LookupSpecial(PyObject* obj, PyObject* attr_name, int with_error) {
    PyObject *res;
    PyTypeObject *tp = Py_TYPE(obj);
#if PY_MAJOR_VERSION < 3
    if (unlikely(PyInstance_Check(obj)))
        return with_error ? __Pyx_PyObject_GetAttrStr(obj, attr_name) : __Pyx_PyObject_GetAttrStrNoError(obj, attr_name);
#endif
    // adapted from CPython's special_lookup() in ceval.c
    res = _PyType_Lookup(tp, attr_name);
    if (likely(res)) {
        descrgetfunc f = Py_TYPE(res)->tp_descr_get;
        if (!f) {
            Py_INCREF(res);
        } else {
            res = f(res, obj, (PyObject *)tp);
        }
    } else if (with_error) {
        PyErr_SetObject(PyExc_AttributeError, attr_name);
    }
    return res;
}
#endif


/////////////// PyObject_GenericGetAttrNoDict.proto ///////////////

// Setting "tp_getattro" to anything but "PyObject_GenericGetAttr" disables fast method calls in Py3.7.
#if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000
static CYTHON_INLINE PyObject* __Pyx_PyObject_GenericGetAttrNoDict(PyObject* obj, PyObject* attr_name);
#else
// No-args macro to allow function pointer assignment.
#define __Pyx_PyObject_GenericGetAttrNoDict PyObject_GenericGetAttr
#endif

/////////////// PyObject_GenericGetAttrNoDict ///////////////

#if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000

static PyObject *__Pyx_RaiseGenericGetAttributeError(PyTypeObject *tp, PyObject *attr_name) {
    __Pyx_TypeName type_name = __Pyx_PyType_GetName(tp);
    PyErr_Format(PyExc_AttributeError,
#if PY_MAJOR_VERSION >= 3
                 "'" __Pyx_FMT_TYPENAME "' object has no attribute '%U'",
                 type_name, attr_name);
#else
                 "'" __Pyx_FMT_TYPENAME "' object has no attribute '%.400s'",
                 type_name, PyString_AS_STRING(attr_name));
#endif
    __Pyx_DECREF_TypeName(type_name);
    return NULL;
}

static CYTHON_INLINE PyObject* __Pyx_PyObject_GenericGetAttrNoDict(PyObject* obj, PyObject* attr_name) {
    // Copied and adapted from _PyObject_GenericGetAttrWithDict() in CPython 3.6/3.7.
    // To be used in the "tp_getattro" slot of extension types that have no instance dict and cannot be subclassed.
    PyObject *descr;
    PyTypeObject *tp = Py_TYPE(obj);

    if (unlikely(!PyString_Check(attr_name))) {
        return PyObject_GenericGetAttr(obj, attr_name);
    }

    assert(!tp->tp_dictoffset);
    descr = _PyType_Lookup(tp, attr_name);
    if (unlikely(!descr)) {
        return __Pyx_RaiseGenericGetAttributeError(tp, attr_name);
    }

    Py_INCREF(descr);

    #if PY_MAJOR_VERSION < 3
    if (likely(PyType_HasFeature(Py_TYPE(descr), Py_TPFLAGS_HAVE_CLASS)))
    #endif
    {
        descrgetfunc f = Py_TYPE(descr)->tp_descr_get;
        // Optimise for the non-descriptor case because it is faster.
        if (unlikely(f)) {
            PyObject *res = f(descr, obj, (PyObject *)tp);
            Py_DECREF(descr);
            return res;
        }
    }
    return descr;
}
#endif


/////////////// PyObject_GenericGetAttr.proto ///////////////

// Setting "tp_getattro" to anything but "PyObject_GenericGetAttr" disables fast method calls in Py3.7.
#if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000
static PyObject* __Pyx_PyObject_GenericGetAttr(PyObject* obj, PyObject* attr_name);
#else
// No-args macro to allow function pointer assignment.
#define __Pyx_PyObject_GenericGetAttr PyObject_GenericGetAttr
#endif

/////////////// PyObject_GenericGetAttr ///////////////
//@requires: PyObject_GenericGetAttrNoDict

#if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000
static PyObject* __Pyx_PyObject_GenericGetAttr(PyObject* obj, PyObject* attr_name) {
    if (unlikely(Py_TYPE(obj)->tp_dictoffset)) {
        return PyObject_GenericGetAttr(obj, attr_name);
    }
    return __Pyx_PyObject_GenericGetAttrNoDict(obj, attr_name);
}
#endif


/////////////// PyObjectGetAttrStrNoError.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStrNoError(PyObject* obj, PyObject* attr_name);/*proto*/

/////////////// PyObjectGetAttrStrNoError ///////////////
//@requires: PyObjectGetAttrStr
//@requires: Exceptions.c::PyThreadStateGet
//@requires: Exceptions.c::PyErrFetchRestore
//@requires: Exceptions.c::PyErrExceptionMatches

#if __PYX_LIMITED_VERSION_HEX < 0x030d00A1
static void __Pyx_PyObject_GetAttrStr_ClearAttributeError(void) {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    if (likely(__Pyx_PyErr_ExceptionMatches(PyExc_AttributeError)))
        __Pyx_PyErr_Clear();
}
#endif

static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStrNoError(PyObject* obj, PyObject* attr_name) {
    PyObject *result;
#if __PYX_LIMITED_VERSION_HEX >= 0x030d00A1
    (void) PyObject_GetOptionalAttr(obj, attr_name, &result);
    return result;
#else
#if CYTHON_COMPILING_IN_CPYTHON && CYTHON_USE_TYPE_SLOTS && PY_VERSION_HEX >= 0x030700B1
    // _PyObject_GenericGetAttrWithDict() in CPython 3.7+ can avoid raising the AttributeError.
    // See https://bugs.python.org/issue32544
    PyTypeObject* tp = Py_TYPE(obj);
    if (likely(tp->tp_getattro == PyObject_GenericGetAttr)) {
        return _PyObject_GenericGetAttrWithDict(obj, attr_name, NULL, 1);
    }
#endif
    result = __Pyx_PyObject_GetAttrStr(obj, attr_name);
    if (unlikely(!result)) {
        __Pyx_PyObject_GetAttrStr_ClearAttributeError();
    }
    return result;
#endif
}


/////////////// PyObjectGetAttrStr.proto ///////////////

#if CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStr(PyObject* obj, PyObject* attr_name);/*proto*/
#else
#define __Pyx_PyObject_GetAttrStr(o,n) PyObject_GetAttr(o,n)
#endif

/////////////// PyObjectGetAttrStr ///////////////

#if CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStr(PyObject* obj, PyObject* attr_name) {
    PyTypeObject* tp = Py_TYPE(obj);
    if (likely(tp->tp_getattro))
        return tp->tp_getattro(obj, attr_name);
#if PY_MAJOR_VERSION < 3
    if (likely(tp->tp_getattr))
        return tp->tp_getattr(obj, PyString_AS_STRING(attr_name));
#endif
    return PyObject_GetAttr(obj, attr_name);
}
#endif


/////////////// PyObjectSetAttrStr.proto ///////////////

#if CYTHON_USE_TYPE_SLOTS
#define __Pyx_PyObject_DelAttrStr(o,n) __Pyx_PyObject_SetAttrStr(o, n, NULL)
static CYTHON_INLINE int __Pyx_PyObject_SetAttrStr(PyObject* obj, PyObject* attr_name, PyObject* value);/*proto*/
#else
#define __Pyx_PyObject_DelAttrStr(o,n)   PyObject_DelAttr(o,n)
#define __Pyx_PyObject_SetAttrStr(o,n,v) PyObject_SetAttr(o,n,v)
#endif

/////////////// PyObjectSetAttrStr ///////////////

#if CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE int __Pyx_PyObject_SetAttrStr(PyObject* obj, PyObject* attr_name, PyObject* value) {
    PyTypeObject* tp = Py_TYPE(obj);
    if (likely(tp->tp_setattro))
        return tp->tp_setattro(obj, attr_name, value);
#if PY_MAJOR_VERSION < 3
    if (likely(tp->tp_setattr))
        return tp->tp_setattr(obj, PyString_AS_STRING(attr_name), value);
#endif
    return PyObject_SetAttr(obj, attr_name, value);
}
#endif


/////////////// PyObjectGetMethod.proto ///////////////

static int __Pyx_PyObject_GetMethod(PyObject *obj, PyObject *name, PyObject **method);/*proto*/

/////////////// PyObjectGetMethod ///////////////
//@requires: PyObjectGetAttrStr

static int __Pyx_PyObject_GetMethod(PyObject *obj, PyObject *name, PyObject **method) {
    PyObject *attr;
#if CYTHON_UNPACK_METHODS && CYTHON_COMPILING_IN_CPYTHON && CYTHON_USE_PYTYPE_LOOKUP
    __Pyx_TypeName type_name;
    // Copied from _PyObject_GetMethod() in CPython 3.7
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *descr;
    descrgetfunc f = NULL;
    PyObject **dictptr, *dict;
    int meth_found = 0;

    assert (*method == NULL);

    if (unlikely(tp->tp_getattro != PyObject_GenericGetAttr)) {
        attr = __Pyx_PyObject_GetAttrStr(obj, name);
        goto try_unpack;
    }
    if (unlikely(tp->tp_dict == NULL) && unlikely(PyType_Ready(tp) < 0)) {
        return 0;
    }

    descr = _PyType_Lookup(tp, name);
    if (likely(descr != NULL)) {
        Py_INCREF(descr);
#if defined(Py_TPFLAGS_METHOD_DESCRIPTOR) && Py_TPFLAGS_METHOD_DESCRIPTOR
        if (__Pyx_PyType_HasFeature(Py_TYPE(descr), Py_TPFLAGS_METHOD_DESCRIPTOR))
#elif PY_MAJOR_VERSION >= 3
        // Repeating the condition below accommodates for MSVC's inability to test macros inside of macro expansions.
        #ifdef __Pyx_CyFunction_USED
        if (likely(PyFunction_Check(descr) || __Pyx_IS_TYPE(descr, &PyMethodDescr_Type) || __Pyx_CyFunction_Check(descr)))
        #else
        if (likely(PyFunction_Check(descr) || __Pyx_IS_TYPE(descr, &PyMethodDescr_Type)))
        #endif
#else
        // "PyMethodDescr_Type" is not part of the C-API in Py2.
        #ifdef __Pyx_CyFunction_USED
        if (likely(PyFunction_Check(descr) || __Pyx_CyFunction_Check(descr)))
        #else
        if (likely(PyFunction_Check(descr)))
        #endif
#endif
        {
            meth_found = 1;
        } else {
            f = Py_TYPE(descr)->tp_descr_get;
            if (f != NULL && PyDescr_IsData(descr)) {
                attr = f(descr, obj, (PyObject *)Py_TYPE(obj));
                Py_DECREF(descr);
                goto try_unpack;
            }
        }
    }

    dictptr = _PyObject_GetDictPtr(obj);
    if (dictptr != NULL && (dict = *dictptr) != NULL) {
        Py_INCREF(dict);
        attr = __Pyx_PyDict_GetItemStr(dict, name);
        if (attr != NULL) {
            Py_INCREF(attr);
            Py_DECREF(dict);
            Py_XDECREF(descr);
            goto try_unpack;
        }
        Py_DECREF(dict);
    }

    if (meth_found) {
        *method = descr;
        return 1;
    }

    if (f != NULL) {
        attr = f(descr, obj, (PyObject *)Py_TYPE(obj));
        Py_DECREF(descr);
        goto try_unpack;
    }

    if (likely(descr != NULL)) {
        *method = descr;
        return 0;
    }

    type_name = __Pyx_PyType_GetName(tp);
    PyErr_Format(PyExc_AttributeError,
#if PY_MAJOR_VERSION >= 3
                 "'" __Pyx_FMT_TYPENAME "' object has no attribute '%U'",
                 type_name, name);
#else
                 "'" __Pyx_FMT_TYPENAME "' object has no attribute '%.400s'",
                 type_name, PyString_AS_STRING(name));
#endif
    __Pyx_DECREF_TypeName(type_name);
    return 0;

// Generic fallback implementation using normal attribute lookup.
#else
    attr = __Pyx_PyObject_GetAttrStr(obj, name);
    goto try_unpack;
#endif

try_unpack:
#if CYTHON_UNPACK_METHODS
    // Even if we failed to avoid creating a bound method object, it's still worth unpacking it now, if possible.
    if (likely(attr) && PyMethod_Check(attr) && likely(PyMethod_GET_SELF(attr) == obj)) {
        PyObject *function = PyMethod_GET_FUNCTION(attr);
        Py_INCREF(function);
        Py_DECREF(attr);
        *method = function;
        return 1;
    }
#endif
    *method = attr;
    return 0;
}


/////////////// UnpackUnboundCMethod.proto ///////////////

typedef struct {
    PyObject *type;
    PyObject **method_name;
    // "func" is set on first access (direct C function pointer)
    PyCFunction func;
    // "method" is set on first access (fallback)
    PyObject *method;
    int flag;
} __Pyx_CachedCFunction;

/////////////// UnpackUnboundCMethod ///////////////
//@requires: PyObjectGetAttrStr

static PyObject *__Pyx_SelflessCall(PyObject *method, PyObject *args, PyObject *kwargs) {
    // NOTE: possible optimization - use vectorcall
    PyObject *result;
    PyObject *selfless_args = PyTuple_GetSlice(args, 1, PyTuple_Size(args));
    if (unlikely(!selfless_args)) return NULL;

    result = PyObject_Call(method, selfless_args, kwargs);
    Py_DECREF(selfless_args);
    return result;
}

static PyMethodDef __Pyx_UnboundCMethod_Def = {
    /* .ml_name  = */ "CythonUnboundCMethod",
    /* .ml_meth  = */ __PYX_REINTERPRET_FUNCION(PyCFunction, __Pyx_SelflessCall),
    /* .ml_flags = */ METH_VARARGS | METH_KEYWORDS,
    /* .ml_doc   = */ NULL
};

static int __Pyx_TryUnpackUnboundCMethod(__Pyx_CachedCFunction* target) {
    PyObject *method;
    method = __Pyx_PyObject_GetAttrStr(target->type, *target->method_name);
    if (unlikely(!method))
        return -1;
    target->method = method;
// FIXME: use functionality from CythonFunction.c/ClassMethod
#if CYTHON_COMPILING_IN_CPYTHON
    #if PY_MAJOR_VERSION >= 3
    if (likely(__Pyx_TypeCheck(method, &PyMethodDescr_Type)))
    #else
    // method descriptor type isn't exported in Py2.x, cannot easily check the type there.
    // Therefore, reverse the check to the most likely alternative
    // (which is returned for class methods)
    if (likely(!__Pyx_CyOrPyCFunction_Check(method)))
    #endif
    {
        PyMethodDescrObject *descr = (PyMethodDescrObject*) method;
        target->func = descr->d_method->ml_meth;
        target->flag = descr->d_method->ml_flags & ~(METH_CLASS | METH_STATIC | METH_COEXIST | METH_STACKLESS);
    } else
#endif
    // bound classmethods need special treatment
#if CYTHON_COMPILING_IN_PYPY
    // In PyPy, functions are regular methods, so just do the self check.
#else
    if (PyCFunction_Check(method))
#endif
    {
        PyObject *self;
        int self_found;
#if CYTHON_COMPILING_IN_LIMITED_API || CYTHON_COMPILING_IN_PYPY
        self = PyObject_GetAttrString(method, "__self__");
        if (!self) {
            PyErr_Clear();
        }
#else
        self = PyCFunction_GET_SELF(method);
#endif
        self_found = (self && self != Py_None);
#if CYTHON_COMPILING_IN_LIMITED_API || CYTHON_COMPILING_IN_PYPY
        Py_XDECREF(self);
#endif
        if (self_found) {
            PyObject *unbound_method = PyCFunction_New(&__Pyx_UnboundCMethod_Def, method);
            if (unlikely(!unbound_method)) return -1;
            // New PyCFunction will own method reference, thus decref __Pyx_PyObject_GetAttrStr
            Py_DECREF(method);
            target->method = unbound_method;
        }
    }
    return 0;
}


/////////////// CallUnboundCMethod0.proto ///////////////
//@substitute: naming

static PyObject* __Pyx__CallUnboundCMethod0(__Pyx_CachedCFunction* cfunc, PyObject* self); /*proto*/
#if CYTHON_COMPILING_IN_CPYTHON
// FASTCALL methods receive "&empty_tuple" as simple "PyObject[0]*"
#define __Pyx_CallUnboundCMethod0(cfunc, self)  \
    (likely((cfunc)->func) ? \
        (likely((cfunc)->flag == METH_NOARGS) ?  (*((cfunc)->func))(self, NULL) : \
         (PY_VERSION_HEX >= 0x030600B1 && likely((cfunc)->flag == METH_FASTCALL) ? \
            (PY_VERSION_HEX >= 0x030700A0 ? \
                (*(__Pyx_PyCFunctionFast)(void*)(PyCFunction)(cfunc)->func)(self, &$empty_tuple, 0) : \
                (*(__Pyx_PyCFunctionFastWithKeywords)(void*)(PyCFunction)(cfunc)->func)(self, &$empty_tuple, 0, NULL)) : \
          (PY_VERSION_HEX >= 0x030700A0 && (cfunc)->flag == (METH_FASTCALL | METH_KEYWORDS) ? \
            (*(__Pyx_PyCFunctionFastWithKeywords)(void*)(PyCFunction)(cfunc)->func)(self, &$empty_tuple, 0, NULL) : \
            (likely((cfunc)->flag == (METH_VARARGS | METH_KEYWORDS)) ?  ((*(PyCFunctionWithKeywords)(void*)(PyCFunction)(cfunc)->func)(self, $empty_tuple, NULL)) : \
               ((cfunc)->flag == METH_VARARGS ?  (*((cfunc)->func))(self, $empty_tuple) : \
               __Pyx__CallUnboundCMethod0(cfunc, self)))))) : \
        __Pyx__CallUnboundCMethod0(cfunc, self))
#else
#define __Pyx_CallUnboundCMethod0(cfunc, self)  __Pyx__CallUnboundCMethod0(cfunc, self)
#endif

/////////////// CallUnboundCMethod0 ///////////////
//@requires: UnpackUnboundCMethod
//@requires: PyObjectCall

static PyObject* __Pyx__CallUnboundCMethod0(__Pyx_CachedCFunction* cfunc, PyObject* self) {
    PyObject *args, *result = NULL;
    if (unlikely(!cfunc->method) && unlikely(__Pyx_TryUnpackUnboundCMethod(cfunc) < 0)) return NULL;
#if CYTHON_ASSUME_SAFE_MACROS
    args = PyTuple_New(1);
    if (unlikely(!args)) goto bad;
    Py_INCREF(self);
    PyTuple_SET_ITEM(args, 0, self);
#else
    args = PyTuple_Pack(1, self);
    if (unlikely(!args)) goto bad;
#endif
    result = __Pyx_PyObject_Call(cfunc->method, args, NULL);
    Py_DECREF(args);
bad:
    return result;
}


/////////////// CallUnboundCMethod1.proto ///////////////

static PyObject* __Pyx__CallUnboundCMethod1(__Pyx_CachedCFunction* cfunc, PyObject* self, PyObject* arg);/*proto*/

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_CallUnboundCMethod1(__Pyx_CachedCFunction* cfunc, PyObject* self, PyObject* arg);/*proto*/
#else
#define __Pyx_CallUnboundCMethod1(cfunc, self, arg)  __Pyx__CallUnboundCMethod1(cfunc, self, arg)
#endif

/////////////// CallUnboundCMethod1 ///////////////
//@requires: UnpackUnboundCMethod
//@requires: PyObjectCall

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_CallUnboundCMethod1(__Pyx_CachedCFunction* cfunc, PyObject* self, PyObject* arg) {
    if (likely(cfunc->func)) {
        int flag = cfunc->flag;
        // Not using #ifdefs for PY_VERSION_HEX to avoid C compiler warnings about unused functions.
        if (flag == METH_O) {
            return (*(cfunc->func))(self, arg);
        } else if ((PY_VERSION_HEX >= 0x030600B1) && flag == METH_FASTCALL) {
            #if PY_VERSION_HEX >= 0x030700A0
                return (*(__Pyx_PyCFunctionFast)(void*)(PyCFunction)cfunc->func)(self, &arg, 1);
            #else
                return (*(__Pyx_PyCFunctionFastWithKeywords)(void*)(PyCFunction)cfunc->func)(self, &arg, 1, NULL);
            #endif
        } else if ((PY_VERSION_HEX >= 0x030700A0) && flag == (METH_FASTCALL | METH_KEYWORDS)) {
            return (*(__Pyx_PyCFunctionFastWithKeywords)(void*)(PyCFunction)cfunc->func)(self, &arg, 1, NULL);
        }
    }
    return __Pyx__CallUnboundCMethod1(cfunc, self, arg);
}
#endif

static PyObject* __Pyx__CallUnboundCMethod1(__Pyx_CachedCFunction* cfunc, PyObject* self, PyObject* arg){
    PyObject *args, *result = NULL;
    if (unlikely(!cfunc->func && !cfunc->method) && unlikely(__Pyx_TryUnpackUnboundCMethod(cfunc) < 0)) return NULL;
#if CYTHON_COMPILING_IN_CPYTHON
    if (cfunc->func && (cfunc->flag & METH_VARARGS)) {
        args = PyTuple_New(1);
        if (unlikely(!args)) goto bad;
        Py_INCREF(arg);
        PyTuple_SET_ITEM(args, 0, arg);
        if (cfunc->flag & METH_KEYWORDS)
            result = (*(PyCFunctionWithKeywords)(void*)(PyCFunction)cfunc->func)(self, args, NULL);
        else
            result = (*cfunc->func)(self, args);
    } else {
        args = PyTuple_New(2);
        if (unlikely(!args)) goto bad;
        Py_INCREF(self);
        PyTuple_SET_ITEM(args, 0, self);
        Py_INCREF(arg);
        PyTuple_SET_ITEM(args, 1, arg);
        result = __Pyx_PyObject_Call(cfunc->method, args, NULL);
    }
#else
    args = PyTuple_Pack(2, self, arg);
    if (unlikely(!args)) goto bad;
    result = __Pyx_PyObject_Call(cfunc->method, args, NULL);
#endif
bad:
    Py_XDECREF(args);
    return result;
}


/////////////// CallUnboundCMethod2.proto ///////////////

static PyObject* __Pyx__CallUnboundCMethod2(__Pyx_CachedCFunction* cfunc, PyObject* self, PyObject* arg1, PyObject* arg2); /*proto*/

#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030600B1
static CYTHON_INLINE PyObject *__Pyx_CallUnboundCMethod2(__Pyx_CachedCFunction *cfunc, PyObject *self, PyObject *arg1, PyObject *arg2); /*proto*/
#else
#define __Pyx_CallUnboundCMethod2(cfunc, self, arg1, arg2)  __Pyx__CallUnboundCMethod2(cfunc, self, arg1, arg2)
#endif

/////////////// CallUnboundCMethod2 ///////////////
//@requires: UnpackUnboundCMethod
//@requires: PyObjectCall

#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030600B1
static CYTHON_INLINE PyObject *__Pyx_CallUnboundCMethod2(__Pyx_CachedCFunction *cfunc, PyObject *self, PyObject *arg1, PyObject *arg2) {
    if (likely(cfunc->func)) {
        PyObject *args[2] = {arg1, arg2};
        if (cfunc->flag == METH_FASTCALL) {
            #if PY_VERSION_HEX >= 0x030700A0
            return (*(__Pyx_PyCFunctionFast)(void*)(PyCFunction)cfunc->func)(self, args, 2);
            #else
            return (*(__Pyx_PyCFunctionFastWithKeywords)(void*)(PyCFunction)cfunc->func)(self, args, 2, NULL);
            #endif
        }
        #if PY_VERSION_HEX >= 0x030700A0
        if (cfunc->flag == (METH_FASTCALL | METH_KEYWORDS))
            return (*(__Pyx_PyCFunctionFastWithKeywords)(void*)(PyCFunction)cfunc->func)(self, args, 2, NULL);
        #endif
    }
    return __Pyx__CallUnboundCMethod2(cfunc, self, arg1, arg2);
}
#endif

static PyObject* __Pyx__CallUnboundCMethod2(__Pyx_CachedCFunction* cfunc, PyObject* self, PyObject* arg1, PyObject* arg2){
    PyObject *args, *result = NULL;
    if (unlikely(!cfunc->func && !cfunc->method) && unlikely(__Pyx_TryUnpackUnboundCMethod(cfunc) < 0)) return NULL;
#if CYTHON_COMPILING_IN_CPYTHON
    if (cfunc->func && (cfunc->flag & METH_VARARGS)) {
        args = PyTuple_New(2);
        if (unlikely(!args)) goto bad;
        Py_INCREF(arg1);
        PyTuple_SET_ITEM(args, 0, arg1);
        Py_INCREF(arg2);
        PyTuple_SET_ITEM(args, 1, arg2);
        if (cfunc->flag & METH_KEYWORDS)
            result = (*(PyCFunctionWithKeywords)(void*)(PyCFunction)cfunc->func)(self, args, NULL);
        else
            result = (*cfunc->func)(self, args);
    } else {
        args = PyTuple_New(3);
        if (unlikely(!args)) goto bad;
        Py_INCREF(self);
        PyTuple_SET_ITEM(args, 0, self);
        Py_INCREF(arg1);
        PyTuple_SET_ITEM(args, 1, arg1);
        Py_INCREF(arg2);
        PyTuple_SET_ITEM(args, 2, arg2);
        result = __Pyx_PyObject_Call(cfunc->method, args, NULL);
    }
#else
    args = PyTuple_Pack(3, self, arg1, arg2);
    if (unlikely(!args)) goto bad;
    result = __Pyx_PyObject_Call(cfunc->method, args, NULL);
#endif
bad:
    Py_XDECREF(args);
    return result;
}


/////////////// PyObjectFastCall.proto ///////////////

#define __Pyx_PyObject_FastCall(func, args, nargs)  __Pyx_PyObject_FastCallDict(func, args, (size_t)(nargs), NULL)
static CYTHON_INLINE PyObject* __Pyx_PyObject_FastCallDict(PyObject *func, PyObject **args, size_t nargs, PyObject *kwargs); /*proto*/

/////////////// PyObjectFastCall ///////////////
//@requires: PyObjectCall
//@requires: PyFunctionFastCall
//@requires: PyObjectCallMethO
//@substitute: naming

#if PY_VERSION_HEX < 0x03090000 || CYTHON_COMPILING_IN_LIMITED_API
static PyObject* __Pyx_PyObject_FastCall_fallback(PyObject *func, PyObject **args, size_t nargs, PyObject *kwargs) {
    PyObject *argstuple;
    PyObject *result = 0;
    size_t i;

    argstuple = PyTuple_New((Py_ssize_t)nargs);
    if (unlikely(!argstuple)) return NULL;
    for (i = 0; i < nargs; i++) {
        Py_INCREF(args[i]);
        if (__Pyx_PyTuple_SET_ITEM(argstuple, (Py_ssize_t)i, args[i]) < 0) goto bad;
    }
    result = __Pyx_PyObject_Call(func, argstuple, kwargs);
  bad:
    Py_DECREF(argstuple);
    return result;
}
#endif

static CYTHON_INLINE PyObject* __Pyx_PyObject_FastCallDict(PyObject *func, PyObject **args, size_t _nargs, PyObject *kwargs) {
    // Special fast paths for 0 and 1 arguments
    // NOTE: in many cases, this is called with a constant value for nargs
    // which is known at compile-time. So the branches below will typically
    // be optimized away.
    Py_ssize_t nargs = __Pyx_PyVectorcall_NARGS(_nargs);
#if CYTHON_COMPILING_IN_CPYTHON
    if (nargs == 0 && kwargs == NULL) {
        if (__Pyx_CyOrPyCFunction_Check(func) && likely( __Pyx_CyOrPyCFunction_GET_FLAGS(func) & METH_NOARGS))
            return __Pyx_PyObject_CallMethO(func, NULL);
    }
    else if (nargs == 1 && kwargs == NULL) {
        if (__Pyx_CyOrPyCFunction_Check(func) && likely( __Pyx_CyOrPyCFunction_GET_FLAGS(func) & METH_O))
            return __Pyx_PyObject_CallMethO(func, args[0]);
    }
#endif

    #if PY_VERSION_HEX < 0x030800B1
    #if CYTHON_FAST_PYCCALL
    if (PyCFunction_Check(func)) {
        if (kwargs) {
            return _PyCFunction_FastCallDict(func, args, nargs, kwargs);
        } else {
            return _PyCFunction_FastCallKeywords(func, args, nargs, NULL);
        }
    }
    #if PY_VERSION_HEX >= 0x030700A1
    if (!kwargs && __Pyx_IS_TYPE(func, &PyMethodDescr_Type)) {
        return _PyMethodDescr_FastCallKeywords(func, args, nargs, NULL);
    }
    #endif
    #endif
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(func)) {
        return __Pyx_PyFunction_FastCallDict(func, args, nargs, kwargs);
    }
    #endif
    #endif

    if (kwargs == NULL) {
        #if CYTHON_VECTORCALL
        #if PY_VERSION_HEX < 0x03090000
        vectorcallfunc f = _PyVectorcall_Function(func);
        #else
        vectorcallfunc f = PyVectorcall_Function(func);
        #endif
        if (f) {
            return f(func, args, (size_t)nargs, NULL);
        }
        #elif defined(__Pyx_CyFunction_USED) && CYTHON_BACKPORT_VECTORCALL
        // exclude fused functions for now
        if (__Pyx_CyFunction_CheckExact(func)) {
            __pyx_vectorcallfunc f = __Pyx_CyFunction_func_vectorcall(func);
            if (f) return f(func, args, (size_t)nargs, NULL);
        }
        #endif
    }

    if (nargs == 0) {
        return __Pyx_PyObject_Call(func, $empty_tuple, kwargs);
    }
    #if PY_VERSION_HEX >= 0x03090000 && !CYTHON_COMPILING_IN_LIMITED_API
    return PyObject_VectorcallDict(func, args, (size_t)nargs, kwargs);
    #else
    return __Pyx_PyObject_FastCall_fallback(func, args, (size_t)nargs, kwargs);
    #endif
}


/////////////// PyObjectCallMethod0.proto ///////////////

static PyObject* __Pyx_PyObject_CallMethod0(PyObject* obj, PyObject* method_name); /*proto*/

/////////////// PyObjectCallMethod0 ///////////////
//@requires: PyObjectGetMethod
//@requires: PyObjectCallOneArg
//@requires: PyObjectCallNoArg

static PyObject* __Pyx_PyObject_CallMethod0(PyObject* obj, PyObject* method_name) {
    PyObject *method = NULL, *result = NULL;
    int is_method = __Pyx_PyObject_GetMethod(obj, method_name, &method);
    if (likely(is_method)) {
        result = __Pyx_PyObject_CallOneArg(method, obj);
        Py_DECREF(method);
        return result;
    }
    if (unlikely(!method)) goto bad;
    result = __Pyx_PyObject_CallNoArg(method);
    Py_DECREF(method);
bad:
    return result;
}


/////////////// PyObjectCallMethod1.proto ///////////////

static PyObject* __Pyx_PyObject_CallMethod1(PyObject* obj, PyObject* method_name, PyObject* arg); /*proto*/

/////////////// PyObjectCallMethod1 ///////////////
//@requires: PyObjectGetMethod
//@requires: PyObjectCallOneArg
//@requires: PyObjectCall2Args

#if !(CYTHON_VECTORCALL && __PYX_LIMITED_VERSION_HEX >= 0x030C00A2)
static PyObject* __Pyx__PyObject_CallMethod1(PyObject* method, PyObject* arg) {
    // Separate function to avoid excessive inlining.
    PyObject *result = __Pyx_PyObject_CallOneArg(method, arg);
    Py_DECREF(method);
    return result;
}
#endif

static PyObject* __Pyx_PyObject_CallMethod1(PyObject* obj, PyObject* method_name, PyObject* arg) {
#if CYTHON_VECTORCALL && __PYX_LIMITED_VERSION_HEX >= 0x030C00A2
    PyObject *args[2] = {obj, arg};
    // avoid unused functions
    (void) __Pyx_PyObject_GetMethod;
    (void) __Pyx_PyObject_CallOneArg;
    (void) __Pyx_PyObject_Call2Args;
    return PyObject_VectorcallMethod(method_name, args, 2 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
#else
    PyObject *method = NULL, *result;
    int is_method = __Pyx_PyObject_GetMethod(obj, method_name, &method);
    if (likely(is_method)) {
        result = __Pyx_PyObject_Call2Args(method, obj, arg);
        Py_DECREF(method);
        return result;
    }
    if (unlikely(!method)) return NULL;
    return __Pyx__PyObject_CallMethod1(method, arg);
#endif
}


/////////////// tp_new.proto ///////////////

#define __Pyx_tp_new(type_obj, args) __Pyx_tp_new_kwargs(type_obj, args, NULL)
static CYTHON_INLINE PyObject* __Pyx_tp_new_kwargs(PyObject* type_obj, PyObject* args, PyObject* kwargs) {
    return (PyObject*) (((PyTypeObject*)type_obj)->tp_new((PyTypeObject*)type_obj, args, kwargs));
}


/////////////// PyObjectCall.proto ///////////////

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw); /*proto*/
#else
#define __Pyx_PyObject_Call(func, arg, kw) PyObject_Call(func, arg, kw)
#endif

/////////////// PyObjectCall ///////////////

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    PyObject *result;
    ternaryfunc call = Py_TYPE(func)->tp_call;

    if (unlikely(!call))
        return PyObject_Call(func, arg, kw);
    #if PY_MAJOR_VERSION < 3
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object")))
        return NULL;
    #else
    if (unlikely(Py_EnterRecursiveCall(" while calling a Python object")))
        return NULL;
    #endif
    result = (*call)(func, arg, kw);
    Py_LeaveRecursiveCall();
    if (unlikely(!result) && unlikely(!PyErr_Occurred())) {
        PyErr_SetString(
            PyExc_SystemError,
            "NULL result without error in PyObject_Call");
    }
    return result;
}
#endif


/////////////// PyObjectCallMethO.proto ///////////////

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallMethO(PyObject *func, PyObject *arg); /*proto*/
#endif

/////////////// PyObjectCallMethO ///////////////

#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallMethO(PyObject *func, PyObject *arg) {
    PyObject *self, *result;
    PyCFunction cfunc;
    cfunc = __Pyx_CyOrPyCFunction_GET_FUNCTION(func);
    self = __Pyx_CyOrPyCFunction_GET_SELF(func);

    #if PY_MAJOR_VERSION < 3
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object")))
        return NULL;
    #else
    if (unlikely(Py_EnterRecursiveCall(" while calling a Python object")))
        return NULL;
    #endif
    result = cfunc(self, arg);
    Py_LeaveRecursiveCall();
    if (unlikely(!result) && unlikely(!PyErr_Occurred())) {
        PyErr_SetString(
            PyExc_SystemError,
            "NULL result without error in PyObject_Call");
    }
    return result;
}
#endif


/////////////// PyFunctionFastCall.proto ///////////////

#if CYTHON_FAST_PYCALL

#if !CYTHON_VECTORCALL
#define __Pyx_PyFunction_FastCall(func, args, nargs) \
    __Pyx_PyFunction_FastCallDict((func), (args), (nargs), NULL)

static PyObject *__Pyx_PyFunction_FastCallDict(PyObject *func, PyObject **args, Py_ssize_t nargs, PyObject *kwargs);
#endif

// Backport from Python 3
// Assert a build-time dependency, as an expression.
//   Your compile will fail if the condition isn't true, or can't be evaluated
//   by the compiler. This can be used in an expression: its value is 0.
// Example:
//   #define foo_to_char(foo)  \
//       ((char *)(foo)        \
//        + Py_BUILD_ASSERT_EXPR(offsetof(struct foo, string) == 0))
//
//   Written by Rusty Russell, public domain, https://ccodearchive.net/
#define __Pyx_BUILD_ASSERT_EXPR(cond) \
    (sizeof(char [1 - 2*!(cond)]) - 1)

#ifndef Py_MEMBER_SIZE
// Get the size of a structure member in bytes
#define Py_MEMBER_SIZE(type, member) sizeof(((type *)0)->member)
#endif

#if !CYTHON_VECTORCALL
#if PY_VERSION_HEX >= 0x03080000
  #include "frameobject.h"
#if PY_VERSION_HEX >= 0x030b00a6 && !CYTHON_COMPILING_IN_LIMITED_API
  #ifndef Py_BUILD_CORE
    #define Py_BUILD_CORE 1
  #endif
  #include "internal/pycore_frame.h"
#endif
  #define __Pxy_PyFrame_Initialize_Offsets()
  #define __Pyx_PyFrame_GetLocalsplus(frame)  ((frame)->f_localsplus)
#else
  // Initialised by module init code.
  static size_t __pyx_pyframe_localsplus_offset = 0;

  #include "frameobject.h"
  // This is the long runtime version of
  //     #define __Pyx_PyFrame_GetLocalsplus(frame)  ((frame)->f_localsplus)
  // offsetof(PyFrameObject, f_localsplus) differs between regular C-Python and Stackless Python < 3.8.
  // Therefore the offset is computed at run time from PyFrame_type.tp_basicsize. That is feasible,
  // because f_localsplus is the last field of PyFrameObject (checked by Py_BUILD_ASSERT_EXPR below).
  #define __Pxy_PyFrame_Initialize_Offsets()  \
    ((void)__Pyx_BUILD_ASSERT_EXPR(sizeof(PyFrameObject) == offsetof(PyFrameObject, f_localsplus) + Py_MEMBER_SIZE(PyFrameObject, f_localsplus)), \
     (void)(__pyx_pyframe_localsplus_offset = ((size_t)PyFrame_Type.tp_basicsize) - Py_MEMBER_SIZE(PyFrameObject, f_localsplus)))
  #define __Pyx_PyFrame_GetLocalsplus(frame)  \
    (assert(__pyx_pyframe_localsplus_offset), (PyObject **)(((char *)(frame)) + __pyx_pyframe_localsplus_offset))
#endif
#endif /* !CYTHON_VECTORCALL */

#endif /* CYTHON_FAST_PYCALL */


/////////////// PyFunctionFastCall ///////////////
// copied from CPython 3.6 ceval.c

#if CYTHON_FAST_PYCALL && !CYTHON_VECTORCALL
static PyObject* __Pyx_PyFunction_FastCallNoKw(PyCodeObject *co, PyObject **args, Py_ssize_t na,
                                               PyObject *globals) {
    PyFrameObject *f;
    PyThreadState *tstate = __Pyx_PyThreadState_Current;
    PyObject **fastlocals;
    Py_ssize_t i;
    PyObject *result;

    assert(globals != NULL);
    /* XXX Perhaps we should create a specialized
       PyFrame_New() that doesn't take locals, but does
       take builtins without sanity checking them.
       */
    assert(tstate != NULL);
    f = PyFrame_New(tstate, co, globals, NULL);
    if (f == NULL) {
        return NULL;
    }

    fastlocals = __Pyx_PyFrame_GetLocalsplus(f);

    for (i = 0; i < na; i++) {
        Py_INCREF(*args);
        fastlocals[i] = *args++;
    }
    result = PyEval_EvalFrameEx(f,0);

    ++tstate->recursion_depth;
    Py_DECREF(f);
    --tstate->recursion_depth;

    return result;
}


static PyObject *__Pyx_PyFunction_FastCallDict(PyObject *func, PyObject **args, Py_ssize_t nargs, PyObject *kwargs) {
    PyCodeObject *co = (PyCodeObject *)PyFunction_GET_CODE(func);
    PyObject *globals = PyFunction_GET_GLOBALS(func);
    PyObject *argdefs = PyFunction_GET_DEFAULTS(func);
    PyObject *closure;
#if PY_MAJOR_VERSION >= 3
    PyObject *kwdefs;
    //#if PY_VERSION_HEX >= 0x03050000
    //PyObject *name, *qualname;
    //#endif
#endif
    PyObject *kwtuple, **k;
    PyObject **d;
    Py_ssize_t nd;
    Py_ssize_t nk;
    PyObject *result;

    assert(kwargs == NULL || PyDict_Check(kwargs));
    nk = kwargs ? PyDict_Size(kwargs) : 0;

    #if PY_MAJOR_VERSION < 3
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object"))) {
        return NULL;
    }
    #else
    if (unlikely(Py_EnterRecursiveCall(" while calling a Python object"))) {
        return NULL;
    }
    #endif

    if (
#if PY_MAJOR_VERSION >= 3
            co->co_kwonlyargcount == 0 &&
#endif
            likely(kwargs == NULL || nk == 0) &&
            co->co_flags == (CO_OPTIMIZED | CO_NEWLOCALS | CO_NOFREE)) {
        /* Fast paths */
        if (argdefs == NULL && co->co_argcount == nargs) {
            result = __Pyx_PyFunction_FastCallNoKw(co, args, nargs, globals);
            goto done;
        }
        else if (nargs == 0 && argdefs != NULL
                 && co->co_argcount == Py_SIZE(argdefs)) {
            /* function called with no arguments, but all parameters have
               a default value: use default values as arguments .*/
            args = &PyTuple_GET_ITEM(argdefs, 0);
            result =__Pyx_PyFunction_FastCallNoKw(co, args, Py_SIZE(argdefs), globals);
            goto done;
        }
    }

    if (kwargs != NULL) {
        Py_ssize_t pos, i;
        kwtuple = PyTuple_New(2 * nk);
        if (kwtuple == NULL) {
            result = NULL;
            goto done;
        }

        k = &PyTuple_GET_ITEM(kwtuple, 0);
        pos = i = 0;
        while (PyDict_Next(kwargs, &pos, &k[i], &k[i+1])) {
            Py_INCREF(k[i]);
            Py_INCREF(k[i+1]);
            i += 2;
        }
        nk = i / 2;
    }
    else {
        kwtuple = NULL;
        k = NULL;
    }

    closure = PyFunction_GET_CLOSURE(func);
#if PY_MAJOR_VERSION >= 3
    kwdefs = PyFunction_GET_KW_DEFAULTS(func);
    //#if PY_VERSION_HEX >= 0x03050000
    //name = ((PyFunctionObject *)func) -> func_name;
    //qualname = ((PyFunctionObject *)func) -> func_qualname;
    //#endif
#endif

    if (argdefs != NULL) {
        d = &PyTuple_GET_ITEM(argdefs, 0);
        nd = Py_SIZE(argdefs);
    }
    else {
        d = NULL;
        nd = 0;
    }

    //#if PY_VERSION_HEX >= 0x03050000
    //return _PyEval_EvalCodeWithName((PyObject*)co, globals, (PyObject *)NULL,
    //                                args, nargs,
    //                                NULL, 0,
    //                                d, nd, kwdefs,
    //                                closure, name, qualname);
    //#elif PY_MAJOR_VERSION >= 3
#if PY_MAJOR_VERSION >= 3
    result = PyEval_EvalCodeEx((PyObject*)co, globals, (PyObject *)NULL,
                               args, (int)nargs,
                               k, (int)nk,
                               d, (int)nd, kwdefs, closure);
#else
    result = PyEval_EvalCodeEx(co, globals, (PyObject *)NULL,
                               args, (int)nargs,
                               k, (int)nk,
                               d, (int)nd, closure);
#endif
    Py_XDECREF(kwtuple);

done:
    Py_LeaveRecursiveCall();
    return result;
}
#endif  /* CYTHON_FAST_PYCALL && !CYTHON_VECTORCALL */


/////////////// PyObjectCall2Args.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_PyObject_Call2Args(PyObject* function, PyObject* arg1, PyObject* arg2); /*proto*/

/////////////// PyObjectCall2Args ///////////////
//@requires: PyObjectFastCall

static CYTHON_INLINE PyObject* __Pyx_PyObject_Call2Args(PyObject* function, PyObject* arg1, PyObject* arg2) {
    PyObject *args[3] = {NULL, arg1, arg2};
    return __Pyx_PyObject_FastCall(function, args+1, 2 | __Pyx_PY_VECTORCALL_ARGUMENTS_OFFSET);
}


/////////////// PyObjectCallOneArg.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg); /*proto*/

/////////////// PyObjectCallOneArg ///////////////
//@requires: PyObjectFastCall

static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg) {
    PyObject *args[2] = {NULL, arg};
    return __Pyx_PyObject_FastCall(func, args+1, 1 | __Pyx_PY_VECTORCALL_ARGUMENTS_OFFSET);
}


/////////////// PyObjectCallNoArg.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_PyObject_CallNoArg(PyObject *func); /*proto*/

/////////////// PyObjectCallNoArg ///////////////
//@requires: PyObjectFastCall

static CYTHON_INLINE PyObject* __Pyx_PyObject_CallNoArg(PyObject *func) {
    PyObject *arg[2] = {NULL, NULL};
    return __Pyx_PyObject_FastCall(func, arg + 1, 0 | __Pyx_PY_VECTORCALL_ARGUMENTS_OFFSET);
}


/////////////// PyVectorcallFastCallDict.proto ///////////////

#if CYTHON_METH_FASTCALL
static CYTHON_INLINE PyObject *__Pyx_PyVectorcall_FastCallDict(PyObject *func, __pyx_vectorcallfunc vc, PyObject *const *args, size_t nargs, PyObject *kw);
#endif

/////////////// PyVectorcallFastCallDict ///////////////

#if CYTHON_METH_FASTCALL
// Slow path when kw is non-empty
static PyObject *__Pyx_PyVectorcall_FastCallDict_kw(PyObject *func, __pyx_vectorcallfunc vc, PyObject *const *args, size_t nargs, PyObject *kw)
{
    // Code based on _PyObject_FastCallDict() and _PyStack_UnpackDict() from CPython
    PyObject *res = NULL;
    PyObject *kwnames;
    PyObject **newargs;
    PyObject **kwvalues;
    Py_ssize_t i, pos;
    size_t j;
    PyObject *key, *value;
    unsigned long keys_are_strings;
    Py_ssize_t nkw = PyDict_GET_SIZE(kw);

    // Copy positional arguments
    newargs = (PyObject **)PyMem_Malloc((nargs + (size_t)nkw) * sizeof(args[0]));
    if (unlikely(newargs == NULL)) {
        PyErr_NoMemory();
        return NULL;
    }
    for (j = 0; j < nargs; j++) newargs[j] = args[j];

    // Copy keyword arguments
    kwnames = PyTuple_New(nkw);
    if (unlikely(kwnames == NULL)) {
        PyMem_Free(newargs);
        return NULL;
    }
    kwvalues = newargs + nargs;
    pos = i = 0;
    keys_are_strings = Py_TPFLAGS_UNICODE_SUBCLASS;
    while (PyDict_Next(kw, &pos, &key, &value)) {
        keys_are_strings &= Py_TYPE(key)->tp_flags;
        Py_INCREF(key);
        Py_INCREF(value);
        PyTuple_SET_ITEM(kwnames, i, key);
        kwvalues[i] = value;
        i++;
    }
    if (unlikely(!keys_are_strings)) {
        PyErr_SetString(PyExc_TypeError, "keywords must be strings");
        goto cleanup;
    }

    // The actual call
    res = vc(func, newargs, nargs, kwnames);

cleanup:
    Py_DECREF(kwnames);
    for (i = 0; i < nkw; i++)
        Py_DECREF(kwvalues[i]);
    PyMem_Free(newargs);
    return res;
}

static CYTHON_INLINE PyObject *__Pyx_PyVectorcall_FastCallDict(PyObject *func, __pyx_vectorcallfunc vc, PyObject *const *args, size_t nargs, PyObject *kw)
{
    if (likely(kw == NULL) || PyDict_GET_SIZE(kw) == 0) {
        return vc(func, args, nargs, NULL);
    }
    return __Pyx_PyVectorcall_FastCallDict_kw(func, vc, args, nargs, kw);
}
#endif


/////////////// MatrixMultiply.proto ///////////////

#if PY_VERSION_HEX >= 0x03050000
  #define __Pyx_PyNumber_MatrixMultiply(x,y)         PyNumber_MatrixMultiply(x,y)
  #define __Pyx_PyNumber_InPlaceMatrixMultiply(x,y)  PyNumber_InPlaceMatrixMultiply(x,y)
#else
#define __Pyx_PyNumber_MatrixMultiply(x,y)         __Pyx__PyNumber_MatrixMultiply(x, y, "@")
static PyObject* __Pyx__PyNumber_MatrixMultiply(PyObject* x, PyObject* y, const char* op_name);
static PyObject* __Pyx_PyNumber_InPlaceMatrixMultiply(PyObject* x, PyObject* y);
#endif

/////////////// MatrixMultiply ///////////////
//@requires: PyObjectGetAttrStrNoError
//@requires: PyObjectCallOneArg
//@requires: PyObjectCall2Args

#if PY_VERSION_HEX < 0x03050000
static PyObject* __Pyx_PyObject_CallMatrixMethod(PyObject* method, PyObject* arg) {
    // NOTE: eats the method reference
    PyObject *result = NULL;
#if CYTHON_UNPACK_METHODS
    if (likely(PyMethod_Check(method))) {
        PyObject *self = PyMethod_GET_SELF(method);
        if (likely(self)) {
            PyObject *function = PyMethod_GET_FUNCTION(method);
            result = __Pyx_PyObject_Call2Args(function, self, arg);
            goto done;
        }
    }
#endif
    result = __Pyx_PyObject_CallOneArg(method, arg);
done:
    Py_DECREF(method);
    return result;
}

#define __Pyx_TryMatrixMethod(x, y, py_method_name) {                     \
    PyObject *func = __Pyx_PyObject_GetAttrStrNoError(x, py_method_name); \
    if (func) {                                                         \
        PyObject *result = __Pyx_PyObject_CallMatrixMethod(func, y);    \
        if (result != Py_NotImplemented)                                \
            return result;                                              \
        Py_DECREF(result);                                              \
    } else if (unlikely(PyErr_Occurred())) {                            \
        return NULL;                                                    \
    }                                                                   \
}

static PyObject* __Pyx__PyNumber_MatrixMultiply(PyObject* x, PyObject* y, const char* op_name) {
    __Pyx_TypeName x_type_name;
    __Pyx_TypeName y_type_name;
    int right_is_subtype = PyObject_IsSubclass((PyObject*)Py_TYPE(y), (PyObject*)Py_TYPE(x));
    if (unlikely(right_is_subtype == -1))
        return NULL;
    if (right_is_subtype) {
        // to allow subtypes to override parent behaviour, try reversed operation first
        // see note at https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
        __Pyx_TryMatrixMethod(y, x, PYIDENT("__rmatmul__"))
    }
    __Pyx_TryMatrixMethod(x, y, PYIDENT("__matmul__"))
    if (!right_is_subtype) {
        __Pyx_TryMatrixMethod(y, x, PYIDENT("__rmatmul__"))
    }
    x_type_name = __Pyx_PyType_GetName(Py_TYPE(x));
    y_type_name = __Pyx_PyType_GetName(Py_TYPE(y));
    PyErr_Format(PyExc_TypeError,
        "unsupported operand type(s) for %.2s: '" __Pyx_FMT_TYPENAME "' and '"
        __Pyx_FMT_TYPENAME "'", op_name, x_type_name, y_type_name);
    __Pyx_DECREF_TypeName(x_type_name);
    __Pyx_DECREF_TypeName(y_type_name);
    return NULL;
}

static PyObject* __Pyx_PyNumber_InPlaceMatrixMultiply(PyObject* x, PyObject* y) {
    __Pyx_TryMatrixMethod(x, y, PYIDENT("__imatmul__"))
    return __Pyx__PyNumber_MatrixMultiply(x, y, "@=");
}

#undef __Pyx_TryMatrixMethod
#endif


/////////////// PyDictVersioning.proto ///////////////

#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_TYPE_SLOTS
#define __PYX_DICT_VERSION_INIT  ((PY_UINT64_T) -1)
#define __PYX_GET_DICT_VERSION(dict)  (((PyDictObject*)(dict))->ma_version_tag)
#define __PYX_UPDATE_DICT_CACHE(dict, value, cache_var, version_var) \
    (version_var) = __PYX_GET_DICT_VERSION(dict); \
    (cache_var) = (value);

#define __PYX_PY_DICT_LOOKUP_IF_MODIFIED(VAR, DICT, LOOKUP) { \
    static PY_UINT64_T __pyx_dict_version = 0; \
    static PyObject *__pyx_dict_cached_value = NULL; \
    if (likely(__PYX_GET_DICT_VERSION(DICT) == __pyx_dict_version)) { \
        (VAR) = __pyx_dict_cached_value; \
    } else { \
        (VAR) = __pyx_dict_cached_value = (LOOKUP); \
        __pyx_dict_version = __PYX_GET_DICT_VERSION(DICT); \
    } \
}

static CYTHON_INLINE PY_UINT64_T __Pyx_get_tp_dict_version(PyObject *obj); /*proto*/
static CYTHON_INLINE PY_UINT64_T __Pyx_get_object_dict_version(PyObject *obj); /*proto*/
static CYTHON_INLINE int __Pyx_object_dict_version_matches(PyObject* obj, PY_UINT64_T tp_dict_version, PY_UINT64_T obj_dict_version); /*proto*/

#else
#define __PYX_GET_DICT_VERSION(dict)  (0)
#define __PYX_UPDATE_DICT_CACHE(dict, value, cache_var, version_var)
#define __PYX_PY_DICT_LOOKUP_IF_MODIFIED(VAR, DICT, LOOKUP)  (VAR) = (LOOKUP);
#endif

/////////////// PyDictVersioning ///////////////

#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PY_UINT64_T __Pyx_get_tp_dict_version(PyObject *obj) {
    PyObject *dict = Py_TYPE(obj)->tp_dict;
    return likely(dict) ? __PYX_GET_DICT_VERSION(dict) : 0;
}

static CYTHON_INLINE PY_UINT64_T __Pyx_get_object_dict_version(PyObject *obj) {
    PyObject **dictptr = NULL;
    Py_ssize_t offset = Py_TYPE(obj)->tp_dictoffset;
    if (offset) {
#if CYTHON_COMPILING_IN_CPYTHON
        dictptr = (likely(offset > 0)) ? (PyObject **) ((char *)obj + offset) : _PyObject_GetDictPtr(obj);
#else
        dictptr = _PyObject_GetDictPtr(obj);
#endif
    }
    return (dictptr && *dictptr) ? __PYX_GET_DICT_VERSION(*dictptr) : 0;
}

static CYTHON_INLINE int __Pyx_object_dict_version_matches(PyObject* obj, PY_UINT64_T tp_dict_version, PY_UINT64_T obj_dict_version) {
    PyObject *dict = Py_TYPE(obj)->tp_dict;
    if (unlikely(!dict) || unlikely(tp_dict_version != __PYX_GET_DICT_VERSION(dict)))
        return 0;
    return obj_dict_version == __Pyx_get_object_dict_version(obj);
}
#endif


/////////////// PyMethodNew.proto ///////////////

#if CYTHON_COMPILING_IN_LIMITED_API
static PyObject *__Pyx_PyMethod_New(PyObject *func, PyObject *self, PyObject *typ) {
    PyObject *typesModule=NULL, *methodType=NULL, *result=NULL;
    CYTHON_UNUSED_VAR(typ);
    if (!self)
        return __Pyx_NewRef(func);
    typesModule = PyImport_ImportModule("types");
    if (!typesModule) return NULL;
    methodType = PyObject_GetAttrString(typesModule, "MethodType");
    Py_DECREF(typesModule);
    if (!methodType) return NULL;
    result = PyObject_CallFunctionObjArgs(methodType, func, self, NULL);
    Py_DECREF(methodType);
    return result;
}
#elif PY_MAJOR_VERSION >= 3
// This should be an actual function (not a macro), such that we can put it
// directly in a tp_descr_get slot.
static PyObject *__Pyx_PyMethod_New(PyObject *func, PyObject *self, PyObject *typ) {
    CYTHON_UNUSED_VAR(typ);
    if (!self)
        return __Pyx_NewRef(func);
    return PyMethod_New(func, self);
}
#else
    #define __Pyx_PyMethod_New PyMethod_New
#endif

///////////// PyMethodNew2Arg.proto /////////////

// Another wrapping of PyMethod_New that matches the Python3 signature
#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyMethod_New2Arg PyMethod_New
#else
#define __Pyx_PyMethod_New2Arg(func, self) PyMethod_New(func, self, (PyObject*)Py_TYPE(self))
#endif

/////////////// UnicodeConcatInPlace.proto ////////////////

# if CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3
// __Pyx_PyUnicode_ConcatInPlace may modify the first argument 'left'
// However, unlike `PyUnicode_Append` it will never NULL it.
// It behaves like a regular function - returns a new reference and NULL on error
    #if CYTHON_REFNANNY
        #define __Pyx_PyUnicode_ConcatInPlace(left, right) __Pyx_PyUnicode_ConcatInPlaceImpl(&left, right, __pyx_refnanny)
    #else
        #define __Pyx_PyUnicode_ConcatInPlace(left, right) __Pyx_PyUnicode_ConcatInPlaceImpl(&left, right)
    #endif
    // __Pyx_PyUnicode_ConcatInPlace is slightly odd because it has the potential to modify the input
    // argument (but only in cases where no user should notice). Therefore, it needs to keep Cython's
    // refnanny informed.
    static CYTHON_INLINE PyObject *__Pyx_PyUnicode_ConcatInPlaceImpl(PyObject **p_left, PyObject *right
        #if CYTHON_REFNANNY
        , void* __pyx_refnanny
        #endif
    ); /* proto */
#else
#define __Pyx_PyUnicode_ConcatInPlace __Pyx_PyUnicode_Concat
#endif
#define __Pyx_PyUnicode_ConcatInPlaceSafe(left, right) ((unlikely((left) == Py_None) || unlikely((right) == Py_None)) ? \
    PyNumber_InPlaceAdd(left, right) : __Pyx_PyUnicode_ConcatInPlace(left, right))

/////////////// UnicodeConcatInPlace ////////////////
//@substitute: naming

# if CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3
// copied directly from unicode_object.c "unicode_modifiable
// removing _PyUnicode_HASH since it's a macro we don't have
//  - this is OK because trying PyUnicode_Resize on a non-modifyable
//  object will still work, it just won't happen in place
static int
__Pyx_unicode_modifiable(PyObject *unicode)
{
    if (Py_REFCNT(unicode) != 1)
        return 0;
    if (!PyUnicode_CheckExact(unicode))
        return 0;
    if (PyUnicode_CHECK_INTERNED(unicode))
        return 0;
    return 1;
}

static CYTHON_INLINE PyObject *__Pyx_PyUnicode_ConcatInPlaceImpl(PyObject **p_left, PyObject *right
        #if CYTHON_REFNANNY
        , void* __pyx_refnanny
        #endif
    ) {
    // heavily based on PyUnicode_Append
    PyObject *left = *p_left;
    Py_ssize_t left_len, right_len, new_len;

    if (unlikely(__Pyx_PyUnicode_READY(left) == -1))
        return NULL;
    if (unlikely(__Pyx_PyUnicode_READY(right) == -1))
        return NULL;

    // Shortcuts
    left_len = PyUnicode_GET_LENGTH(left);
    if (left_len == 0) {
        Py_INCREF(right);
        return right;
    }
    right_len = PyUnicode_GET_LENGTH(right);
    if (right_len == 0) {
        Py_INCREF(left);
        return left;
    }
    if (unlikely(left_len > PY_SSIZE_T_MAX - right_len)) {
        PyErr_SetString(PyExc_OverflowError,
                        "strings are too large to concat");
        return NULL;
    }
    new_len = left_len + right_len;

    if (__Pyx_unicode_modifiable(left)
            && PyUnicode_CheckExact(right)
            && PyUnicode_KIND(right) <= PyUnicode_KIND(left)
            // Don't resize for ascii += latin1. Convert ascii to latin1 requires
            //   to change the structure size, but characters are stored just after
            //   the structure, and so it requires to move all characters which is
            //   not so different than duplicating the string.
            && !(PyUnicode_IS_ASCII(left) && !PyUnicode_IS_ASCII(right))) {

        int ret;
        // GIVEREF/GOTREF since we expect *p_left to change (although it won't change on failures).
        __Pyx_GIVEREF(*p_left);
        ret = PyUnicode_Resize(p_left, new_len);
        __Pyx_GOTREF(*p_left);
        if (unlikely(ret != 0))
            return NULL;

        // copy 'right' into the newly allocated area of 'left'
        #if PY_VERSION_HEX >= 0x030d0000
        if (unlikely(PyUnicode_CopyCharacters(*p_left, left_len, right, 0, right_len) < 0)) return NULL;
        #else
        _PyUnicode_FastCopyCharacters(*p_left, left_len, right, 0, right_len);
        #endif
        __Pyx_INCREF(*p_left);
        __Pyx_GIVEREF(*p_left);
        return *p_left;
    } else {
        return __Pyx_PyUnicode_Concat(left, right);
    }
  }
#endif

////////////// StrConcatInPlace.proto ///////////////////////
//@requires: UnicodeConcatInPlace

#if PY_MAJOR_VERSION >= 3
    // allow access to the more efficient versions where we know str_type is unicode
    #define __Pyx_PyStr_Concat __Pyx_PyUnicode_Concat
    #define __Pyx_PyStr_ConcatInPlace __Pyx_PyUnicode_ConcatInPlace
#else
    #define __Pyx_PyStr_Concat PyNumber_Add
    #define __Pyx_PyStr_ConcatInPlace PyNumber_InPlaceAdd
#endif
#define __Pyx_PyStr_ConcatSafe(a, b) ((unlikely((a) == Py_None) || unlikely((b) == Py_None)) ? \
    PyNumber_Add(a, b) : __Pyx_PyStr_Concat(a, b))
#define __Pyx_PyStr_ConcatInPlaceSafe(a, b) ((unlikely((a) == Py_None) || unlikely((b) == Py_None)) ? \
    PyNumber_InPlaceAdd(a, b) : __Pyx_PyStr_ConcatInPlace(a, b))


/////////////// PySequenceMultiply.proto ///////////////

#define __Pyx_PySequence_Multiply_Left(mul, seq)  __Pyx_PySequence_Multiply(seq, mul)
static CYTHON_INLINE PyObject* __Pyx_PySequence_Multiply(PyObject *seq, Py_ssize_t mul);

/////////////// PySequenceMultiply ///////////////

static PyObject* __Pyx_PySequence_Multiply_Generic(PyObject *seq, Py_ssize_t mul) {
    PyObject *result, *pymul = PyInt_FromSsize_t(mul);
    if (unlikely(!pymul))
        return NULL;
    result = PyNumber_Multiply(seq, pymul);
    Py_DECREF(pymul);
    return result;
}

static CYTHON_INLINE PyObject* __Pyx_PySequence_Multiply(PyObject *seq, Py_ssize_t mul) {
#if CYTHON_USE_TYPE_SLOTS
    PyTypeObject *type = Py_TYPE(seq);
    if (likely(type->tp_as_sequence && type->tp_as_sequence->sq_repeat)) {
        return type->tp_as_sequence->sq_repeat(seq, mul);
    } else
#endif
    {
        return __Pyx_PySequence_Multiply_Generic(seq, mul);
    }
}


/////////////// FormatTypeName.proto ///////////////

#if CYTHON_COMPILING_IN_LIMITED_API
typedef PyObject *__Pyx_TypeName;
#define __Pyx_FMT_TYPENAME "%U"
static __Pyx_TypeName __Pyx_PyType_GetName(PyTypeObject* tp); /*proto*/
#define __Pyx_DECREF_TypeName(obj) Py_XDECREF(obj)
#else
typedef const char *__Pyx_TypeName;
#define __Pyx_FMT_TYPENAME "%.200s"
#define __Pyx_PyType_GetName(tp) ((tp)->tp_name)
#define __Pyx_DECREF_TypeName(obj)
#endif

/////////////// FormatTypeName ///////////////

#if CYTHON_COMPILING_IN_LIMITED_API
static __Pyx_TypeName
__Pyx_PyType_GetName(PyTypeObject* tp)
{
    PyObject *name = __Pyx_PyObject_GetAttrStr((PyObject *)tp,
                                               PYIDENT("__name__"));
    if (unlikely(name == NULL) || unlikely(!PyUnicode_Check(name))) {
        PyErr_Clear();
        Py_XDECREF(name);
        name = __Pyx_NewRef(PYIDENT("?"));
    }
    return name;
}
#endif


/////////////// RaiseUnexpectedTypeError.proto ///////////////

static int __Pyx_RaiseUnexpectedTypeError(const char *expected, PyObject *obj); /*proto*/

/////////////// RaiseUnexpectedTypeError ///////////////

static int
__Pyx_RaiseUnexpectedTypeError(const char *expected, PyObject *obj)
{
    __Pyx_TypeName obj_type_name = __Pyx_PyType_GetName(Py_TYPE(obj));
    PyErr_Format(PyExc_TypeError, "Expected %s, got " __Pyx_FMT_TYPENAME,
                 expected, obj_type_name);
    __Pyx_DECREF_TypeName(obj_type_name);
    return 0;
}


/////////////// RaiseUnboundLocalError.proto ///////////////
static CYTHON_INLINE void __Pyx_RaiseUnboundLocalError(const char *varname);/*proto*/

/////////////// RaiseUnboundLocalError ///////////////
static CYTHON_INLINE void __Pyx_RaiseUnboundLocalError(const char *varname) {
    PyErr_Format(PyExc_UnboundLocalError, "local variable '%s' referenced before assignment", varname);
}


/////////////// RaiseClosureNameError.proto ///////////////
static CYTHON_INLINE void __Pyx_RaiseClosureNameError(const char *varname);/*proto*/

/////////////// RaiseClosureNameError ///////////////
static CYTHON_INLINE void __Pyx_RaiseClosureNameError(const char *varname) {
    PyErr_Format(PyExc_NameError, "free variable '%s' referenced before assignment in enclosing scope", varname);
}


/////////////// RaiseUnboundMemoryviewSliceNogil.proto ///////////////
static void __Pyx_RaiseUnboundMemoryviewSliceNogil(const char *varname);/*proto*/

/////////////// RaiseUnboundMemoryviewSliceNogil ///////////////
//@requires: RaiseUnboundLocalError

// Don't inline the function, it should really never be called in production
static void __Pyx_RaiseUnboundMemoryviewSliceNogil(const char *varname) {
    #ifdef WITH_THREAD
    PyGILState_STATE gilstate = PyGILState_Ensure();
    #endif
    __Pyx_RaiseUnboundLocalError(varname);
    #ifdef WITH_THREAD
    PyGILState_Release(gilstate);
    #endif
}

//////////////// RaiseCppGlobalNameError.proto ///////////////////////
static CYTHON_INLINE void __Pyx_RaiseCppGlobalNameError(const char *varname); /*proto*/

/////////////// RaiseCppGlobalNameError //////////////////////////////
static CYTHON_INLINE void __Pyx_RaiseCppGlobalNameError(const char *varname) {
    PyErr_Format(PyExc_NameError, "C++ global '%s' is not initialized", varname);
}

//////////////// RaiseCppAttributeError.proto ///////////////////////
static CYTHON_INLINE void __Pyx_RaiseCppAttributeError(const char *varname); /*proto*/

/////////////// RaiseCppAttributeError //////////////////////////////
static CYTHON_INLINE void __Pyx_RaiseCppAttributeError(const char *varname) {
    PyErr_Format(PyExc_AttributeError, "C++ attribute '%s' is not initialized", varname);
}
