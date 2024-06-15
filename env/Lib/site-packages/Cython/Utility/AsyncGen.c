// This is copied from genobject.c in CPython 3.6.
// Try to keep it in sync by doing this from time to time:
//    sed -e 's|__pyx_||ig'  Cython/Utility/AsyncGen.c | diff -udw - cpython/Objects/genobject.c | less

//////////////////// AsyncGenerator.proto ////////////////////
//@requires: Coroutine.c::Coroutine

#define __Pyx_AsyncGen_USED
typedef struct {
    __pyx_CoroutineObject coro;
    PyObject *ag_finalizer;
    int ag_hooks_inited;
    int ag_closed;
    int ag_running_async;
} __pyx_PyAsyncGenObject;

static PyTypeObject *__pyx__PyAsyncGenWrappedValueType = 0;
static PyTypeObject *__pyx__PyAsyncGenASendType = 0;
static PyTypeObject *__pyx__PyAsyncGenAThrowType = 0;
static PyTypeObject *__pyx_AsyncGenType = 0;

#define __Pyx_AsyncGen_CheckExact(obj) __Pyx_IS_TYPE(obj, __pyx_AsyncGenType)
#define __pyx_PyAsyncGenASend_CheckExact(o) \
                    __Pyx_IS_TYPE(o, __pyx__PyAsyncGenASendType)
#define __pyx_PyAsyncGenAThrow_CheckExact(o) \
                    __Pyx_IS_TYPE(o, __pyx__PyAsyncGenAThrowType)

static PyObject *__Pyx_async_gen_anext(PyObject *o);
static CYTHON_INLINE PyObject *__Pyx_async_gen_asend_iternext(PyObject *o);
static PyObject *__Pyx_async_gen_asend_send(PyObject *o, PyObject *arg);
static PyObject *__Pyx_async_gen_asend_close(PyObject *o, PyObject *args);
static PyObject *__Pyx_async_gen_athrow_close(PyObject *o, PyObject *args);

static PyObject *__Pyx__PyAsyncGenValueWrapperNew(PyObject *val);


static __pyx_CoroutineObject *__Pyx_AsyncGen_New(
            __pyx_coroutine_body_t body, PyObject *code, PyObject *closure,
            PyObject *name, PyObject *qualname, PyObject *module_name) {
    __pyx_PyAsyncGenObject *gen = PyObject_GC_New(__pyx_PyAsyncGenObject, __pyx_AsyncGenType);
    if (unlikely(!gen))
        return NULL;
    gen->ag_finalizer = NULL;
    gen->ag_closed = 0;
    gen->ag_hooks_inited = 0;
    gen->ag_running_async = 0;
    return __Pyx__Coroutine_NewInit((__pyx_CoroutineObject*)gen, body, code, closure, name, qualname, module_name);
}

static int __pyx_AsyncGen_init(PyObject *module);
static void __Pyx_PyAsyncGen_Fini(void);

//////////////////// AsyncGenerator.cleanup ////////////////////

__Pyx_PyAsyncGen_Fini();

//////////////////// AsyncGeneratorInitFinalizer ////////////////////

// this is separated out because it needs more adaptation

#if PY_VERSION_HEX < 0x030600B0
static int __Pyx_async_gen_init_hooks(__pyx_PyAsyncGenObject *o) {
#if 0
    // TODO: implement finalizer support in older Python versions
    PyThreadState *tstate;
    PyObject *finalizer;
    PyObject *firstiter;
#endif

    if (likely(o->ag_hooks_inited)) {
        return 0;
    }

    o->ag_hooks_inited = 1;

#if 0
    tstate = __Pyx_PyThreadState_Current;

    finalizer = tstate->async_gen_finalizer;
    if (finalizer) {
        Py_INCREF(finalizer);
        o->ag_finalizer = finalizer;
    }

    firstiter = tstate->async_gen_firstiter;
    if (firstiter) {
        PyObject *res;

        Py_INCREF(firstiter);
        res = __Pyx_PyObject_CallOneArg(firstiter, (PyObject*)o);
        Py_DECREF(firstiter);
        if (res == NULL) {
            return 1;
        }
        Py_DECREF(res);
    }
#endif

    return 0;
}
#endif


//////////////////// AsyncGenerator ////////////////////
//@requires: AsyncGeneratorInitFinalizer
//@requires: Coroutine.c::Coroutine
//@requires: Coroutine.c::ReturnWithStopIteration
//@requires: ObjectHandling.c::PyObjectCall2Args
//@requires: ObjectHandling.c::PyObject_GenericGetAttrNoDict

PyDoc_STRVAR(__Pyx_async_gen_send_doc,
"send(arg) -> send 'arg' into generator,\n\
return next yielded value or raise StopIteration.");

PyDoc_STRVAR(__Pyx_async_gen_close_doc,
"close() -> raise GeneratorExit inside generator.");

PyDoc_STRVAR(__Pyx_async_gen_throw_doc,
"throw(typ[,val[,tb]]) -> raise exception in generator,\n\
return next yielded value or raise StopIteration.");

PyDoc_STRVAR(__Pyx_async_gen_await_doc,
"__await__() -> return a representation that can be passed into the 'await' expression.");

// COPY STARTS HERE:

static PyObject *__Pyx_async_gen_asend_new(__pyx_PyAsyncGenObject *, PyObject *);
static PyObject *__Pyx_async_gen_athrow_new(__pyx_PyAsyncGenObject *, PyObject *);

static const char *__Pyx_NON_INIT_CORO_MSG = "can't send non-None value to a just-started coroutine";
static const char *__Pyx_ASYNC_GEN_IGNORED_EXIT_MSG = "async generator ignored GeneratorExit";
static const char *__Pyx_ASYNC_GEN_CANNOT_REUSE_SEND_MSG = "cannot reuse already awaited __anext__()/asend()";
static const char *__Pyx_ASYNC_GEN_CANNOT_REUSE_CLOSE_MSG = "cannot reuse already awaited aclose()/athrow()";

typedef enum {
    __PYX_AWAITABLE_STATE_INIT,   /* new awaitable, has not yet been iterated */
    __PYX_AWAITABLE_STATE_ITER,   /* being iterated */
    __PYX_AWAITABLE_STATE_CLOSED, /* closed */
} __pyx_AwaitableState;

typedef struct {
    PyObject_HEAD
    __pyx_PyAsyncGenObject *ags_gen;

    /* Can be NULL, when in the __anext__() mode (equivalent of "asend(None)") */
    PyObject *ags_sendval;

    __pyx_AwaitableState ags_state;
} __pyx_PyAsyncGenASend;


typedef struct {
    PyObject_HEAD
    __pyx_PyAsyncGenObject *agt_gen;

    /* Can be NULL, when in the "aclose()" mode (equivalent of "athrow(GeneratorExit)") */
    PyObject *agt_args;

    __pyx_AwaitableState agt_state;
} __pyx_PyAsyncGenAThrow;


typedef struct {
    PyObject_HEAD
    PyObject *agw_val;
} __pyx__PyAsyncGenWrappedValue;


#ifndef _PyAsyncGen_MAXFREELIST
#define _PyAsyncGen_MAXFREELIST 80
#endif

// Freelists boost performance 6-10%; they also reduce memory
// fragmentation, as _PyAsyncGenWrappedValue and PyAsyncGenASend
// are short-living objects that are instantiated for every
// __anext__ call.

static __pyx__PyAsyncGenWrappedValue *__Pyx_ag_value_freelist[_PyAsyncGen_MAXFREELIST];
static int __Pyx_ag_value_freelist_free = 0;

static __pyx_PyAsyncGenASend *__Pyx_ag_asend_freelist[_PyAsyncGen_MAXFREELIST];
static int __Pyx_ag_asend_freelist_free = 0;

#define __pyx__PyAsyncGenWrappedValue_CheckExact(o) \
                    __Pyx_IS_TYPE(o, __pyx__PyAsyncGenWrappedValueType)


static int
__Pyx_async_gen_traverse(__pyx_PyAsyncGenObject *gen, visitproc visit, void *arg)
{
    Py_VISIT(gen->ag_finalizer);
    return __Pyx_Coroutine_traverse((__pyx_CoroutineObject*)gen, visit, arg);
}


static PyObject *
__Pyx_async_gen_repr(__pyx_CoroutineObject *o)
{
    // avoid NULL pointer dereference for qualname during garbage collection
    return PyUnicode_FromFormat("<async_generator object %S at %p>",
                                o->gi_qualname ? o->gi_qualname : Py_None, o);
}


#if PY_VERSION_HEX >= 0x030600B0
static int
__Pyx_async_gen_init_hooks(__pyx_PyAsyncGenObject *o)
{
#if !CYTHON_COMPILING_IN_PYPY
    PyThreadState *tstate;
#endif
    PyObject *finalizer;
    PyObject *firstiter;

    if (o->ag_hooks_inited) {
        return 0;
    }

    o->ag_hooks_inited = 1;

#if CYTHON_COMPILING_IN_PYPY
    finalizer = _PyEval_GetAsyncGenFinalizer();
#else
    tstate = __Pyx_PyThreadState_Current;
    finalizer = tstate->async_gen_finalizer;
#endif
    if (finalizer) {
        Py_INCREF(finalizer);
        o->ag_finalizer = finalizer;
    }

#if CYTHON_COMPILING_IN_PYPY
    firstiter = _PyEval_GetAsyncGenFirstiter();
#else
    firstiter = tstate->async_gen_firstiter;
#endif
    if (firstiter) {
        PyObject *res;
#if CYTHON_UNPACK_METHODS
        PyObject *self;
#endif

        Py_INCREF(firstiter);
        // at least asyncio stores methods here => optimise the call
#if CYTHON_UNPACK_METHODS
        if (likely(PyMethod_Check(firstiter)) && likely((self = PyMethod_GET_SELF(firstiter)) != NULL)) {
            PyObject *function = PyMethod_GET_FUNCTION(firstiter);
            res = __Pyx_PyObject_Call2Args(function, self, (PyObject*)o);
        } else
#endif
        res = __Pyx_PyObject_CallOneArg(firstiter, (PyObject*)o);

        Py_DECREF(firstiter);
        if (unlikely(res == NULL)) {
            return 1;
        }
        Py_DECREF(res);
    }

    return 0;
}
#endif


static PyObject *
__Pyx_async_gen_anext(PyObject *g)
{
    __pyx_PyAsyncGenObject *o = (__pyx_PyAsyncGenObject*) g;
    if (unlikely(__Pyx_async_gen_init_hooks(o))) {
        return NULL;
    }
    return __Pyx_async_gen_asend_new(o, NULL);
}

static PyObject *
__Pyx_async_gen_anext_method(PyObject *g, PyObject *arg) {
    CYTHON_UNUSED_VAR(arg);
    return __Pyx_async_gen_anext(g);
}


static PyObject *
__Pyx_async_gen_asend(__pyx_PyAsyncGenObject *o, PyObject *arg)
{
    if (unlikely(__Pyx_async_gen_init_hooks(o))) {
        return NULL;
    }
    return __Pyx_async_gen_asend_new(o, arg);
}


static PyObject *
__Pyx_async_gen_aclose(__pyx_PyAsyncGenObject *o, PyObject *arg)
{
    CYTHON_UNUSED_VAR(arg);
    if (unlikely(__Pyx_async_gen_init_hooks(o))) {
        return NULL;
    }
    return __Pyx_async_gen_athrow_new(o, NULL);
}


static PyObject *
__Pyx_async_gen_athrow(__pyx_PyAsyncGenObject *o, PyObject *args)
{
    if (unlikely(__Pyx_async_gen_init_hooks(o))) {
        return NULL;
    }
    return __Pyx_async_gen_athrow_new(o, args);
}


static PyObject *
__Pyx_async_gen_self_method(PyObject *g, PyObject *arg) {
    CYTHON_UNUSED_VAR(arg);
    return __Pyx_NewRef(g);
}


static PyGetSetDef __Pyx_async_gen_getsetlist[] = {
    {(char*) "__name__", (getter)__Pyx_Coroutine_get_name, (setter)__Pyx_Coroutine_set_name,
     (char*) PyDoc_STR("name of the async generator"), 0},
    {(char*) "__qualname__", (getter)__Pyx_Coroutine_get_qualname, (setter)__Pyx_Coroutine_set_qualname,
     (char*) PyDoc_STR("qualified name of the async generator"), 0},
    //REMOVED: {(char*) "ag_await", (getter)coro_get_cr_await, NULL,
    //REMOVED:  (char*) PyDoc_STR("object being awaited on, or None")},
    {0, 0, 0, 0, 0} /* Sentinel */
};

static PyMemberDef __Pyx_async_gen_memberlist[] = {
    //REMOVED: {(char*) "ag_frame",   T_OBJECT, offsetof(__pyx_PyAsyncGenObject, ag_frame),   READONLY},
    {(char*) "ag_running", T_BOOL,   offsetof(__pyx_PyAsyncGenObject, ag_running_async), READONLY, NULL},
    //REMOVED: {(char*) "ag_code",    T_OBJECT, offsetof(__pyx_PyAsyncGenObject, ag_code),    READONLY},
    //ADDED: "ag_await"
    {(char*) "ag_await", T_OBJECT, offsetof(__pyx_CoroutineObject, yieldfrom), READONLY,
     (char*) PyDoc_STR("object being awaited on, or None")},
    {(char *) "__module__", T_OBJECT, offsetof(__pyx_CoroutineObject, gi_modulename), 0, 0},
#if CYTHON_USE_TYPE_SPECS
    {(char *) "__weaklistoffset__", T_PYSSIZET, offsetof(__pyx_CoroutineObject, gi_weakreflist), READONLY, 0},
#endif
    {0, 0, 0, 0, 0}      /* Sentinel */
};

PyDoc_STRVAR(__Pyx_async_aclose_doc,
"aclose() -> raise GeneratorExit inside generator.");

PyDoc_STRVAR(__Pyx_async_asend_doc,
"asend(v) -> send 'v' in generator.");

PyDoc_STRVAR(__Pyx_async_athrow_doc,
"athrow(typ[,val[,tb]]) -> raise exception in generator.");

PyDoc_STRVAR(__Pyx_async_aiter_doc,
"__aiter__(v) -> return an asynchronous iterator.");

PyDoc_STRVAR(__Pyx_async_anext_doc,
"__anext__(v) -> continue asynchronous iteration and return the next element.");

static PyMethodDef __Pyx_async_gen_methods[] = {
    {"asend", (PyCFunction)__Pyx_async_gen_asend, METH_O, __Pyx_async_asend_doc},
    {"athrow",(PyCFunction)__Pyx_async_gen_athrow, METH_VARARGS, __Pyx_async_athrow_doc},
    {"aclose", (PyCFunction)__Pyx_async_gen_aclose, METH_NOARGS, __Pyx_async_aclose_doc},
    {"__aiter__", (PyCFunction)__Pyx_async_gen_self_method, METH_NOARGS, __Pyx_async_aiter_doc},
    {"__anext__", (PyCFunction)__Pyx_async_gen_anext_method, METH_NOARGS, __Pyx_async_anext_doc},
    {0, 0, 0, 0}        /* Sentinel */
};


#if CYTHON_USE_TYPE_SPECS
static PyType_Slot __pyx_AsyncGenType_slots[] = {
    {Py_tp_dealloc, (void *)__Pyx_Coroutine_dealloc},
    {Py_am_aiter, (void *)PyObject_SelfIter},
    {Py_am_anext, (void *)__Pyx_async_gen_anext},
    {Py_tp_repr, (void *)__Pyx_async_gen_repr},
    {Py_tp_traverse, (void *)__Pyx_async_gen_traverse},
    {Py_tp_methods, (void *)__Pyx_async_gen_methods},
    {Py_tp_members, (void *)__Pyx_async_gen_memberlist},
    {Py_tp_getset, (void *)__Pyx_async_gen_getsetlist},
#if CYTHON_USE_TP_FINALIZE
    {Py_tp_finalize, (void *)__Pyx_Coroutine_del},
#endif
    {0, 0},
};

static PyType_Spec __pyx_AsyncGenType_spec = {
    __PYX_TYPE_MODULE_PREFIX "async_generator",
    sizeof(__pyx_PyAsyncGenObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_FINALIZE, /*tp_flags*/
    __pyx_AsyncGenType_slots
};
#else /* CYTHON_USE_TYPE_SPECS */

#if CYTHON_USE_ASYNC_SLOTS
static __Pyx_PyAsyncMethodsStruct __Pyx_async_gen_as_async = {
    0,                                          /* am_await */
    PyObject_SelfIter,                          /* am_aiter */
    (unaryfunc)__Pyx_async_gen_anext,           /* am_anext */
#if PY_VERSION_HEX >= 0x030A00A3
    0, /*am_send*/
#endif
};
#endif

static PyTypeObject __pyx_AsyncGenType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator",                          /* tp_name */
    sizeof(__pyx_PyAsyncGenObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)__Pyx_Coroutine_dealloc,        /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if CYTHON_USE_ASYNC_SLOTS
    &__Pyx_async_gen_as_async,                        /* tp_as_async */
#else
    0,                                          /*tp_reserved*/
#endif
    (reprfunc)__Pyx_async_gen_repr,                   /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
        Py_TPFLAGS_HAVE_FINALIZE,               /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_traverse,           /* tp_traverse */
    0,                                          /* tp_clear */
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    offsetof(__pyx_CoroutineObject, gi_weakreflist), /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    __Pyx_async_gen_methods,                          /* tp_methods */
    __Pyx_async_gen_memberlist,                       /* tp_members */
    __Pyx_async_gen_getsetlist,                       /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
#if CYTHON_USE_TP_FINALIZE
    0,                                  /*tp_del*/
#else
    __Pyx_Coroutine_del,                /*tp_del*/
#endif
    0,                                          /* tp_version_tag */
#if CYTHON_USE_TP_FINALIZE
    __Pyx_Coroutine_del,                        /* tp_finalize */
#elif PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if __PYX_NEED_TP_PRINT_SLOT
    0,                                          /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030C0000
    0,                                          /*tp_watched*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000 && PY_VERSION_HEX < 0x030a0000
    0,                                          /*tp_pypy_flags*/
#endif
};
#endif  /* CYTHON_USE_TYPE_SPECS */


static int
__Pyx_PyAsyncGen_ClearFreeLists(void)
{
    int ret = __Pyx_ag_value_freelist_free + __Pyx_ag_asend_freelist_free;

    while (__Pyx_ag_value_freelist_free) {
        __pyx__PyAsyncGenWrappedValue *o;
        o = __Pyx_ag_value_freelist[--__Pyx_ag_value_freelist_free];
        assert(__pyx__PyAsyncGenWrappedValue_CheckExact(o));
        __Pyx_PyHeapTypeObject_GC_Del(o);
    }

    while (__Pyx_ag_asend_freelist_free) {
        __pyx_PyAsyncGenASend *o;
        o = __Pyx_ag_asend_freelist[--__Pyx_ag_asend_freelist_free];
        assert(__Pyx_IS_TYPE(o, __pyx__PyAsyncGenASendType));
        __Pyx_PyHeapTypeObject_GC_Del(o);
    }

    return ret;
}

static void
__Pyx_PyAsyncGen_Fini(void)
{
    __Pyx_PyAsyncGen_ClearFreeLists();
}


static PyObject *
__Pyx_async_gen_unwrap_value(__pyx_PyAsyncGenObject *gen, PyObject *result)
{
    if (result == NULL) {
        PyObject *exc_type = PyErr_Occurred();
        if (!exc_type) {
            PyErr_SetNone(__Pyx_PyExc_StopAsyncIteration);
            gen->ag_closed = 1;
        } else if (__Pyx_PyErr_GivenExceptionMatches2(exc_type, __Pyx_PyExc_StopAsyncIteration, PyExc_GeneratorExit)) {
            gen->ag_closed = 1;
        }

        gen->ag_running_async = 0;
        return NULL;
    }

    if (__pyx__PyAsyncGenWrappedValue_CheckExact(result)) {
        /* async yield */
        __Pyx_ReturnWithStopIteration(((__pyx__PyAsyncGenWrappedValue*)result)->agw_val);
        Py_DECREF(result);
        gen->ag_running_async = 0;
        return NULL;
    }

    return result;
}


/* ---------- Async Generator ASend Awaitable ------------ */


static void
__Pyx_async_gen_asend_dealloc(__pyx_PyAsyncGenASend *o)
{
    PyObject_GC_UnTrack((PyObject *)o);
    Py_CLEAR(o->ags_gen);
    Py_CLEAR(o->ags_sendval);
    if (likely(__Pyx_ag_asend_freelist_free < _PyAsyncGen_MAXFREELIST)) {
        assert(__pyx_PyAsyncGenASend_CheckExact(o));
        __Pyx_ag_asend_freelist[__Pyx_ag_asend_freelist_free++] = o;
    } else {
        __Pyx_PyHeapTypeObject_GC_Del(o);
    }
}

static int
__Pyx_async_gen_asend_traverse(__pyx_PyAsyncGenASend *o, visitproc visit, void *arg)
{
    Py_VISIT(o->ags_gen);
    Py_VISIT(o->ags_sendval);
    return 0;
}


static PyObject *
__Pyx_async_gen_asend_send(PyObject *g, PyObject *arg)
{
    __pyx_PyAsyncGenASend *o = (__pyx_PyAsyncGenASend*) g;
    PyObject *result;

    if (unlikely(o->ags_state == __PYX_AWAITABLE_STATE_CLOSED)) {
        PyErr_SetString(PyExc_RuntimeError, __Pyx_ASYNC_GEN_CANNOT_REUSE_SEND_MSG);
        return NULL;
    }

    if (o->ags_state == __PYX_AWAITABLE_STATE_INIT) {
        if (unlikely(o->ags_gen->ag_running_async)) {
            PyErr_SetString(
                PyExc_RuntimeError,
                "anext(): asynchronous generator is already running");
            return NULL;
        }

        if (arg == NULL || arg == Py_None) {
            arg = o->ags_sendval ? o->ags_sendval : Py_None;
        }
        o->ags_state = __PYX_AWAITABLE_STATE_ITER;
    }

    o->ags_gen->ag_running_async = 1;
    result = __Pyx_Coroutine_Send((PyObject*)o->ags_gen, arg);
    result = __Pyx_async_gen_unwrap_value(o->ags_gen, result);

    if (result == NULL) {
        o->ags_state = __PYX_AWAITABLE_STATE_CLOSED;
    }

    return result;
}


static CYTHON_INLINE PyObject *
__Pyx_async_gen_asend_iternext(PyObject *o)
{
    return __Pyx_async_gen_asend_send(o, Py_None);
}


static PyObject *
__Pyx_async_gen_asend_throw(__pyx_PyAsyncGenASend *o, PyObject *args)
{
    PyObject *result;

    if (unlikely(o->ags_state == __PYX_AWAITABLE_STATE_CLOSED)) {
        PyErr_SetString(PyExc_RuntimeError, __Pyx_ASYNC_GEN_CANNOT_REUSE_SEND_MSG);
        return NULL;
    }

    result = __Pyx_Coroutine_Throw((PyObject*)o->ags_gen, args);
    result = __Pyx_async_gen_unwrap_value(o->ags_gen, result);

    if (result == NULL) {
        o->ags_state = __PYX_AWAITABLE_STATE_CLOSED;
    }

    return result;
}


static PyObject *
__Pyx_async_gen_asend_close(PyObject *g, PyObject *args)
{
    __pyx_PyAsyncGenASend *o = (__pyx_PyAsyncGenASend*) g;
    CYTHON_UNUSED_VAR(args);
    o->ags_state = __PYX_AWAITABLE_STATE_CLOSED;
    Py_RETURN_NONE;
}


static PyMethodDef __Pyx_async_gen_asend_methods[] = {
    {"send", (PyCFunction)__Pyx_async_gen_asend_send, METH_O, __Pyx_async_gen_send_doc},
    {"throw", (PyCFunction)__Pyx_async_gen_asend_throw, METH_VARARGS, __Pyx_async_gen_throw_doc},
    {"close", (PyCFunction)__Pyx_async_gen_asend_close, METH_NOARGS, __Pyx_async_gen_close_doc},
    {"__await__", (PyCFunction)__Pyx_async_gen_self_method, METH_NOARGS, __Pyx_async_gen_await_doc},
    {0, 0, 0, 0}        /* Sentinel */
};


#if CYTHON_USE_TYPE_SPECS
static PyType_Slot __pyx__PyAsyncGenASendType_slots[] = {
    {Py_tp_dealloc, (void *)__Pyx_async_gen_asend_dealloc},
    {Py_am_await, (void *)PyObject_SelfIter},
    {Py_tp_traverse, (void *)__Pyx_async_gen_asend_traverse},
    {Py_tp_methods, (void *)__Pyx_async_gen_asend_methods},
    {Py_tp_iter, (void *)PyObject_SelfIter},
    {Py_tp_iternext, (void *)__Pyx_async_gen_asend_iternext},
    {0, 0},
};

static PyType_Spec __pyx__PyAsyncGenASendType_spec = {
    __PYX_TYPE_MODULE_PREFIX "async_generator_asend",
    sizeof(__pyx_PyAsyncGenASend),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
    __pyx__PyAsyncGenASendType_slots
};
#else /* CYTHON_USE_TYPE_SPECS */

#if CYTHON_USE_ASYNC_SLOTS
static __Pyx_PyAsyncMethodsStruct __Pyx_async_gen_asend_as_async = {
    PyObject_SelfIter,                          /* am_await */
    0,                                          /* am_aiter */
    0,                                          /* am_anext */
#if PY_VERSION_HEX >= 0x030A00A3
    0, /*am_send*/
#endif
};
#endif

static PyTypeObject __pyx__PyAsyncGenASendType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator_asend",                    /* tp_name */
    sizeof(__pyx_PyAsyncGenASend),                    /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)__Pyx_async_gen_asend_dealloc,        /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if CYTHON_USE_ASYNC_SLOTS
    &__Pyx_async_gen_asend_as_async,                  /* tp_as_async */
#else
    0,                                          /*tp_reserved*/
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,    /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_asend_traverse,  /* tp_traverse */
    0,                                          /* tp_clear */
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    0,                                          /* tp_weaklistoffset */
    PyObject_SelfIter,                          /* tp_iter */
    (iternextfunc)__Pyx_async_gen_asend_iternext,     /* tp_iternext */
    __Pyx_async_gen_asend_methods,                    /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
#if PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if __PYX_NEED_TP_PRINT_SLOT
    0,                                          /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030C0000
    0,                                          /*tp_watched*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000 && PY_VERSION_HEX < 0x030a0000
    0,                                          /*tp_pypy_flags*/
#endif
};
#endif /* CYTHON_USE_TYPE_SPECS */


static PyObject *
__Pyx_async_gen_asend_new(__pyx_PyAsyncGenObject *gen, PyObject *sendval)
{
    __pyx_PyAsyncGenASend *o;
    if (likely(__Pyx_ag_asend_freelist_free)) {
        __Pyx_ag_asend_freelist_free--;
        o = __Pyx_ag_asend_freelist[__Pyx_ag_asend_freelist_free];
        _Py_NewReference((PyObject *)o);
    } else {
        o = PyObject_GC_New(__pyx_PyAsyncGenASend, __pyx__PyAsyncGenASendType);
        if (unlikely(o == NULL)) {
            return NULL;
        }
    }

    Py_INCREF(gen);
    o->ags_gen = gen;

    Py_XINCREF(sendval);
    o->ags_sendval = sendval;

    o->ags_state = __PYX_AWAITABLE_STATE_INIT;

    PyObject_GC_Track((PyObject*)o);
    return (PyObject*)o;
}


/* ---------- Async Generator Value Wrapper ------------ */


static void
__Pyx_async_gen_wrapped_val_dealloc(__pyx__PyAsyncGenWrappedValue *o)
{
    PyObject_GC_UnTrack((PyObject *)o);
    Py_CLEAR(o->agw_val);
    if (likely(__Pyx_ag_value_freelist_free < _PyAsyncGen_MAXFREELIST)) {
        assert(__pyx__PyAsyncGenWrappedValue_CheckExact(o));
        __Pyx_ag_value_freelist[__Pyx_ag_value_freelist_free++] = o;
    } else {
        __Pyx_PyHeapTypeObject_GC_Del(o);
    }
}


static int
__Pyx_async_gen_wrapped_val_traverse(__pyx__PyAsyncGenWrappedValue *o,
                                     visitproc visit, void *arg)
{
    Py_VISIT(o->agw_val);
    return 0;
}


#if CYTHON_USE_TYPE_SPECS
static PyType_Slot __pyx__PyAsyncGenWrappedValueType_slots[] = {
    {Py_tp_dealloc, (void *)__Pyx_async_gen_wrapped_val_dealloc},
    {Py_tp_traverse, (void *)__Pyx_async_gen_wrapped_val_traverse},
    {0, 0},
};

static PyType_Spec __pyx__PyAsyncGenWrappedValueType_spec = {
    __PYX_TYPE_MODULE_PREFIX "async_generator_wrapped_value",
    sizeof(__pyx__PyAsyncGenWrappedValue),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
    __pyx__PyAsyncGenWrappedValueType_slots
};
#else /* CYTHON_USE_TYPE_SPECS */

static PyTypeObject __pyx__PyAsyncGenWrappedValueType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator_wrapped_value",            /* tp_name */
    sizeof(__pyx__PyAsyncGenWrappedValue),            /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)__Pyx_async_gen_wrapped_val_dealloc,  /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,    /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_wrapped_val_traverse,    /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
#if PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if __PYX_NEED_TP_PRINT_SLOT
    0,                                          /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030C0000
    0,                                          /*tp_watched*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000 && PY_VERSION_HEX < 0x030a0000
    0,                                          /*tp_pypy_flags*/
#endif
};
#endif /* CYTHON_USE_TYPE_SPECS */


static PyObject *
__Pyx__PyAsyncGenValueWrapperNew(PyObject *val)
{
    // NOTE: steals a reference to val !
    __pyx__PyAsyncGenWrappedValue *o;
    assert(val);

    if (likely(__Pyx_ag_value_freelist_free)) {
        __Pyx_ag_value_freelist_free--;
        o = __Pyx_ag_value_freelist[__Pyx_ag_value_freelist_free];
        assert(__pyx__PyAsyncGenWrappedValue_CheckExact(o));
        _Py_NewReference((PyObject*)o);
    } else {
        o = PyObject_GC_New(__pyx__PyAsyncGenWrappedValue, __pyx__PyAsyncGenWrappedValueType);
        if (unlikely(!o)) {
            Py_DECREF(val);
            return NULL;
        }
    }
    o->agw_val = val;
    // no Py_INCREF(val) - steals reference!
    PyObject_GC_Track((PyObject*)o);
    return (PyObject*)o;
}


/* ---------- Async Generator AThrow awaitable ------------ */


static void
__Pyx_async_gen_athrow_dealloc(__pyx_PyAsyncGenAThrow *o)
{
    PyObject_GC_UnTrack((PyObject *)o);
    Py_CLEAR(o->agt_gen);
    Py_CLEAR(o->agt_args);
    __Pyx_PyHeapTypeObject_GC_Del(o);
}


static int
__Pyx_async_gen_athrow_traverse(__pyx_PyAsyncGenAThrow *o, visitproc visit, void *arg)
{
    Py_VISIT(o->agt_gen);
    Py_VISIT(o->agt_args);
    return 0;
}


static PyObject *
__Pyx_async_gen_athrow_send(__pyx_PyAsyncGenAThrow *o, PyObject *arg)
{
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject*)o->agt_gen;
    PyObject *retval, *exc_type;

    if (unlikely(o->agt_state == __PYX_AWAITABLE_STATE_CLOSED)) {
        PyErr_SetString(PyExc_RuntimeError, __Pyx_ASYNC_GEN_CANNOT_REUSE_CLOSE_MSG);
        return NULL;
    }

    if (unlikely(gen->resume_label == -1)) {
        // already run past the end
        o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    if (o->agt_state == __PYX_AWAITABLE_STATE_INIT) {
        if (unlikely(o->agt_gen->ag_running_async)) {
            o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
            if (o->agt_args == NULL) {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "aclose(): asynchronous generator is already running");
            } else {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "athrow(): asynchronous generator is already running");
            }
            return NULL;
        }

        if (unlikely(o->agt_gen->ag_closed)) {
            o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
            PyErr_SetNone(__Pyx_PyExc_StopAsyncIteration);
            return NULL;
        }

        if (unlikely(arg != Py_None)) {
            PyErr_SetString(PyExc_RuntimeError, __Pyx_NON_INIT_CORO_MSG);
            return NULL;
        }

        o->agt_state = __PYX_AWAITABLE_STATE_ITER;
        o->agt_gen->ag_running_async = 1;

        if (o->agt_args == NULL) {
            /* aclose() mode */
            o->agt_gen->ag_closed = 1;

            retval = __Pyx__Coroutine_Throw((PyObject*)gen,
                /* Do not close generator when PyExc_GeneratorExit is passed */
                PyExc_GeneratorExit, NULL, NULL, NULL, 0);

            if (retval && __pyx__PyAsyncGenWrappedValue_CheckExact(retval)) {
                Py_DECREF(retval);
                goto yield_close;
            }
        } else {
            PyObject *typ;
            PyObject *tb = NULL;
            PyObject *val = NULL;

            if (unlikely(!PyArg_UnpackTuple(o->agt_args, "athrow", 1, 3, &typ, &val, &tb))) {
                return NULL;
            }

            retval = __Pyx__Coroutine_Throw((PyObject*)gen,
                /* Do not close generator when PyExc_GeneratorExit is passed */
                typ, val, tb, o->agt_args, 0);
            retval = __Pyx_async_gen_unwrap_value(o->agt_gen, retval);
        }
        if (retval == NULL) {
            goto check_error;
        }
        return retval;
    }

    assert (o->agt_state == __PYX_AWAITABLE_STATE_ITER);

    retval = __Pyx_Coroutine_Send((PyObject *)gen, arg);
    if (o->agt_args) {
        return __Pyx_async_gen_unwrap_value(o->agt_gen, retval);
    } else {
        /* aclose() mode */
        if (retval) {
            if (unlikely(__pyx__PyAsyncGenWrappedValue_CheckExact(retval))) {
                Py_DECREF(retval);
                goto yield_close;
            }
            else {
                return retval;
            }
        }
        else {
            goto check_error;
        }
    }

yield_close:
    o->agt_gen->ag_running_async = 0;
    o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
    PyErr_SetString(
        PyExc_RuntimeError, __Pyx_ASYNC_GEN_IGNORED_EXIT_MSG);
    return NULL;

check_error:
    o->agt_gen->ag_running_async = 0;
    o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
    exc_type = PyErr_Occurred();
    if (__Pyx_PyErr_GivenExceptionMatches2(exc_type, __Pyx_PyExc_StopAsyncIteration, PyExc_GeneratorExit)) {
        if (o->agt_args == NULL) {
            // when aclose() is called we don't want to propagate
            // StopAsyncIteration or GeneratorExit; just raise
            // StopIteration, signalling that this 'aclose()' await
            // is done.
            PyErr_Clear();
            PyErr_SetNone(PyExc_StopIteration);
        }
    }
    return NULL;
}


static PyObject *
__Pyx_async_gen_athrow_throw(__pyx_PyAsyncGenAThrow *o, PyObject *args)
{
    PyObject *retval;

    if (unlikely(o->agt_state == __PYX_AWAITABLE_STATE_CLOSED)) {
        PyErr_SetString(PyExc_RuntimeError, __Pyx_ASYNC_GEN_CANNOT_REUSE_CLOSE_MSG);
        return NULL;
    }

    retval = __Pyx_Coroutine_Throw((PyObject*)o->agt_gen, args);
    if (o->agt_args) {
        return __Pyx_async_gen_unwrap_value(o->agt_gen, retval);
    } else {
        // aclose() mode
        PyObject *exc_type;
        if (unlikely(retval && __pyx__PyAsyncGenWrappedValue_CheckExact(retval))) {
            o->agt_gen->ag_running_async = 0;
            o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
            Py_DECREF(retval);
            PyErr_SetString(PyExc_RuntimeError, __Pyx_ASYNC_GEN_IGNORED_EXIT_MSG);
            return NULL;
        }
        exc_type = PyErr_Occurred();
        if (__Pyx_PyErr_GivenExceptionMatches2(exc_type, __Pyx_PyExc_StopAsyncIteration, PyExc_GeneratorExit)) {
            // when aclose() is called we don't want to propagate
            // StopAsyncIteration or GeneratorExit; just raise
            // StopIteration, signalling that this 'aclose()' await
            // is done.
            PyErr_Clear();
            PyErr_SetNone(PyExc_StopIteration);
        }
        return retval;
    }
}


static PyObject *
__Pyx_async_gen_athrow_iternext(__pyx_PyAsyncGenAThrow *o)
{
    return __Pyx_async_gen_athrow_send(o, Py_None);
}


static PyObject *
__Pyx_async_gen_athrow_close(PyObject *g, PyObject *args)
{
    __pyx_PyAsyncGenAThrow *o = (__pyx_PyAsyncGenAThrow*) g;
    CYTHON_UNUSED_VAR(args);
    o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
    Py_RETURN_NONE;
}


static PyMethodDef __Pyx_async_gen_athrow_methods[] = {
    {"send", (PyCFunction)__Pyx_async_gen_athrow_send, METH_O, __Pyx_async_gen_send_doc},
    {"throw", (PyCFunction)__Pyx_async_gen_athrow_throw, METH_VARARGS, __Pyx_async_gen_throw_doc},
    {"close", (PyCFunction)__Pyx_async_gen_athrow_close, METH_NOARGS, __Pyx_async_gen_close_doc},
    {"__await__", (PyCFunction)__Pyx_async_gen_self_method, METH_NOARGS, __Pyx_async_gen_await_doc},
    {0, 0, 0, 0}        /* Sentinel */
};


#if CYTHON_USE_TYPE_SPECS
static PyType_Slot __pyx__PyAsyncGenAThrowType_slots[] = {
    {Py_tp_dealloc, (void *)__Pyx_async_gen_athrow_dealloc},
    {Py_am_await, (void *)PyObject_SelfIter},
    {Py_tp_traverse, (void *)__Pyx_async_gen_athrow_traverse},
    {Py_tp_iter, (void *)PyObject_SelfIter},
    {Py_tp_iternext, (void *)__Pyx_async_gen_athrow_iternext},
    {Py_tp_methods, (void *)__Pyx_async_gen_athrow_methods},
    {Py_tp_getattro, (void *)__Pyx_PyObject_GenericGetAttrNoDict},
    {0, 0},
};

static PyType_Spec __pyx__PyAsyncGenAThrowType_spec = {
    __PYX_TYPE_MODULE_PREFIX "async_generator_athrow",
    sizeof(__pyx_PyAsyncGenAThrow),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
    __pyx__PyAsyncGenAThrowType_slots
};
#else /* CYTHON_USE_TYPE_SPECS */

#if CYTHON_USE_ASYNC_SLOTS
static __Pyx_PyAsyncMethodsStruct __Pyx_async_gen_athrow_as_async = {
    PyObject_SelfIter,                          /* am_await */
    0,                                          /* am_aiter */
    0,                                          /* am_anext */
#if PY_VERSION_HEX >= 0x030A00A3
    0, /*am_send*/
#endif
};
#endif

static PyTypeObject __pyx__PyAsyncGenAThrowType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator_athrow",                   /* tp_name */
    sizeof(__pyx_PyAsyncGenAThrow),                   /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)__Pyx_async_gen_athrow_dealloc,       /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if CYTHON_USE_ASYNC_SLOTS
    &__Pyx_async_gen_athrow_as_async,                 /* tp_as_async */
#else
    0,                                          /*tp_reserved*/
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,    /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_athrow_traverse,    /* tp_traverse */
    0,                                          /* tp_clear */
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    0,                                          /* tp_weaklistoffset */
    PyObject_SelfIter,                          /* tp_iter */
    (iternextfunc)__Pyx_async_gen_athrow_iternext,    /* tp_iternext */
    __Pyx_async_gen_athrow_methods,                   /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
#if PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if __PYX_NEED_TP_PRINT_SLOT
    0,                                          /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030C0000
    0,                                          /*tp_watched*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000 && PY_VERSION_HEX < 0x030a0000
    0,                                          /*tp_pypy_flags*/
#endif
};
#endif /* CYTHON_USE_TYPE_SPECS */


static PyObject *
__Pyx_async_gen_athrow_new(__pyx_PyAsyncGenObject *gen, PyObject *args)
{
    __pyx_PyAsyncGenAThrow *o;
    o = PyObject_GC_New(__pyx_PyAsyncGenAThrow, __pyx__PyAsyncGenAThrowType);
    if (unlikely(o == NULL)) {
        return NULL;
    }
    o->agt_gen = gen;
    o->agt_args = args;
    o->agt_state = __PYX_AWAITABLE_STATE_INIT;
    Py_INCREF(gen);
    Py_XINCREF(args);
    PyObject_GC_Track((PyObject*)o);
    return (PyObject*)o;
}


/* ---------- global type sharing ------------ */

static int __pyx_AsyncGen_init(PyObject *module) {
#if CYTHON_USE_TYPE_SPECS
    __pyx_AsyncGenType = __Pyx_FetchCommonTypeFromSpec(module, &__pyx_AsyncGenType_spec, NULL);
#else
    CYTHON_MAYBE_UNUSED_VAR(module);
    // on Windows, C-API functions can't be used in slots statically
    __pyx_AsyncGenType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx_AsyncGenType = __Pyx_FetchCommonType(&__pyx_AsyncGenType_type);
#endif
    if (unlikely(!__pyx_AsyncGenType))
        return -1;

#if CYTHON_USE_TYPE_SPECS
    __pyx__PyAsyncGenAThrowType = __Pyx_FetchCommonTypeFromSpec(module, &__pyx__PyAsyncGenAThrowType_spec, NULL);
#else
    __pyx__PyAsyncGenAThrowType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx__PyAsyncGenAThrowType = __Pyx_FetchCommonType(&__pyx__PyAsyncGenAThrowType_type);
#endif
    if (unlikely(!__pyx__PyAsyncGenAThrowType))
        return -1;

#if CYTHON_USE_TYPE_SPECS
    __pyx__PyAsyncGenWrappedValueType = __Pyx_FetchCommonTypeFromSpec(module, &__pyx__PyAsyncGenWrappedValueType_spec, NULL);
#else
    __pyx__PyAsyncGenWrappedValueType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx__PyAsyncGenWrappedValueType = __Pyx_FetchCommonType(&__pyx__PyAsyncGenWrappedValueType_type);
#endif
    if (unlikely(!__pyx__PyAsyncGenWrappedValueType))
        return -1;

#if CYTHON_USE_TYPE_SPECS
    __pyx__PyAsyncGenASendType = __Pyx_FetchCommonTypeFromSpec(module, &__pyx__PyAsyncGenASendType_spec, NULL);
#else
    __pyx__PyAsyncGenASendType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx__PyAsyncGenASendType = __Pyx_FetchCommonType(&__pyx__PyAsyncGenASendType_type);
#endif
    if (unlikely(!__pyx__PyAsyncGenASendType))
        return -1;

    return 0;
}
