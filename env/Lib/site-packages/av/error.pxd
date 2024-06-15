
cdef int stash_exception(exc_info=*)

cpdef int err_check(int res, filename=*) except -1
cpdef make_error(int res, filename=*, log=*)
