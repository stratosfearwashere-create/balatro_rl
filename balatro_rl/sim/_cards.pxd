cdef class Card:
    cdef dict __dict__                 # ad-hoc attributes still work (the bridge tags cards with their API index)
    cdef public int rank
    cdef public int suit
    cdef public str enh
    cdef public str edition
    cdef public str seal
    cdef public int extra_chips
    cdef public bint debuffed
    cdef public bint hidden
    cdef public long long uid

    cdef bint stone(self)
    cdef Card dup(self)
    cpdef int chip_value(self)
    cpdef bint is_face(self, bint pareidolia=*, bint from_boss=*)
    cpdef bint has_suit(self, int s, bint smeared=*)
    cpdef bint flush_suit(self, int s, bint smeared=*)
    cpdef bint live_suit(self, int s, bint smeared=*)
