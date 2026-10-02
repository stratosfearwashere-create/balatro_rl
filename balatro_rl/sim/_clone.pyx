# cython: language_level=3, boundscheck=False, wraparound=False
"""Compiled Game.clone: the same copier as game._clone (fast paths for the game's own types, memoised so
sharing is preserved), with C-level card copies. game.py registers its types and uses clone_game when
this is built."""
cimport cython
import copy
import random

from ._cards cimport Card

cdef object _Consumable = None
cdef object _Joker = None
cdef object _ShopItem = None
cdef tuple _ATOMS = (int, float, str, bool, type(None), frozenset)
cdef object _Random = random.Random
cdef object _deepcopy = copy.deepcopy


def register(consumable, joker, shop_item):
    """game.py hands over its classes (importing them here would be circular)."""
    global _Consumable, _Joker, _ShopItem
    _Consumable, _Joker, _ShopItem = consumable, joker, shop_item


cdef object _clone(object v, dict memo):
    cdef type t = type(v)
    cdef list new_list
    cdef dict new_dict
    cdef object new, got
    cdef Py_ssize_t key
    if t in _ATOMS:
        return v
    key = <Py_ssize_t><void*>v
    got = memo.get(key)
    if got is not None:
        return got
    if t is list:
        new_list = []
        memo[key] = new_list
        for x in <list>v:
            new_list.append(_clone(x, memo))
        return new_list
    if t is Card:
        new = (<Card>v).dup()
    elif t is _Consumable:                           # a dataclass of immutable fields
        new = t.__new__(t)
        new.__dict__ = dict(v.__dict__)
    elif t is _Joker:
        new = t.__new__(t)
        new.__dict__ = dict(v.__dict__)              # its JokerDef `d` is shared, as deepcopy does
        new.state = _clone(v.state, memo)
    elif t is _ShopItem:
        new = t.__new__(t)
        new_dict = {}
        for k, x in (<dict>v.__dict__).items():
            new_dict[k] = _clone(x, memo)
        new.__dict__ = new_dict
    elif t is dict:
        new_dict = {}
        for k, x in (<dict>v).items():
            new_dict[k] = _clone(x, memo)
        new = new_dict
    elif t is set:
        new = set(v) if all(type(x) in _ATOMS for x in v) else _deepcopy(v)
    elif t is tuple:
        new = tuple([_clone(x, memo) for x in <tuple>v])
    elif t is _Random:
        new = _Random()
        new.setstate(v.getstate())
    else:
        new = _deepcopy(v)
    memo[key] = new
    return new


def clone_value(v):
    """game._clone for one value (tests)."""
    return _clone(v, {})


def clone_game(g):
    """An independent copy of a Game (see Game.clone)."""
    cdef dict memo = {}
    cdef dict d = {}
    new = type(g).__new__(type(g))
    for k, v in (<dict>g.__dict__).items():
        d[k] = _clone(v, memo)
    new.__dict__ = d
    return new
