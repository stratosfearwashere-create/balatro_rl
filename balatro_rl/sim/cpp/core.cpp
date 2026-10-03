// balatro_rl.sim._core: the C++ game core (pybind11). Built by  python setup_cython.py build  (needs
// pybind11); BALATRO_PURE=1 disables it like the other compiled extensions.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyrandom.hpp"
#include "card.hpp"
#include "hands.hpp"
#include "../_cards_api.h"
#include "solver.hpp"

namespace py = pybind11;
using balatro::PyRandom;

// ------------------------------------------------------------------ Random: Python-facing wrapper
struct Random {
    PyRandom r;
    Random(py::object seed) {
        if (seed.is_none()) throw py::value_error("Random needs an int seed (None: use random.Random)");
        py::int_ s = py::int_(seed);
        // abs value as 32-bit words, little-endian (random.seed(int))
        py::object a = py::module_::import("builtins").attr("abs")(s);
        std::vector<uint32_t> key;
        py::int_ v = a;
        while (py::bool_(v)) {
            key.push_back(py::cast<uint32_t>(v & py::int_(0xffffffffu)));
            v = v >> py::int_(32);
        }
        r.seed_words(key);
    }
    py::tuple getstate() const {
        py::tuple st(PyRandom::N + 1);
        for (int i = 0; i < PyRandom::N; i++) st[i] = py::int_(r.mt[i]);
        st[PyRandom::N] = py::int_(r.index);
        return py::make_tuple(3, st, py::none());
    }
    void setstate(py::tuple state) {
        py::tuple st = state[1];
        if (py::len(st) != PyRandom::N + 1) throw py::value_error("state vector is the wrong size");
        for (int i = 0; i < PyRandom::N; i++) r.mt[i] = py::cast<uint32_t>(st[i]);
        r.index = py::cast<int>(st[PyRandom::N]);
    }
    double random() { return r.random(); }
    py::int_ getrandbits(int k) { return py::int_(r.getrandbits(k)); }
    int64_t randrange(int64_t a, py::object b) {
        if (b.is_none()) return r.randrange(a);
        return r.randrange(a, py::cast<int64_t>(b));
    }
    int64_t randint(int64_t a, int64_t b) { return r.randint(a, b); }
    py::object choice(py::sequence seq) { return seq[r.choice_index(py::len(seq))]; }
    void shuffle(py::list x) {
        size_t n = py::len(x);
        for (size_t i = n - 1; i + 1 >= 2; i--) {
            size_t j = (size_t)r.randbelow(i + 1);
            py::object t = x[i];
            x[i] = x[j];
            x[j] = t;
        }
    }
    py::list sample(py::sequence pop, size_t k) {
        py::list out;
        for (size_t i : r.sample_indices(py::len(pop), k)) out.append(pop[i]);
        return out;
    }
    py::list choices(py::sequence pop, py::object weights, size_t k) {
        py::list out;
        std::vector<size_t> idx;
        if (weights.is_none()) idx = r.choices_indices(py::len(pop), k);
        else idx = r.choices_indices(py::cast<std::vector<double>>(weights), k);
        for (size_t i : idx) out.append(pop[i]);
        return out;
    }
    double uniform(double a, double b) { return r.uniform(a, b); }
};

// ------------------------------------------------------------------ hands.evaluate
static py::tuple evaluate_py(py::sequence cards, bool four_fingers, bool shortcut, bool smeared) {
    Py_ssize_t n = py::len(cards);
    if (n > 5) throw py::value_error("evaluate: at most 5 cards");
    balatro::CardView v[5];
    const balatro::CardView* ptr[5];
    for (Py_ssize_t i = 0; i < n; i++) {
        py::object o = cards[i];
        if (!PyObject_TypeCheck(o.ptr(), &CardType)) throw py::type_error("evaluate: a Card is needed");
        v[i] = balatro::CardView::of(o.ptr());
        ptr[i] = &v[i];
    }
    balatro::HandEval h = balatro::evaluate(ptr, (int)n, four_fingers, shortcut, smeared);
    py::list scoring;
    for (int i = 0; i < n; i++) if (h.scoring & (1 << i)) scoring.append(i);
    py::set contains;
    for (int t = 0; t < balatro::N_HANDS; t++) if (h.contains & (1 << t)) contains.add(t);
    return py::make_tuple(h.hand, scoring, contains);
}

// ------------------------------------------------------------------ the round solver's core
struct SolverCorePy {
    balatro::SolverCore core;
    py::object fs_ref;                   // keeps the scorer alive
    SolverCorePy(py::list pool, py::object fs, int hand_size, std::string boss, double target, int hl, int dl,
                 double chips, int types, int mouth, int inner, int max_steps)
        : core(convert(pool), fs.ptr(), hand_size, boss, target, hl, dl, chips, types, mouth, inner, max_steps),
          fs_ref(fs) {}
    static std::vector<balatro::PoolCard> convert(py::list pool) {
        std::vector<balatro::PoolCard> out;
        out.reserve(py::len(pool));
        for (py::handle h : pool) {
            if (!PyObject_TypeCheck(h.ptr(), &CardType)) throw py::type_error("pool: a Card is needed");
            balatro::CardView v = balatro::CardView::of(h.ptr());
            out.push_back(balatro::PoolCard{v.rank, v.suit, v.chip_value(), v.is_stone(), v.uid});
        }
        return out;
    }
    void set_rng_state(py::tuple state) {
        py::tuple st = state[1];
        if (py::len(st) != balatro::PyRandom::N + 1) throw py::value_error("state vector is the wrong size");
        for (int i = 0; i < balatro::PyRandom::N; i++) core.rng.mt[i] = py::cast<uint32_t>(st[i]);
        core.rng.index = py::cast<int>(st[balatro::PyRandom::N]);
    }
    py::tuple rng_state() const {
        py::tuple st(balatro::PyRandom::N + 1);
        for (int i = 0; i < balatro::PyRandom::N; i++) st[i] = py::int_(core.rng.mt[i]);
        st[balatro::PyRandom::N] = py::int_(core.rng.index);
        return py::make_tuple(3, st, py::none());
    }
    void make_futures(std::vector<int> hand, std::vector<int> order, std::vector<int> hidden_pos, int samples) {
        core.make_futures(hand, order, hidden_pos, samples);
    }
    py::tuple evaluate(std::vector<int> pos, double score, int hand, bool is_play, int depth) {
        std::pair<double, double> v;
        try {
            v = core.evaluate(pos, score, hand, is_play, depth);
        } catch (const std::length_error& e) {
            throw py::value_error(e.what());
        } catch (const std::runtime_error&) {
            throw py::error_already_set();
        }
        return py::make_tuple(v.first, v.second);
    }
    long long calls() const { return core.calls; }
    long long cache_hits() const { return core.cache_hits; }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ game core";
    if (import_balatro_rl__sim___cards() < 0) throw py::import_error("balatro_rl.sim._cards is needed");
    if (import_balatro_rl__sim___fastscore() < 0) throw py::import_error("balatro_rl.sim._fastscore is needed");
    py::class_<SolverCorePy>(m, "SolverCore")
        .def(py::init<py::list, py::object, int, std::string, double, int, int, double, int, int, int, int>(),
             py::arg("pool"), py::arg("scorer"), py::arg("hand_size"), py::arg("boss"), py::arg("target"),
             py::arg("hands_left"), py::arg("discards_left"), py::arg("chips"), py::arg("types"), py::arg("mouth"),
             py::arg("inner"), py::arg("max_steps"))
        .def("set_rng_state", &SolverCorePy::set_rng_state)
        .def("rng_state", &SolverCorePy::rng_state)
        .def("make_futures", &SolverCorePy::make_futures, py::arg("hand"), py::arg("order"), py::arg("hidden"),
             py::arg("samples"))
        .def("evaluate", &SolverCorePy::evaluate, py::arg("pos"), py::arg("score"), py::arg("hand"),
             py::arg("is_play"), py::arg("depth"))
        .def_property_readonly("calls", &SolverCorePy::calls)
        .def_property_readonly("cache_hits", &SolverCorePy::cache_hits);
    m.def("evaluate", &evaluate_py, py::arg("cards"), py::arg("four_fingers") = false, py::arg("shortcut") = false,
          py::arg("smeared") = false, "hands.evaluate: (hand, scoring positions, contained hand types)");
    py::class_<Random>(m, "Random")
        .def(py::init<py::object>(), py::arg("seed"))
        .def("getstate", &Random::getstate)
        .def("setstate", &Random::setstate)
        .def("random", &Random::random)
        .def("getrandbits", &Random::getrandbits)
        .def("randrange", &Random::randrange, py::arg("start"), py::arg("stop") = py::none())
        .def("randint", &Random::randint)
        .def("choice", &Random::choice)
        .def("shuffle", &Random::shuffle)
        .def("sample", &Random::sample, py::arg("population"), py::arg("k"))
        .def("choices", &Random::choices, py::arg("population"), py::arg("weights") = py::none(), py::arg("k") = 1)
        .def("uniform", &Random::uniform);
}
