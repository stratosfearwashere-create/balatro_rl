// The round solver's playouts (az/solver.py RoundSolver, az/_solver.pyx) in C++: cards are indices into
// the solve's pool (the hand, then the draw pile), hands are scored by the compiled scorer through its C
// API, and the agent's random generator is a C++ replica whose state is loaded at the start of the solve
// and written back at the end. The random draws happen in exactly the order of the Python code, so
// P(clear) and the expected chips are bit for bit the same (scores are doubles here and Python ints
// there: identical below 2^53).
#pragma once
#include <Python.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "pyrandom.hpp"
#include "../_fastscore_api.h"

namespace balatro {

struct PoolCard {
    int rank, suit, chip_value;
    bool stone, wild;
    long long uid;
};

// The hooks a round runs on the jokers besides scoring (jokers.py): on a discard and after a hand is played.
// Only the ones that change a joker's "val" are tabled; the table is built in az/solver.py by joker key.
enum DiscardKind { DK_NONE = 0, DK_GREEN = 1, DK_CASTLE = 2, DK_ROAD = 3, DK_RAMEN = 4 };
enum AfterKind { AK_NONE = 0, AK_ICE = 1 };
struct JokerHook {
    int dk = DK_NONE, ak = AK_NONE, suit = 0;
    bool active = true;                 // a debuffed joker's hooks don't run
};

// What a round changes hand by hand and the scorer reads: joker vals, hands played per type (run / round)
struct RoundState {
    std::vector<double> val;
    std::vector<int> hv;
    int hplayed[12], hpr[12];
};

struct Pred {              // one of the two best plays of a hand
    double score;
    int hand;
    std::vector<int> pos;  // positions in the hand
};

struct Round {
    std::vector<int> hand, draw;
    int hl, dl;
    double chips;
    int types;             // bitmask of hand types played this round (The Eye / The Mouth)
    int mouth;
    RoundState st;         // carried through the playout when the solver tracks joker state
};

class SolverCore {
public:
    std::vector<PoolCard> pool;
    PyObject* fs;                       // _fastscore.Scorer over the pool (borrowed; the owner keeps it alive)
    PyRandom rng;
    double target;                      // g.target (the comparisons)
    double target_f;                    // max(1.0, float(g.target)) (e_chips)
    int hand_size;
    bool hook = false, serpent = false, eye = false;    // eye: The Eye or The Mouth
    int base_types, base_mouth;
    int last_types, last_mouth;         // what the last _preds left in the game (read by the hidden path)
    int inner, max_steps;
    std::vector<std::vector<int>> f_hands, f_orders;    // the sampled futures
    Round start;
    bool hidden = false;
    long long calls = 0, cache_hits = 0;
    std::unordered_map<std::string, std::vector<Pred>> cache;
    bool carry = false;                 // track joker state and hands played through the playouts
    int nj = 0;
    std::vector<JokerHook> hooks;

    SolverCore(std::vector<PoolCard> pool_, PyObject* fs_, int hand_size_, const std::string& boss,
               double target_, int hl, int dl, double chips, int types, int mouth, int inner_, int max_steps_,
               bool carry_ = false, std::vector<JokerHook> hooks_ = {})
        : pool(std::move(pool_)), fs(fs_), target(target_), target_f(std::max(1.0, target_)), hand_size(hand_size_),
          base_types(types), base_mouth(mouth), last_types(types), last_mouth(mouth), inner(inner_),
          max_steps(max_steps_), carry(carry_), hooks(std::move(hooks_)) {
        hook = boss == "hook";
        serpent = boss == "serpent";
        eye = boss == "eye" || boss == "mouth";
        start.hl = hl; start.dl = dl; start.chips = chips; start.types = types; start.mouth = mouth;
        nj = fs_nj(fs);
        if (nj < 0) throw std::runtime_error("scorer");
        start.st.val.assign(nj, 0.0);
        start.st.hv.assign(nj, 0);
        if (fs_get_state(fs, start.st.val.data(), start.st.hv.data(), start.st.hplayed, start.st.hpr) < 0)
            throw std::runtime_error("scorer");
        hooks.resize(nj);
    }

    // ---- round state
    void set_state(const RoundState& st) {
        if (!carry) return;
        if (fs_set_state(fs, st.val.data(), st.hv.data(), st.hplayed, st.hpr) < 0) throw std::runtime_error("scorer");
    }

    void key_state(std::string& key, const RoundState& st) const {
        if (!carry) return;
        key.append((const char*)st.val.data(), st.val.size() * sizeof(double));
        key.append((const char*)st.hv.data(), st.hv.size() * sizeof(int));
        key.append((const char*)st.hplayed, sizeof(st.hplayed));
        key.append((const char*)st.hpr, sizeof(st.hpr));
    }

    // the state after `pos` is played and scored from r (the scorer's "before" / card hooks, then the
    // jokers' after-play hooks); a hand the boss forbids scores nothing and changes no joker
    void advance_play(Round& r, const std::vector<int>& pos, int htype) {
        if (!carry) return;
        if (r.hand.size() > 16) throw std::length_error("hand too large for the compiled scorer");
        int types = base_types, mouth = base_mouth;
        if (eye) { types = r.types; mouth = r.mouth; }
        set_state(r.st);
        double sc; int h, viol;
        std::vector<double> val(nj);
        std::vector<int> hv(nj);
        if (fs_score_state(fs, r.hand.data(), (int)r.hand.size(), pos.data(), (int)pos.size(), r.hl, r.dl,
                           (long long)r.draw.size(), types, mouth, &sc, &h, &viol, val.data(), hv.data()) < 0)
            throw std::runtime_error("scorer");
        if (!viol) {
            r.st.val.swap(val);
            r.st.hv.swap(hv);
            for (int j = 0; j < nj; j++) {
                if (!hooks[j].active) continue;
                if (hooks[j].ak == AK_ICE) {               // Ice Cream: -5 chips per hand; gone at 0
                    double v = (r.st.hv[j] ? r.st.val[j] : 100.0) - 5.0;
                    r.st.val[j] = v > 0 ? v : 0.0;
                    r.st.hv[j] = 1;
                }
            }
        }
        if (htype >= 0 && htype < 12) { r.st.hplayed[htype] += 1; r.st.hpr[htype] += 1; }
    }

    void advance_discard(Round& r, const std::vector<int>& pos) {
        if (!carry) return;
        for (int j = 0; j < nj; j++) {
            const JokerHook& hk = hooks[j];
            if (!hk.active || hk.dk == DK_NONE) continue;
            double v = r.st.hv[j] ? r.st.val[j] : 0.0;
            int n = 0;
            switch (hk.dk) {
                case DK_GREEN:                              // -1 Mult per discard, not below 0
                    v = std::max(0.0, v - 1.0);
                    break;
                case DK_CASTLE:                             // +3 chips per discarded card of its suit
                    for (int i : pos) {
                        const PoolCard& c = pool[r.hand[i]];
                        if (!c.stone && (c.wild || c.suit == hk.suit)) n++;
                    }
                    v += 3.0 * n;
                    break;
                case DK_ROAD:                               // +0.5 X per discarded Jack
                    if (!r.st.hv[j]) v = 1.0;
                    for (int i : pos) {
                        const PoolCard& c = pool[r.hand[i]];
                        if (!c.stone && c.rank == 11) n++;
                    }
                    v += 0.5 * n;
                    break;
                case DK_RAMEN:                              // -0.01 X per discarded card; gone at 1
                    if (!r.st.hv[j]) v = 2.0;
                    v -= 0.01 * (double)pos.size();
                    if (v <= 1.0) v = 1.0;
                    break;
            }
            r.st.val[j] = v;
            r.st.hv[j] = 1;
        }
    }

    // futures: `order` is the draw pile plus the face-down cards of the hand, sorted by uid, shuffled; the
    // face-down slots take cards from its end
    void make_futures(const std::vector<int>& hand, const std::vector<int>& order0, const std::vector<int>& hidden_pos,
                      int samples) {
        hidden = !hidden_pos.empty();
        for (int k = 0; k < samples; k++) {
            std::vector<int> order = order0;
            rng.shuffle(order);
            std::vector<int> h = hand;
            for (int i : hidden_pos) { h[i] = order.back(); order.pop_back(); }
            f_hands.push_back(std::move(h));
            f_orders.push_back(std::move(order));
        }
    }

    // ---- scoring with a cache
    const std::vector<Pred>& preds(const Round& r) {
        std::string key;
        key.reserve(r.hand.size() * 4 + 24 + (carry ? nj * 12 + 96 : 0));
        key.append((const char*)r.hand.data(), r.hand.size() * sizeof(int));
        int extra[5] = {r.hl, r.dl, (int)r.draw.size(), eye ? r.types : -1, eye ? r.mouth : -1};
        key.append((const char*)extra, sizeof(extra));
        key_state(key, r.st);
        auto it = cache.find(key);
        if (it != cache.end()) { cache_hits++; return it->second; }
        calls++;
        if (r.hand.size() > 16) throw std::length_error("hand too large for the compiled scorer");
        int types = base_types, mouth = base_mouth;
        if (eye) { types = r.types; mouth = r.mouth; last_types = types; last_mouth = mouth; }
        set_state(r.st);
        double sc[2]; int hd[2], sb[2];
        int cnt = fs_best_two(fs, r.hand.data(), (int)r.hand.size(), r.hl, r.dl, (long long)r.draw.size(),
                              types, mouth, sc, hd, sb);
        if (cnt < 0) throw std::runtime_error("scorer");
        std::vector<Pred> out;
        for (int i = 0; i < cnt; i++) {
            Pred p;
            p.score = sc[i]; p.hand = hd[i];
            int pos[5];
            int k = fs_subset((int)r.hand.size(), sb[i], pos);
            if (k < 0) throw std::runtime_error("scorer");
            p.pos.assign(pos, pos + k);
            out.push_back(std::move(p));
        }
        auto res = cache.emplace(std::move(key), std::move(out));
        return res.first->second;
    }

    double score_one(const Round& r, const std::vector<int>& pos, int* hand_type) {
        // the hidden path: sg.predict_many([pos]) with the game left as the last _preds left it
        if (r.hand.size() > 16) throw std::length_error("hand too large for the compiled scorer");
        set_state(r.st);
        double sc;
        if (fs_score_one(fs, r.hand.data(), (int)r.hand.size(), pos.data(), (int)pos.size(), r.hl, r.dl,
                         (long long)r.draw.size(), last_types, last_mouth, &sc, hand_type) < 0)
            throw std::runtime_error("scorer");
        return sc;
    }

    // ---- one simulated round
    void sort_hand(std::vector<int>& h) const {
        // rank desc (stones last), then suit, then uid: a total order, so any sort gives the same result
        std::sort(h.begin(), h.end(), [this](int a, int b) {
            const PoolCard& x = pool[a];
            const PoolCard& y = pool[b];
            int rx = x.stone ? 0 : x.rank, ry = y.stone ? 0 : y.rank;
            if (rx != ry) return rx > ry;
            if (x.suit != y.suit) return x.suit < y.suit;
            return x.uid < y.uid;
        });
    }

    static bool in_pos(const std::vector<int>& pos, int i) {
        for (int p : pos) if (p == i) return true;
        return false;
    }

    void redraw(Round& r, const std::vector<int>& pos, bool after_play) {
        std::vector<int> keep;
        for (size_t i = 0; i < r.hand.size(); i++) if (!in_pos(pos, (int)i)) keep.push_back(r.hand[i]);
        if (after_play && hook && !keep.empty()) {
            size_t k = std::min<size_t>(2, keep.size());
            std::vector<size_t> idx = rng.sample_indices(keep.size(), k);
            std::vector<int> picked;
            for (size_t j : idx) picked.push_back(keep[j]);
            for (int c : picked) keep.erase(std::find(keep.begin(), keep.end(), c));
        }
        int n = serpent ? 3 : hand_size - (int)keep.size();
        size_t n_draw = std::min<size_t>((size_t)std::max(0, n), r.draw.size());
        for (size_t k = 0; k < n_draw; k++) { keep.push_back(r.draw.back()); r.draw.pop_back(); }
        sort_hand(keep);
        r.hand = std::move(keep);
    }

    bool play(Round& r, const std::vector<int>& pos, double score, int htype) {
        advance_play(r, pos, htype);
        r.chips = r.chips + score;
        r.hl -= 1;
        if (eye) {
            r.types |= 1 << htype;
            if (r.mouth < 0) r.mouth = htype;
        }
        if (r.chips >= target || r.hl <= 0) return true;
        redraw(r, pos, true);
        return r.hand.empty();
    }

    void discard(Round& r, const std::vector<int>& pos) {
        advance_discard(r, pos);
        r.dl -= 1;
        redraw(r, pos, false);
    }

    std::vector<int> rest_discard(const Round& r, const std::vector<int>& pos) const {
        // the 5 lowest cards (by chip value, a stable sort) outside `pos`, as sorted positions
        std::vector<int> rest;
        for (size_t i = 0; i < r.hand.size(); i++) if (!in_pos(pos, (int)i)) rest.push_back((int)i);
        std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) {
            return pool[r.hand[a]].chip_value < pool[r.hand[b]].chip_value;
        });
        if (rest.size() > 5) rest.resize(5);
        std::sort(rest.begin(), rest.end());
        return rest;
    }

    bool policy_step(Round& r) {
        const std::vector<Pred>& ps = preds(r);
        if (ps.empty()) return true;
        Pred p = ps[0];                   // a copy: the cache may move while we play
        double need = target - r.chips;
        if (p.score < need && r.dl > 0 && p.score * r.hl < need) {
            std::vector<int> rest = rest_discard(r, p.pos);
            if (!rest.empty()) {
                discard(r, rest);
                return false;
            }
        }
        return play(r, p.pos, p.score, p.hand);
    }

    void playout(Round& r) {
        for (int k = 0; k < max_steps; k++)
            if (policy_step(r)) break;
    }

    std::pair<double, double> max_node(const Round& r) {
        std::vector<Pred> ps = preds(r);
        struct Act { bool play; Pred p; std::vector<int> d; };
        std::vector<Act> acts;
        for (size_t i = 0; i < ps.size() && i < 2; i++) acts.push_back(Act{true, ps[i], {}});
        if (r.dl > 0) {
            std::vector<std::vector<int>> seen;
            for (size_t i = 0; i < ps.size() && i < 2; i++) {
                std::vector<int> d = rest_discard(r, ps[i].pos);
                if (!d.empty() && std::find(seen.begin(), seen.end(), d) == seen.end()) {
                    seen.push_back(d);
                    acts.push_back(Act{false, Pred{}, d});
                }
            }
        }
        std::pair<double, double> best(-1.0, 0.0);
        for (const Act& a : acts) {
            double tot_p = 0.0, tot_c = 0.0;
            for (int k = 0; k < inner; k++) {
                Round q = r;
                rng.shuffle(q.draw);
                bool over;
                if (a.play) over = play(q, a.p.pos, a.p.score, a.p.hand);
                else { discard(q, a.d); over = false; }
                if (!over) playout(q);
                tot_p += (q.chips >= target) ? 1.0 : 0.0;
                tot_c += q.chips;
            }
            std::pair<double, double> v(tot_p / inner, tot_c / inner);
            if (v > best) best = v;
        }
        return best;
    }

    // ---- one candidate on every future: (p_clear, e_chips)
    std::pair<double, double> evaluate(const std::vector<int>& pos, double score, int hand, bool is_play, int depth) {
        double tot_p = 0.0, tot_c = 0.0;
        size_t n = f_hands.size();
        for (size_t f = 0; f < n; f++) {
            Round r;
            r.hand = f_hands[f]; r.draw = f_orders[f];
            r.hl = start.hl; r.dl = start.dl; r.chips = start.chips; r.types = start.types; r.mouth = start.mouth;
            r.st = start.st;
            bool over;
            if (is_play) {
                double sc = score;
                int h = hand;
                if (hidden) sc = score_one(r, pos, &h);
                over = play(r, pos, sc, h);
            } else if (!pos.empty()) {
                discard(r, pos);
                over = false;
            } else {
                over = false;                      // no action: the playout from the start state
            }
            if (!over) {
                if (depth >= 2) {
                    std::pair<double, double> v = max_node(r);
                    tot_p += v.first;
                    tot_c += v.second;
                    continue;
                }
                playout(r);
            }
            tot_p += (r.chips >= target) ? 1.0 : 0.0;
            tot_c += r.chips;
        }
        return {tot_p / (double)n, std::min(5.0, tot_c / (double)n / target_f)};
    }
};

}  // namespace balatro
