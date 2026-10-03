// hands.evaluate: poker hand detection with Four Fingers, Shortcut and Smeared Joker, for up to 5 cards
// (the Python version takes any number; the game only ever evaluates 1 to 5 played cards). Same results as
// hands.py: the hand, the scoring positions and the set of contained hands, as bitmasks.
#pragma once
#include "card.hpp"
#include <algorithm>

namespace balatro {

enum Hand { HC = 0, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH, FIVE_KIND,
            FLUSH_HOUSE, FLUSH_FIVE, N_HANDS };

struct HandEval {
    int hand;
    int scoring;      // bitmask over the played positions
    int contains;     // bitmask over hand types
};

inline bool straight_ok(int* r, int n, int maxgap) {
    std::sort(r, r + n);
    for (int i = 0; i < n - 1; i++)
        if (!(0 < r[i + 1] - r[i] && r[i + 1] - r[i] <= maxgap)) return false;
    return true;
}

inline bool is_straight_ranks(const int* ranks, int n, bool shortcut) {
    int r[5];
    for (int i = 0; i < n; i++) r[i] = ranks[i];
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (r[i] == r[j]) return false;
    int maxgap = shortcut ? 2 : 1;
    bool ace = false;
    for (int i = 0; i < n; i++) if (r[i] == 14) ace = true;
    if (straight_ok(r, n, maxgap)) return true;
    if (ace) {
        for (int i = 0; i < n; i++) r[i] = ranks[i] == 14 ? 1 : ranks[i];
        if (straight_ok(r, n, maxgap)) return true;
    }
    return false;
}

template <class C>   // C: anything with rank, suit, enh, deb and is_stone()
inline HandEval evaluate(const C* const* cs, int n, bool four_fingers, bool shortcut, bool smeared) {
    int need = four_fingers ? 4 : 5;
    int normal[5], nn = 0, stones = 0;
    int cnt[16] = {0};
    for (int i = 0; i < n; i++) {
        if (cs[i]->is_stone()) stones |= 1 << i;
        else { normal[nn++] = i; cnt[cs[i]->rank]++; }
    }
    int g0r = -1, g0c = 0, g1r = -1, g1c = 0, distinct = 0;
    for (int r = 14; r >= 0; r--) {               // groups ordered by (-count, -rank)
        if (cnt[r] > 0) distinct++;
        if (cnt[r] > g0c) { g0c = cnt[r]; g0r = r; }
    }
    for (int r = 14; r >= 0; r--)
        if (r != g0r && cnt[r] > g1c) { g1c = cnt[r]; g1r = r; }
    int top = g0c, second = g1c;

    int fmask = 0;
    if (nn >= need) {
        int ns[4] = {0, 0, 0, 0}, wild = 0;
        for (int j = 0; j < nn; j++) {
            const C* c = cs[normal[j]];
            if (c->enh == E_WILD && !c->deb) wild++; else ns[c->suit]++;
        }
        if (smeared) { ns[0] = ns[2] = ns[0] + ns[2]; ns[1] = ns[3] = ns[1] + ns[3]; }
        int best_s = -1;
        for (int s = 0; s < 4; s++) if (ns[s] + wild >= need) { best_s = s; break; }
        if (best_s >= 0)
            for (int j = 0; j < nn; j++) {
                const C* c = cs[normal[j]];
                if ((c->enh == E_WILD && !c->deb) || (smeared ? c->suit % 2 == best_s % 2 : c->suit == best_s))
                    fmask |= 1 << normal[j];
            }
    }
    bool is_flush = fmask != 0;

    int smask = 0;
    if (nn >= need && distinct >= need) {
        for (int m = 1; m < (1 << nn); m++) {
            int ranks[5], k = 0;
            for (int j = 0; j < nn; j++) if (m & (1 << j)) ranks[k++] = cs[normal[j]]->rank;
            if (k < need || k > 5) continue;
            if (is_straight_ranks(ranks, k, shortcut))
                for (int j = 0; j < nn; j++) if (m & (1 << j)) smask |= 1 << normal[j];
        }
    }
    bool is_straight = smask != 0;

    int contains = 1 << HC;
    if (top >= 2) contains |= 1 << PAIR;
    if (top >= 3) contains |= 1 << TRIPS;
    if (top >= 4) contains |= 1 << QUADS;
    if (top >= 5) contains |= 1 << FIVE_KIND;
    if (top >= 2 && second >= 2) contains |= 1 << TWO_PAIR;
    if (is_straight) contains |= 1 << STRAIGHT;
    if (is_flush) contains |= 1 << FLUSH;
    if (is_straight && is_flush) contains |= 1 << STRAIGHT_FLUSH;
    bool full = top >= 3 && second >= 2;
    if (full) contains |= 1 << FULL_HOUSE;

    auto rank_mask = [&](int r1, int r2) {
        int m = 0;
        for (int j = 0; j < nn; j++) {
            int r = cs[normal[j]]->rank;
            if (r == r1 || r == r2) m |= 1 << normal[j];
        }
        return m;
    };
    int hand, sc = 0;
    if (top >= 5 && is_flush) { hand = FLUSH_FIVE; sc = rank_mask(g0r, -1); }
    else if (full && is_flush) { hand = FLUSH_HOUSE; sc = rank_mask(g0r, g1r); }
    else if (top >= 5) { hand = FIVE_KIND; sc = rank_mask(g0r, -1); }
    else if (is_straight && is_flush) { hand = STRAIGHT_FLUSH; sc = fmask | smask; }
    else if (top >= 4) { hand = QUADS; sc = rank_mask(g0r, -1); }
    else if (full) { hand = FULL_HOUSE; sc = rank_mask(g0r, g1r); }
    else if (is_flush) { hand = FLUSH; sc = fmask; }
    else if (is_straight) { hand = STRAIGHT; sc = smask; }
    else if (top >= 3) { hand = TRIPS; sc = rank_mask(g0r, -1); }
    else if (top >= 2 && second >= 2) { hand = TWO_PAIR; sc = rank_mask(g0r, g1r); }
    else if (top >= 2) { hand = PAIR; sc = rank_mask(g0r, -1); }
    else {
        hand = HC;
        if (nn > 0) {
            int k = normal[0];
            for (int j = 1; j < nn; j++) if (cs[normal[j]]->rank > cs[k]->rank) k = normal[j];
            sc = 1 << k;
        }
    }
    if (hand == FLUSH_FIVE) contains |= (1 << FLUSH) | (1 << FIVE_KIND);
    else if (hand == FLUSH_HOUSE) contains |= (1 << FLUSH) | (1 << FULL_HOUSE);
    return HandEval{hand, sc | stones, contains};
}

}  // namespace balatro
