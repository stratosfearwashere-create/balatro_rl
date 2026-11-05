// Balatro's seeded random numbers: LuaJIT's math.random (the tw223 generator) and the game's pseudohash / pseudoseed /
// pseudorandom / pseudoshuffle on top of it. The same algorithms as balatro_rl/sim/seedrng.py (verified there against the
// game's own LuaJIT); the exact-seed mode of the engine draws the shop, bosses, tags, packs and the deck shuffle from
// named streams (Game::sg) instead of one generator.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace bal {

struct LuaRand {
    uint64_t u[4] = {0, 0, 0, 0};
    static uint64_t d2u(double d) { uint64_t x; std::memcpy(&x, &d, 8); return x; }
    static double u2d(uint64_t x) { double d; std::memcpy(&d, &x, 8); return d; }
    uint64_t step() {
        static const int P[4][4] = {{0, 63, 31, 18}, {1, 58, 19, 28}, {2, 55, 24, 7}, {3, 47, 21, 8}};
        uint64_t r = 0;
        for (int n = 0; n < 4; n++) {
            const int i = P[n][0], k = P[n][1], q = P[n][2], s = P[n][3];
            uint64_t z = u[i];
            z = ((((z << q) ^ z) >> (k - s)) ^ ((z & (~0ull << (64 - k))) << s));
            r ^= z;
            u[i] = z;
        }
        return r;
    }
    void seed(double d) {                                   // math.randomseed(d)
        uint32_t r = 0x11090601u;
        for (int i = 0; i < 4; i++) {
            uint64_t m = 1ull << (r & 255);
            r >>= 8;
            d = d * 3.14159265358979323846 + 2.7182818284590452354;
            uint64_t x = d2u(d);
            if (x < m) x += m;
            u[i] = x;
        }
        for (int i = 0; i < 10; i++) step();
    }
    double random() { return u2d((step() & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull) - 1.0; }
    int randint(int lo, int hi) { return (int)(random() * (double)(hi - lo + 1)) + lo; }     // math.random(lo, hi)
};

inline double pseudohash(const std::string& s) {
    double num = 1.0;
    for (int i = (int)s.size(); i >= 1; i--) {
        double t = (1.1239285023 / num) * (double)(unsigned char)s[i - 1];
        t = t * 3.141592653589793;
        t = t + 3.141592653589793 * (double)i;
        num = std::fmod(t, 1.0);
    }
    return num;
}

// G.GAME.pseudorandom: the per-key states of one run
// the named streams' states: a flat vector (copying a game copies it: one allocation instead of one per key), looked up by a hash of the key
struct FlatStreams {
    using Item = std::pair<std::string, double>;
    std::vector<Item> v;
    std::vector<uint64_t> h;
    static uint64_t hk(const std::string& s) {
        uint64_t x = 1469598103934665603ull;
        for (unsigned char c : s) { x ^= c; x *= 1099511628211ull; }
        return x;
    }
    using iterator = std::vector<Item>::iterator;
    using const_iterator = std::vector<Item>::const_iterator;
    iterator begin() { return v.begin(); }
    iterator end() { return v.end(); }
    const_iterator begin() const { return v.begin(); }
    const_iterator end() const { return v.end(); }
    iterator find(const std::string& k) {
        uint64_t x = hk(k);
        for (size_t i = 0; i < h.size(); i++) if (h[i] == x && v[i].first == k) return v.begin() + (std::ptrdiff_t)i;
        return v.end();
    }
    const_iterator find(const std::string& k) const {
        uint64_t x = hk(k);
        for (size_t i = 0; i < h.size(); i++) if (h[i] == x && v[i].first == k) return v.begin() + (std::ptrdiff_t)i;
        return v.end();
    }
    std::pair<iterator, bool> emplace(const std::string& k, double val) {
        auto it = find(k);
        if (it != v.end()) return {it, false};
        v.emplace_back(k, val);
        h.push_back(hk(k));
        return {v.end() - 1, true};
    }
    size_t count(const std::string& k) const { return find(k) != v.end() ? 1 : 0; }
    size_t size() const { return v.size(); }
    void clear() { v.clear(); h.clear(); }
    bool operator==(const FlatStreams& o) const {
        if (v.size() != o.v.size()) return false;
        for (const Item& it : v) { auto j = o.find(it.first); if (j == o.v.end() || j->second != it.second) return false; }
        return true;
    }
    bool operator!=(const FlatStreams& o) const { return !(*this == o); }
};

struct Streams {
    std::string seed;
    FlatStreams st;
    double hashed_seed = 0.0;
    LuaRand rng;                                            // the global math.random state
    void init(const std::string& s) { seed = s; st.clear(); hashed_seed = pseudohash(s); }
    double pseudoseed(const std::string& key) {
        auto it = st.find(key);
        if (it == st.end()) it = st.emplace(key, pseudohash(key + seed)).first;
        double v = std::fmod(2.134453429141 + it->second * 1.72431234, 1.0);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.13f", v);
        it->second = std::fabs(std::strtod(buf, nullptr));
        return (it->second + hashed_seed) / 2;
    }
    double pseudorandom(const std::string& key) { rng.seed(pseudoseed(key)); return rng.random(); }
    int pseudorandom(const std::string& key, int lo, int hi) { rng.seed(pseudoseed(key)); return rng.randint(lo, hi); }
    template <class T> void shuffle(std::vector<T>& items, const std::string& key) {
        rng.seed(pseudoseed(key));
        for (int i = (int)items.size(); i >= 2; i--) {
            int j = rng.randint(1, i);
            std::swap(items[i - 1], items[j - 1]);
        }
    }
};

// the 8-character seed of a run from an integer (the same mapping in Python: balatro_rl/sim/seedrng.py seed_string)
inline std::string seed_string(uint64_t n) {
    static const char A[] = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";
    std::string s(8, '1');
    for (int i = 0; i < 8; i++) { s[i] = A[n % 34]; n /= 34; }
    return s;
}

}  // namespace bal
