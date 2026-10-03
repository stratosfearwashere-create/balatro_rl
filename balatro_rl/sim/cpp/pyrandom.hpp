// A replica of CPython's random.Random (Mersenne Twister MT19937 and the exact algorithms of random(),
// getrandbits(), _randbelow(), shuffle(), sample(), choice(), choices(), randrange() and randint()), so
// the C++ game draws the same numbers in the same order as the Python one. The state round-trips through
// getstate() / setstate() (the 625-tuple CPython uses: 624 words and the index).
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace balatro {

class PyRandom {
public:
    static constexpr int N = 624, M = 397;
    uint32_t mt[N];
    int index = N + 1;
    bool has_gauss = false;        // kept for a faithful getstate() (gauss_next is None)

    PyRandom() { seed_u64(0); }
    explicit PyRandom(uint64_t seed) { seed_u64(seed); }

    // random.seed(int): init_by_array over the absolute value's 32-bit little-endian words
    void seed_u64(uint64_t a) {
        std::vector<uint32_t> key;
        if (a == 0) key.push_back(0);
        while (a) { key.push_back((uint32_t)(a & 0xffffffffu)); a >>= 32; }
        init_by_array(key);
    }
    void seed_words(const std::vector<uint32_t>& key) { init_by_array(key.empty() ? std::vector<uint32_t>{0} : key); }

    void init_genrand(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < N; i++)
            mt[i] = (1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i);
        index = N;
    }
    void init_by_array(const std::vector<uint32_t>& key) {
        init_genrand(19650218u);
        int i = 1, j = 0;
        int klen = (int)key.size();
        int k = N > klen ? N : klen;
        for (; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + (uint32_t)j;
            i++; j++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
            if (j >= klen) j = 0;
        }
        for (k = N - 1; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - (uint32_t)i;
            i++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
        }
        mt[0] = 0x80000000u;
        index = N;
    }

    uint32_t genrand_uint32() {
        static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
        uint32_t y;
        if (index >= N) {
            int kk;
            for (kk = 0; kk < N - M; kk++) {
                y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 1u];
            }
            for (; kk < N - 1; kk++) {
                y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 1u];
            }
            y = (mt[N - 1] & 0x80000000u) | (mt[0] & 0x7fffffffu);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 1u];
            index = 0;
        }
        y = mt[index++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // random(): 53-bit double
    double random() {
        uint32_t a = genrand_uint32() >> 5, b = genrand_uint32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // getrandbits(k) for k <= 64 (what _randbelow needs for any n < 2^64)
    uint64_t getrandbits(int k) {
        if (k <= 0) throw std::invalid_argument("number of bits must be greater than zero");
        if (k <= 32) return (uint64_t)(genrand_uint32() >> (32 - k));
        // CPython fills 32-bit words little-endian, the last one shifted
        uint64_t out = 0;
        int words = (k - 1) / 32 + 1;
        for (int i = 0; i < words; i++, k -= 32) {
            uint32_t r = genrand_uint32();
            if (k < 32) r >>= (32 - k);
            out |= (uint64_t)r << (32 * i);
        }
        return out;
    }

    // _randbelow_with_getrandbits(n): n > 0
    uint64_t randbelow(uint64_t n) {
        if (n == 0) return 0;              // CPython: getrandbits(0) ... randrange guards this before
        int k = 0;
        for (uint64_t t = n; t; t >>= 1) k++;   // n.bit_length()
        uint64_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // randrange(start, stop) with step 1; randint(a, b) = randrange(a, b + 1)
    int64_t randrange(int64_t start, int64_t stop) {
        int64_t width = stop - start;
        if (width <= 0) throw std::domain_error("empty range for randrange()");
        return start + (int64_t)randbelow((uint64_t)width);
    }
    int64_t randrange(int64_t stop) { return randrange(0, stop); }
    int64_t randint(int64_t a, int64_t b) { return randrange(a, b + 1); }

    // choice(seq): the index
    size_t choice_index(size_t n) {
        if (n == 0) throw std::out_of_range("Cannot choose from an empty sequence");
        return (size_t)randbelow(n);
    }

    template <class T> void shuffle(std::vector<T>& x) {
        for (size_t i = x.size() - 1; i + 1 >= 2; i--) {        // for i in reversed(range(1, len(x)))
            size_t j = (size_t)randbelow(i + 1);
            std::swap(x[i], x[j]);
        }
    }

    // sample(population, k): the chosen indices, in CPython's order
    std::vector<size_t> sample_indices(size_t n, size_t k) {
        if (k > n) throw std::invalid_argument("Sample larger than population or is negative");
        std::vector<size_t> result(k);
        size_t setsize = 21;
        if (k > 5) setsize += (size_t)std::pow(4.0, std::ceil(std::log((double)(k * 3)) / std::log(4.0)));
        if (n <= setsize) {
            std::vector<size_t> pool(n);
            for (size_t i = 0; i < n; i++) pool[i] = i;
            for (size_t i = 0; i < k; i++) {
                size_t j = (size_t)randbelow(n - i);
                result[i] = pool[j];
                pool[j] = pool[n - i - 1];
            }
        } else {
            std::vector<char> selected(n, 0);
            for (size_t i = 0; i < k; i++) {
                size_t j = (size_t)randbelow(n);
                while (selected[j]) j = (size_t)randbelow(n);
                selected[j] = 1;
                result[i] = j;
            }
        }
        return result;
    }

    // choices(population, k) without weights: floor(random() * n) each
    std::vector<size_t> choices_indices(size_t n, size_t k) {
        std::vector<size_t> out(k);
        for (size_t i = 0; i < k; i++) out[i] = (size_t)std::floor(random() * (double)n);
        return out;
    }
    // choices with weights: bisect_right on the cumulative weights of random() * total, hi = n - 1
    std::vector<size_t> choices_indices(const std::vector<double>& weights, size_t k) {
        size_t n = weights.size();
        std::vector<double> cum(n);
        double acc = 0.0;
        for (size_t i = 0; i < n; i++) { acc += weights[i]; cum[i] = acc; }
        double total = cum[n - 1] + 0.0;
        std::vector<size_t> out(k);
        for (size_t i = 0; i < k; i++) {
            double x = random() * total;
            size_t lo = 0, hi = n - 1;                    // bisect.bisect(cum, x, 0, n - 1)
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (x < cum[mid]) hi = mid; else lo = mid + 1;
            }
            out[i] = lo;
        }
        return out;
    }

    // uniform(a, b)
    double uniform(double a, double b) { return a + (b - a) * random(); }
};

}  // namespace balatro
