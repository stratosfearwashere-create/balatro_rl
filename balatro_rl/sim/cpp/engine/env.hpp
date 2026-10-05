// The RL environment on the C++ game: a port of balatro_rl/env.py (action layout, candidate plays and
// discards, legal-action mask, observation and action features, rewards). The observation arrays are the
// ones env.encode() makes, value for value (tests/test_engine.py), written into caller-provided memory so a
// batch of environments fills one set of arrays (VecEnv).
#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "game.hpp"

namespace bal {

// ------------------------------------------------------------------ action layout (env.py)
static const int BASE_HAND = 8;
static const int N_SUB = 218;
static const int MAX_ENUM_CARDS = 10;
static const int A_PLAY = 0;
static const int A_DISC = A_PLAY + N_SUB;
static const int A_SELECT = A_DISC + N_SUB;
static const int A_SKIP = A_SELECT + 1;
static const int A_REROLL_BOSS = A_SKIP + 1;
static const int A_BUY = A_REROLL_BOSS + 1;
static const int A_BUY_PACK = A_BUY + MAX_SHOP;
static const int A_VOUCHER = A_BUY_PACK + 2;
static const int A_REROLL = A_VOUCHER + MAX_VOUCHERS;
static const int A_LEAVE = A_REROLL + 1;
static const int A_SELL_J = A_LEAVE + 1;
static const int A_SELL_C = A_SELL_J + MAX_JOKERS;
static const int A_USE_C = A_SELL_C + MAX_CONSUMABLES;
static const int A_PICK = A_USE_C + MAX_CONSUMABLES;
static const int A_PSKIP = A_PICK + MAX_PACK;
static const int A_SWAP = A_PSKIP + 1;
static const int A_SWAP_END = A_SWAP + MAX_JOKERS - 1;
static const int A_REPLACE = A_SWAP_END;       // sell joker j, buy shop joker i: A_REPLACE + i * MAX_JOKERS + j
static const int N_ACTIONS = A_REPLACE + MAX_SHOP * MAX_JOKERS;
static const int N_ACTION_TYPES = 16;
static const int MAX_REROLLS_PER_SHOP = 8, MAX_SWAPS_PER_PHASE = 4, MAX_STEPS = 2500;

static const int F_CARD = 1 + 13 + 4 + 9 + 5 + 5 + 3;                 // 40
static const int F_JOKER = 1 + 5 + 7;                                 // 13
static const int F_ACT = N_ACTION_TYPES + 1 + N_HANDS + 5 + 6 + 13 + 8 + 2;   // 63
static const int F_GLOBAL = 196;
static const int OFF_PLAY = N_ACTION_TYPES;
static const int RATIO_COL = OFF_PLAY + 1 + N_HANDS;
static const int OFF_DISC = OFF_PLAY + 1 + N_HANDS + 5;
static const int OFF_ITEM = OFF_DISC + 6;

struct SubsetTable {
    int8_t s[N_SUB][5];
    int8_t len[N_SUB];
    int8_t type_of[N_ACTIONS];
    int8_t best_window[1 << 15];
    SubsetTable() {
        int n = 0;
        for (int k = 1; k <= 5; k++) {
            int idx[5];
            for (int i = 0; i < k; i++) idx[i] = i;
            while (true) {
                for (int i = 0; i < 5; i++) s[n][i] = i < k ? (int8_t)idx[i] : -1;
                len[n] = (int8_t)k;
                n++;
                int i = k - 1;
                while (i >= 0 && idx[i] == BASE_HAND - k + i) i--;
                if (i < 0) break;
                idx[i]++;
                for (int j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
            }
        }
        static const int bounds[17] = {A_PLAY, A_DISC, A_SELECT, A_SKIP, A_REROLL_BOSS, A_BUY, A_BUY_PACK, A_VOUCHER,
                                       A_REROLL, A_LEAVE, A_SELL_J, A_SELL_C, A_USE_C, A_PICK, A_PSKIP, A_SWAP,
                                       A_SWAP_END};
        for (int t = 0; t < 16; t++)
            for (int a = bounds[t]; a < bounds[t + 1]; a++) type_of[a] = (int8_t)t;
        for (int a = A_REPLACE; a < N_ACTIONS; a++) type_of[a] = 5;      // a replace is a buy in the feature one-hot
        for (int m = 0; m < (1 << 15); m++) {
            int best = 0;
            for (int sh = 1; sh < 11; sh++) {
                int v = (m >> sh) & 31, c = 0;
                while (v) { c += v & 1; v >>= 1; }
                if (c > best) best = c;
            }
            best_window[m] = (int8_t)best;
        }
    }
};
inline const SubsetTable& subset_table() {
    static const SubsetTable t;
    return t;
}

// where an observation is written
struct ObsPtr {
    float* g;            // [F_GLOBAL]
    float* cards;        // [MAX_HAND, F_CARD]
    int64_t* jid;        // [MAX_JOKERS]
    float* jf;           // [MAX_JOKERS, F_JOKER]
    int64_t* cid;        // [MAX_CONSUMABLES]
    int64_t* aid;        // [N_ACTIONS]
    float* af;           // [N_ACTIONS, F_ACT]
    bool* mask;          // [N_ACTIONS]
    int64_t* subs;       // [2 * N_SUB, 5]
    int64_t* aid2;       // [N_ACTIONS]: replace rows, the joker given up
};
struct Obs {
    float g[F_GLOBAL];
    float cards[MAX_HAND * F_CARD];
    int64_t jid[MAX_JOKERS];
    float jf[MAX_JOKERS * F_JOKER];
    int64_t cid[MAX_CONSUMABLES];
    int64_t aid[N_ACTIONS];
    float af[N_ACTIONS * F_ACT];
    bool mask[N_ACTIONS];
    int64_t subs[2 * N_SUB * 5];
    int64_t aid2[N_ACTIONS];
    ObsPtr ptr() { return ObsPtr{g, cards, jid, jf, cid, aid, af, mask, subs, aid2}; }
};

struct StepResult {
    double reward = 0.0;
    bool done = false;
};

inline double log_feat(double x, double scale = 6.0) { return std::log10((x > 0 ? x : 0) + 1) / scale; }

// multiplier jokers a training game may start with (Env::start_xmult): no legendaries, nothing that eats jokers
static const int16_t START_XMULT[] = {
    J_DUO, J_TRIO, J_FAMILY, J_ORDER, J_TRIBE, J_BARON, J_ANCIENT, J_OBELISK, J_CAVENDISH, J_CARD_SHARP,
    J_LOYALTY_CARD, J_ACROBAT, J_BLACKBOARD, J_FLOWER_POT, J_SEEING_DOUBLE, J_IDOL, J_CONSTELLATION, J_RAMEN,
    J_STENCIL, J_PHOTOGRAPH, J_BLOODSTONE, J_VAMPIRE, J_HOLOGRAM, J_STEEL_JOKER, J_LUCKY_CAT, J_BASEBALL,
    J_CAMPFIRE, J_THROWBACK, J_GLASS, J_HIT_THE_ROAD, J_DRIVERS_LICENSE};
static const int N_START_XMULT = (int)(sizeof(START_XMULT) / sizeof(START_XMULT[0]));

struct Env {
    int deck, stake, win_ante;
    double start_xmult = 0.0;        // share of games that start holding one random multiplier joker (training aid)
    double start_any = 0.0;          // share that start holding one random joker of any kind but legendary:
                                     //   rarity common 30% / uncommon 40% / rare 30% (the ones a run seldom meets)
    double verdant_bonus = 0.0;      // reward, once, for the sale that lifts Verdant Leaf's debuff
    double weights[24];
    double win_bonus, shape_chips, shape_gamma;
    Game g;
    int steps = 0, rerolls = 0, swaps = 0;
    double phi = 0.0;
    Obs* own = nullptr;              // storage when no external arrays were given
    ObsPtr p;
    // the candidates of the last encode (what an action index means)
    int8_t plays[N_SUB][5], discs[N_SUB][5];
    int8_t play_len[N_SUB], disc_len[N_SUB];        // 0: unused slot
    double pred_sc[N_SUB];
    int pred_hand[N_SUB];

    Env(int deck_ = D_RED, int stake_ = 7, double ante_weight = 1.0, double win_bonus_ = 10.0, int win_ante_ = 8,
        double shape_chips_ = 0.0, double shape_gamma_ = 0.995)
        : deck(deck_), stake(stake_), win_ante(win_ante_), win_bonus(win_bonus_), shape_chips(shape_chips_),
          shape_gamma(shape_gamma_), g(0, deck_, stake_, win_ante_) {
        double raw[24], sum = 0.0;
        for (int b = 0; b < 24; b++) { raw[b] = 1 + (ante_weight - 1) * (b / 3) / 7; sum += raw[b]; }
        double scale = 24 / sum;
        for (int b = 0; b < 24; b++) weights[b] = raw[b] * scale;
        own = new Obs();
        p = own->ptr();
    }
    Env(const Env&) = delete;
    Env& operator=(const Env&) = delete;
    ~Env() { delete own; }
    void bind(const ObsPtr& ptr) { p = ptr; }
    Obs& obs_ref() { return *own; }

    double potential() const {
        if (g.done()) return 0.0;
        double ph = 0.0;
        if (shape_chips != 0.0 && g.state == ST_SELECTING_HAND && g.target > 0) {
            int b = 3 * (g.ante - 1) + g.blind_idx;
            if (b + 1 > g.furthest_blind) ph += shape_chips * weights[std::min(b, 23)] * std::min(1.0, g.chips / g.target);
        }
        return ph;
    }
    void reset(uint64_t seed) {
        g = Game(seed, deck, stake, win_ante);
        if (start_xmult > 0.0) {
            PyRandom r(seed ^ 0x9e3779b97f4a7c15ull);       // its own stream: the choice does not shift the game's draws
            if (r.random() < start_xmult) {
                Joker j;
                j.key = START_XMULT[r.choice_index((size_t)N_START_XMULT)];
                j.base_cost = j.d().cost;
                g.add_joker(j);
            }
        }
        if (start_any > 0.0) {
            PyRandom r(seed ^ 0xc2b2ae3d27d4eb4full);
            if (r.random() < start_any) {
                double x = r.random();
                int rarity = x < 0.3 ? 1 : (x < 0.7 ? 2 : 3);
                int16_t opts[J_COUNT];
                int n = 0;
                for (int k = 0; k < J_COUNT; k++) if (JOKER_DEFS[k].rarity == rarity) opts[n++] = (int16_t)k;
                Joker j;
                j.key = opts[r.choice_index((size_t)n)];
                j.base_cost = j.d().cost;
                g.add_joker(j);
            }
        }
        steps = 0; rerolls = 0; swaps = 0;
        encode();
        phi = potential();
    }

    // ---------------------------------------------------------------- candidates
    static double kept_feat_key(const double* kf, int len) {
        return -(3 * kf[0] + 2 * kf[1] + 2 * kf[2] + kf[3] - kf[4] + 0.1 * len);
    }
    void kept_feats(const int8_t* sub, int len, double* out) const {
        int suit[4] = {0, 0, 0, 0}, ranks[15] = {0};
        int n_kept = 0;
        const std::vector<int>& hand = g.hand;
        for (int i = 0; i < (int)hand.size(); i++) {
            bool in = false;
            for (int k = 0; k < len; k++) if (sub[k] == i) in = true;
            const Card& c = g.cards[hand[i]];
            if (in || c.hidden) continue;
            if (c.enh == E_STONE || c.enh == E_HIDDEN) continue;
            if (c.enh == E_WILD) { suit[0]++; suit[1]++; suit[2]++; suit[3]++; }
            else suit[c.suit]++;
            ranks[c.rank]++;
            n_kept++;
        }
        int present = 0;
        for (int r = 2; r < 15; r++) if (ranks[r]) present |= 1 << r;
        if (present & (1 << 14)) present |= 2;
        int best_window = subset_table().best_window[present];
        int ms = std::max(std::max(suit[0], suit[1]), std::max(suit[2], suit[3]));
        int mr = 0, pairs = 0;
        for (int r = 0; r < 15; r++) { if (ranks[r] > mr) mr = ranks[r]; if (ranks[r] >= 2) pairs++; }
        int chips = 0;
        for (int k = 0; k < len; k++) chips += g.cards[hand[sub[k]]].chip_value();
        out[0] = ms / 5.0;
        out[1] = mr / 4.0;
        out[2] = best_window / 5.0;
        out[3] = pairs / 3.0;
        out[4] = (double)chips / (11 * std::max(1, len));
        out[5] = n_kept / 8.0;
    }
    static void clear_slots(int8_t (*s)[5], int8_t* len) {
        for (int i = 0; i < N_SUB; i++) { len[i] = 0; for (int k = 0; k < 5; k++) s[i][k] = -1; }
    }
    // all k-subsets (k = lo..hi) of `pool` (ascending positions), in itertools.combinations order
    template <class F> static void for_combinations(const std::vector<int>& pool, int lo, int hi, F f) {
        int n = (int)pool.size();
        for (int k = lo; k <= hi; k++) {
            if (k > n) break;
            int idx[5];
            for (int i = 0; i < k; i++) idx[i] = i;
            while (true) {
                int8_t s[5];
                for (int i = 0; i < k; i++) s[i] = (int8_t)pool[idx[i]];
                f(s, k);
                int i = k - 1;
                while (i >= 0 && idx[i] == n - k + i) i--;
                if (i < 0) break;
                idx[i]++;
                for (int j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
            }
        }
    }
    void candidates() {
        const SubsetTable& T = subset_table();
        clear_slots(plays, play_len);
        clear_slots(discs, disc_len);
        int n = (int)g.hand.size();
        int forced = g.forced_pos();
        Plan plan;
        g.make_plan(plan);
        Game::View view;
        g.hand_view(view);
        if (n <= BASE_HAND) {
            for (int i = 0; i < N_SUB; i++) {
                int len = T.len[i];
                if (T.s[i][len - 1] >= n) continue;
                if (forced >= 0) {
                    bool in = false;
                    for (int k = 0; k < len; k++) if (T.s[i][k] == forced) in = true;
                    if (!in) continue;
                }
                play_len[i] = (int8_t)len;
                int pos[5];
                for (int k = 0; k < len; k++) { plays[i][k] = T.s[i][k]; pos[k] = T.s[i][k]; }
                pred_sc[i] = g.predict(pos, len, plan, view, pred_hand[i]);
                if (g.discards_left > 0) {
                    disc_len[i] = (int8_t)len;
                    for (int k = 0; k < len; k++) discs[i][k] = T.s[i][k];
                }
            }
            return;
        }
        // a big hand: plays over the (at most) 10 most useful cards, discards over the 10 least useful
        int suit_cnt[4] = {0, 0, 0, 0}, rank_cnt[16] = {0};
        for (int id : g.hand) {
            const Card& c = g.cards[id];
            for (int s = 0; s < 4; s++) if (c.has_suit(s)) suit_cnt[s]++;
            rank_cnt[c.rank]++;
        }
        std::vector<double> use(n);
        for (int i = 0; i < n; i++) {
            const Card& c = g.cards[g.hand[i]];
            double v = g.card_value(c);
            if (!c.is_stone()) v += 3 * suit_cnt[c.suit] + 8 * rank_cnt[c.rank];
            use[i] = v;
        }
        std::vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&use](int a, int b) { return use[a] > use[b]; });
        std::vector<int> pool(order.begin(), order.begin() + std::min(n, MAX_ENUM_CARDS));
        std::vector<int> low(order.end() - std::min(n, MAX_ENUM_CARDS), order.end());
        std::sort(pool.begin(), pool.end());
        std::sort(low.begin(), low.end());
        if (forced >= 0) {
            if (std::find(pool.begin(), pool.end(), forced) == pool.end()) {
                pool.pop_back();
                pool.push_back(forced);
                std::sort(pool.begin(), pool.end());
            }
            if (std::find(low.begin(), low.end(), forced) == low.end()) {
                low.erase(low.begin());
                low.push_back(forced);
                std::sort(low.begin(), low.end());
            }
        }
        struct Scored { double neg; int len; int8_t s[5]; double sc; int hand; };
        std::vector<Scored> scored;
        for_combinations(pool, 1, 5, [&](const int8_t* s, int k) {
            if (forced >= 0) {
                bool in = false;
                for (int q = 0; q < k; q++) if (s[q] == forced) in = true;
                if (!in) return;
            }
            Scored x;
            int pos[5];
            for (int q = 0; q < 5; q++) x.s[q] = q < k ? s[q] : -1;
            for (int q = 0; q < k; q++) pos[q] = s[q];
            x.len = k;
            x.sc = g.predict(pos, k, plan, view, x.hand);
            x.neg = -x.sc;
            scored.push_back(x);
        });
        std::stable_sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
            if (a.neg != b.neg) return a.neg < b.neg;
            if (a.len != b.len) return a.len < b.len;
            for (int q = 0; q < a.len; q++) if (a.s[q] != b.s[q]) return a.s[q] < b.s[q];
            return false;
        });
        for (int i = 0; i < N_SUB && i < (int)scored.size(); i++) {
            play_len[i] = (int8_t)scored[i].len;
            for (int q = 0; q < 5; q++) plays[i][q] = scored[i].s[q];
            pred_sc[i] = scored[i].sc;
            pred_hand[i] = scored[i].hand;
        }
        if (g.discards_left > 0) {
            struct D { double key; int len; int8_t s[5]; };
            std::vector<D> ds;
            for_combinations(low, 1, 5, [&](const int8_t* s, int k) {
                if (forced >= 0) {
                    bool in = false;
                    for (int q = 0; q < k; q++) if (s[q] == forced) in = true;
                    if (!in) return;
                }
                D x;
                for (int q = 0; q < 5; q++) x.s[q] = q < k ? s[q] : -1;
                x.len = k;
                double kf[6];
                kept_feats(s, k, kf);
                x.key = kept_feat_key(kf, k);
                ds.push_back(x);
            });
            std::stable_sort(ds.begin(), ds.end(), [](const D& a, const D& b) { return a.key < b.key; });
            for (int i = 0; i < N_SUB && i < (int)ds.size(); i++) {
                disc_len[i] = (int8_t)ds[i].len;
                for (int q = 0; q < 5; q++) discs[i][q] = ds[i].s[q];
            }
        }
    }
    void target_candidates() {
        const SubsetTable& T = subset_table();
        clear_slots(plays, play_len);
        clear_slots(discs, disc_len);
        const std::vector<int>& cs = g.target_cards();
        int lo, hi;
        Game::target_range(g.targeting_cons, lo, hi);
        int n = (int)cs.size();
        int name = g.targeting_cons.id;
        bool aura = name == C_AURA;
        auto ok = [&](const int8_t* s, int len) {
            if (!aura) return true;
            for (int k = 0; k < len; k++) if (g.cards[cs[s[k]]].ed != ED_NONE) return false;
            return true;
        };
        if (n <= BASE_HAND) {
            for (int i = 0; i < N_SUB; i++) {
                int len = T.len[i];
                if (T.s[i][len - 1] < n && lo <= len && len <= hi && ok(T.s[i], len)) {
                    play_len[i] = (int8_t)len;
                    for (int k = 0; k < len; k++) plays[i][k] = T.s[i][k];
                }
            }
            return;
        }
        std::vector<int> autos = g.auto_targets(name, cs);
        std::vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            bool na = !Game::in_pile(autos, cs[a]), nb = !Game::in_pile(autos, cs[b]);
            if (na != nb) return !na && nb;
            double va = -g.card_value(g.cards[cs[a]]), vb = -g.card_value(g.cards[cs[b]]);
            return va < vb;
        });
        int take = hi <= 2 ? n : std::min(n, MAX_ENUM_CARDS);
        std::vector<int> pool(order.begin(), order.begin() + take);
        std::sort(pool.begin(), pool.end());
        int m = 0;
        for_combinations(pool, lo, hi, [&](const int8_t* s, int k) {
            if (m >= N_SUB || !ok(s, k)) return;
            play_len[m] = (int8_t)k;
            for (int q = 0; q < k; q++) plays[m][q] = s[q];
            m++;
        });
    }

    // ---------------------------------------------------------------- features
    void item_feats(float* out, int obj_kind, const Joker* j, const Cons* c, const Card* card, long long cost) const {
        // obj_kind: 0 none, 1 joker, 2 consumable, 3 playing card
        int ed = ED_NONE;
        double eternal = 0, perish = 0, rental = 0, rarity = 0, slot_ok = 1.0, sell = 0;
        double cardf[8] = {0, 0, 0, 0, 0, 0, 0, 0}, planet[2] = {0, 0};
        if (obj_kind == 1) {
            ed = j->edition;
            eternal = j->eternal ? 1.0 : 0.0;
            perish = (j->has_perish ? j->perishable : 0) / 5.0;
            rental = j->rental ? 1.0 : 0.0;
            rarity = j->d().rarity / 3.0;
            slot_ok = (ed == ED_NEG || (int)g.jokers.size() < g.joker_slots) ? 1.0 : 0.0;
            sell = j->sell_value() / 10.0;
        } else if (obj_kind == 2) {
            slot_ok = g.cons_room() ? 1.0 : 0.0;
            if (c->kind() == CK_PLANET) {
                int h = CONS_DEFS[c->id].planet_hand;
                planet[0] = g.hand_levels[h] / 10.0;
                planet[1] = std::min(g.hand_played[h], 60) / 30.0;
            }
        } else if (obj_kind == 3) {
            cardf[0] = card->is_stone() ? 0.0 : card->rank / 14.0;
            for (int s = 0; s < 4; s++) cardf[1 + s] = (!card->is_stone() && card->suit == s) ? 1.0 : 0.0;
            cardf[5] = card->enh != E_NONE ? 1.0 : 0.0;
            cardf[6] = card->ed != ED_NONE ? 1.0 : 0.0;
            cardf[7] = card->seal != S_NONE ? 1.0 : 0.0;
            ed = card->ed;
        }
        int k = 0;
        out[k++] = (float)(cost / 10.0);
        out[k++] = g.can_afford(cost) ? 1.0f : 0.0f;
        for (int e = 0; e < 5; e++) out[k++] = ed == e ? 1.0f : 0.0f;
        out[k++] = (float)eternal; out[k++] = (float)perish; out[k++] = (float)rental; out[k++] = (float)rarity;
        out[k++] = (float)slot_ok; out[k++] = (float)sell;
        for (int q = 0; q < 8; q++) out[k++] = (float)cardf[q];
        out[k++] = (float)planet[0]; out[k++] = (float)planet[1];
    }
    static int joker_vocab(const Joker& j) { return j.hidden ? VOCAB_J_UNKNOWN : JOKER_DEFS[j.key].vocab; }

    void encode_global() {
        float* f = p.g;
        int k = 0;
        bool in_round = g.state == ST_SELECTING_HAND;
        f[k++] = (float)(g.ante / 8.0);
        for (int i = 0; i < 3; i++) f[k++] = g.blind_idx == i ? 1.0f : 0.0f;
        for (int s = 0; s < 4; s++) f[k++] = g.state == s ? 1.0f : 0.0f;
        double target = in_round ? g.target : (double)g.blind_target(std::min(g.blind_idx, 2));
        f[k++] = (float)log_feat(target);
        f[k++] = (float)(in_round ? std::min(2.0, g.chips / std::max(target, 1.0)) : 0.0);
        f[k++] = (float)log_feat(std::max(target - g.chips, 0.0));
        f[k++] = (float)(g.hands_left / 5.0);
        f[k++] = (float)(g.discards_left / 5.0);
        f[k++] = (float)(g.round_hands() / 5.0);
        f[k++] = (float)(g.round_discards() / 5.0);
        f[k++] = (float)(g.effective_hand_size() / 8.0);
        f[k++] = (float)(g.hand.size() / 8.0);
        f[k++] = (float)(std::min<long long>(g.money, 100) / 50.0);
        f[k++] = (float)log_feat((double)std::max<long long>(g.money, 0), 3);
        f[k++] = g.money < 0 ? 1.0f : 0.0f;
        f[k++] = (float)(-g.debt_limit() / 20.0);
        f[k++] = (float)(g.reroll_cost / 10.0);
        f[k++] = (float)(g.free_rerolls / 2.0);
        f[k++] = (float)(rerolls / (double)MAX_REROLLS_PER_SHOP);
        f[k++] = (float)(swaps / (double)MAX_SWAPS_PER_PHASE);
        for (int b = 0; b < B_COUNT; b++) f[k++] = g.boss == b ? 1.0f : 0.0f;
        f[k++] = g.boss_active() != B_NONE ? 1.0f : 0.0f;
        f[k++] = g.boss_disabled ? 1.0f : 0.0f;
        f[k++] = g.can_reroll_boss() ? 1.0f : 0.0f;
        int tag = (g.state == ST_BLIND_SELECT && g.blind_idx < 2) ? g.tags_offered[g.blind_idx] : -1;
        for (int t = 0; t < T_COUNT; t++) f[k++] = tag == t ? 1.0f : 0.0f;
        f[k++] = (float)(g.blinds_skipped / 5.0);
        for (int h = 0; h < N_HANDS; h++) f[k++] = (float)(g.hand_levels[h] / 10.0);
        for (int h = 0; h < N_HANDS; h++) f[k++] = (float)(std::min(g.hand_played[h], 60) / 30.0);
        for (int h = 0; h < N_HANDS; h++) f[k++] = (float)(std::min(g.hand_played_round[h], 5) / 3.0);
        int rc[13] = {0}, sc[4] = {0};
        for (int id : (in_round ? g.deck : g.full_deck)) {
            const Card& c = g.cards[id];
            if (!c.is_stone()) { rc[c.rank - 2]++; sc[c.suit]++; }
        }
        for (int r = 0; r < 13; r++) f[k++] = (float)(rc[r] / 4.0);
        for (int s = 0; s < 4; s++) f[k++] = (float)(sc[s] / 13.0);
        f[k++] = (float)(g.deck.size() / 52.0);
        f[k++] = (float)(g.full_deck.size() / 52.0);
        int ec[10] = {0}, seals[5] = {0};
        for (int id : g.full_deck) {
            const Card& c = g.cards[id];
            ec[c.enh]++;
            seals[c.seal]++;
        }
        for (int e = 1; e < 9; e++) f[k++] = (float)(ec[e] / 10.0);
        for (int s = 1; s < 5; s++) f[k++] = (float)(seals[s] / 10.0);
        f[k++] = (float)(g.joker_slots / 7.0);
        f[k++] = (float)(g.jokers.size() / 7.0);
        f[k++] = (float)(g.consumable_slots / 3.0);
        f[k++] = (float)(g.cons_used() / 3.0);
        for (int v = 0; v < V_COUNT; v++) f[k++] = g.has_voucher(v) ? 1.0f : 0.0f;
        for (int i = 0; i < 3; i++) f[k++] = (float)log_feat((double)g.blind_target(i));
        for (int pk = 0; pk < 5; pk++) f[k++] = (g.state == ST_PACK && g.pack_kind == pk) ? 1.0f : 0.0f;
        f[k++] = (float)(g.state == ST_PACK ? g.pack_picks / 2.0 : 0.0);
        f[k++] = g.deck_type == D_PLASMA ? 1.0f : 0.0f;
        f[k++] = (float)(g.tarots_used / 20.0);
        int np = 0;
        for (int q = 0; q < N_PLANETS; q++) if ((g.planets_used >> q) & 1u) np++;
        f[k++] = (float)(np / 12.0);
        if (k != F_GLOBAL) throw std::runtime_error("encode_global: wrong size");
    }
    void encode_card(float* v, const Card& c, bool forced) const {
        int k = 0;
        v[k++] = 1.0f;
        bool hidden = c.hidden;
        for (int r = 0; r < 13; r++) v[k++] = (!c.is_stone() && !hidden && c.rank - 2 == r) ? 1.0f : 0.0f;
        for (int s = 0; s < 4; s++) v[k++] = (!c.is_stone() && !hidden && c.suit == s) ? 1.0f : 0.0f;
        for (int e = 0; e < 9; e++) v[k++] = (!hidden && c.enh == e) ? 1.0f : 0.0f;
        for (int e = 0; e < 5; e++) v[k++] = (!hidden && c.ed == e) ? 1.0f : 0.0f;
        for (int e = 0; e < 5; e++) v[k++] = (!hidden && c.seal == e) ? 1.0f : 0.0f;
        v[k++] = c.deb ? 1.0f : 0.0f;
        v[k++] = forced ? 1.0f : 0.0f;
        v[k++] = hidden ? 1.0f : 0.0f;
    }
    void encode_joker(float* v, const Joker& j) const {
        int k = 0;
        v[k++] = 1.0f;
        if (j.hidden) {
            for (int q = 0; q < 11; q++) v[k++] = 0.0f;
            v[k++] = 1.0f;
            return;
        }
        for (int e = 0; e < 5; e++) v[k++] = j.edition == e ? 1.0f : 0.0f;
        v[k++] = j.eternal ? 1.0f : 0.0f;
        v[k++] = (float)((j.has_perish ? j.perishable : 0) / 5.0);
        v[k++] = j.rental ? 1.0f : 0.0f;
        v[k++] = j.debuffed ? 1.0f : 0.0f;
        v[k++] = (float)(j.sell_value() / 10.0);
        v[k++] = (float)(std::log1p(std::fabs(j.st.get(0))) / 4);
        v[k++] = 0.0f;
    }

    void legal_mask() {
        bool* m = p.mask;
        std::memset(m, 0, N_ACTIONS * sizeof(bool));
        if (g.targeting) {
            for (int i = 0; i < N_SUB; i++) m[A_PLAY + i] = play_len[i] > 0;
            return;
        }
        int st = g.state;
        static const std::vector<int> none;
        if (st == ST_SELECTING_HAND) {
            for (int i = 0; i < N_SUB; i++) {
                m[A_PLAY + i] = play_len[i] > 0;
                m[A_DISC + i] = disc_len[i] > 0 && g.discards_left > 0;
            }
        } else if (st == ST_BLIND_SELECT) {
            m[A_SELECT] = true;
            m[A_SKIP] = g.blind_idx < 2;
            m[A_REROLL_BOSS] = g.can_reroll_boss();
        } else if (st == ST_SHOP) {
            for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) m[A_BUY + i] = g.can_buy(g.shop[i]);
            for (int i = 0; i < (int)g.shop_packs.size() && i < 2; i++) m[A_BUY_PACK + i] = g.can_afford(g.shop_packs[i].cost);
            for (int i = 0; i < (int)g.shop_vouchers.size() && i < MAX_VOUCHERS; i++)
                m[A_VOUCHER + i] = g.can_afford(g.shop_vouchers[i].cost);
            m[A_REROLL] = rerolls < MAX_REROLLS_PER_SHOP && (g.free_rerolls > 0 || g.can_afford(g.reroll_cost));
            m[A_LEAVE] = true;
            if ((int)g.jokers.size() >= g.joker_slots)
                for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) {
                    const ShopItem& it = g.shop[i];
                    if (it.kind != SK_JOKER || it.joker.edition == ED_NEG) continue;
                    for (int j = 0; j < (int)g.jokers.size() && j < MAX_JOKERS; j++) {
                        const Joker& jk = g.jokers[j];
                        m[A_REPLACE + i * MAX_JOKERS + j] =
                            !jk.eternal && jk.edition != ED_NEG && g.can_afford((long long)it.cost - jk.sell_value());
                    }
                }
        } else if (st == ST_PACK) {
            for (int i = 0; i < (int)g.pack_cards.size() && i < MAX_PACK; i++) m[A_PICK + i] = g.pack_pick_ok(i);
            m[A_PSKIP] = true;
        }
        if (st == ST_SHOP || st == ST_SELECTING_HAND) {
            int nj = std::min((int)g.jokers.size(), MAX_JOKERS);
            for (int i = 0; i < nj; i++) m[A_SELL_J + i] = !g.jokers[i].eternal;
            for (int i = 0; i < (int)g.consumables.size() && i < MAX_CONSUMABLES; i++) {
                m[A_SELL_C + i] = true;
                m[A_USE_C + i] = g.consumable_usable(g.consumables[i], st == ST_SELECTING_HAND ? g.hand : none);
            }
            if (swaps < MAX_SWAPS_PER_PHASE && g.jokers.size() >= 2)
                for (int i = 0; i < nj - 1; i++) m[A_SWAP + i] = true;
        }
    }

    void put(int a, int vocab, int obj_kind, const Joker* j, const Cons* c, const Card* card, long long cost) {
        p.aid[a] = vocab;
        item_feats(p.af + (size_t)a * F_ACT + OFF_ITEM, obj_kind, j, c, card, cost);
    }
    void action_features() {
        const SubsetTable& T = subset_table();
        std::memset(p.aid, 0, N_ACTIONS * sizeof(int64_t));
        std::memset(p.af, 0, (size_t)N_ACTIONS * F_ACT * sizeof(float));
        for (int a = 0; a < N_ACTIONS; a++) p.af[(size_t)a * F_ACT + T.type_of[a]] = 1.0f;
        if (g.targeting) {
            const Cons& cons = g.targeting_cons;
            float cf[23];
            item_feats(cf, 2, nullptr, &cons, nullptr, 0);
            for (int i = 0; i < N_SUB; i++) {
                if (play_len[i] == 0) continue;
                p.aid[A_PLAY + i] = CONS_DEFS[cons.id].vocab;
                float* row = p.af + (size_t)(A_PLAY + i) * F_ACT;
                row[OFF_PLAY] = (float)(play_len[i] / 5.0);
                for (int q = 0; q < 23; q++) row[OFF_ITEM + q] = cf[q];
            }
            return;
        }
        if (g.state == ST_SELECTING_HAND) {
            double remaining = std::max(g.target - g.chips, 1.0);
            double best = 0.0;
            for (int i = 0; i < N_SUB; i++) {
                if (play_len[i] == 0) continue;
                double sc = pred_sc[i];
                int h = pred_hand[i];
                if (sc > best) best = sc;
                float* row = p.af + (size_t)(A_PLAY + i) * F_ACT;
                row[OFF_PLAY] = (float)(play_len[i] / 5.0);
                row[OFF_PLAY + 1 + h] = 1.0f;
                int b = RATIO_COL;
                row[b] = (float)std::min(5.0, sc / remaining);
                row[b + 1] = (float)log_feat(sc, 7);
                row[b + 2] = sc >= remaining ? 1.0f : 0.0f;
                row[b + 3] = g.hands_left == 1 ? 1.0f : 0.0f;
                row[b + 4] = (float)std::min(5.0, sc * g.hands_left / remaining);
            }
            for (int i = 0; i < N_SUB; i++) {
                if (disc_len[i] == 0 || !p.mask[A_DISC + i]) continue;
                float* row = p.af + (size_t)(A_DISC + i) * F_ACT;
                double kf[6];
                kept_feats(discs[i], disc_len[i], kf);
                row[OFF_PLAY] = (float)(disc_len[i] / 5.0);
                row[RATIO_COL] = (float)std::min(5.0, best / remaining);
                for (int q = 0; q < 6; q++) row[OFF_DISC + q] = (float)kf[q];
            }
        }
        if (g.state == ST_SHOP) {
            for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) {
                const ShopItem& it = g.shop[i];
                if (it.kind == SK_JOKER) put(A_BUY + i, JOKER_DEFS[it.key].vocab, 1, &it.joker, nullptr, nullptr, it.cost);
                else if (it.kind == SK_CARD) put(A_BUY + i, VOCAB_PLAYING_CARD, 3, nullptr, nullptr, &g.cards[it.card], it.cost);
                else {
                    Cons c = Game::make_cons(it.key);
                    put(A_BUY + i, CONS_DEFS[it.key].vocab, 2, nullptr, &c, nullptr, it.cost);
                }
            }
            for (int i = 0; i < (int)g.shop_packs.size() && i < 2; i++)
                put(A_BUY_PACK + i, PACK_DEFS[g.shop_packs[i].key].vocab, 0, nullptr, nullptr, nullptr, g.shop_packs[i].cost);
            for (int i = 0; i < (int)g.shop_vouchers.size() && i < MAX_VOUCHERS; i++)
                put(A_VOUCHER + i, VOUCHER_DEFS[g.shop_vouchers[i].key].vocab, 0, nullptr, nullptr, nullptr,
                    g.shop_vouchers[i].cost);
            p.af[(size_t)A_REROLL * F_ACT + OFF_ITEM] = (float)((g.free_rerolls ? 0 : g.reroll_cost) / 10.0);
            for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) {
                const ShopItem& it = g.shop[i];
                if (it.kind != SK_JOKER) continue;
                for (int j = 0; j < (int)g.jokers.size() && j < MAX_JOKERS; j++) {
                    int a = A_REPLACE + i * MAX_JOKERS + j;
                    if (p.mask[a])
                        put(a, JOKER_DEFS[it.key].vocab, 1, &it.joker, nullptr, nullptr,
                            (long long)it.cost - g.jokers[j].sell_value());
                }
            }
        }
        if (g.state == ST_BLIND_SELECT) p.af[(size_t)A_REROLL_BOSS * F_ACT + OFF_ITEM] = 1.0f;
        if (g.state == ST_PACK) {
            for (int i = 0; i < (int)g.pack_cards.size() && i < MAX_PACK; i++) {
                const PackItem& x = g.pack_cards[i];
                if (x.kind == PI_JOKER) put(A_PICK + i, JOKER_DEFS[x.joker.key].vocab, 1, &x.joker, nullptr, nullptr, 0);
                else if (x.kind == PI_CONS) put(A_PICK + i, CONS_DEFS[x.cons.id].vocab, 2, nullptr, &x.cons, nullptr, 0);
                else put(A_PICK + i, VOCAB_PLAYING_CARD, 3, nullptr, nullptr, &g.cards[x.card], 0);
            }
        }
        int nj = std::min((int)g.jokers.size(), MAX_JOKERS);
        for (int i = 0; i < nj; i++) {
            const Joker& j = g.jokers[i];
            put(A_SELL_J + i, joker_vocab(j), 1, &j, nullptr, nullptr, -(long long)j.sell_value());
            if (i + 1 < nj) p.aid[A_SWAP + i] = joker_vocab(j);
        }
        for (int i = 0; i < (int)g.consumables.size() && i < MAX_CONSUMABLES; i++) {
            const Cons& c = g.consumables[i];
            put(A_SELL_C + i, CONS_DEFS[c.id].vocab, 2, nullptr, &c, nullptr, -(long long)c.sell_value());
            put(A_USE_C + i, CONS_DEFS[c.id].vocab, 2, nullptr, &c, nullptr, 0);
        }
    }

    void encode() {
        if (g.targeting) target_candidates();
        else if (g.state == ST_SELECTING_HAND) candidates();
        else { clear_slots(plays, play_len); clear_slots(discs, disc_len); }
        legal_mask();
        action_features();
        std::memset(p.cards, 0, (size_t)MAX_HAND * F_CARD * sizeof(float));
        static const std::vector<int> none;
        const std::vector<int>& src = g.state == ST_SELECTING_HAND ? g.hand : (g.state == ST_PACK ? g.pack_hand : none);
        int forced = g.forced_pos();
        for (int i = 0; i < (int)src.size() && i < MAX_HAND; i++)
            encode_card(p.cards + (size_t)i * F_CARD, g.cards[src[i]], i == forced);
        std::memset(p.jid, 0, MAX_JOKERS * sizeof(int64_t));
        std::memset(p.jf, 0, (size_t)MAX_JOKERS * F_JOKER * sizeof(float));
        for (int i = 0; i < (int)g.jokers.size() && i < MAX_JOKERS; i++) {
            p.jid[i] = joker_vocab(g.jokers[i]);
            encode_joker(p.jf + (size_t)i * F_JOKER, g.jokers[i]);
        }
        std::memset(p.cid, 0, MAX_CONSUMABLES * sizeof(int64_t));
        for (int i = 0; i < (int)g.consumables.size() && i < MAX_CONSUMABLES; i++)
            p.cid[i] = CONS_DEFS[g.consumables[i].id].vocab;
        encode_global();
        std::memset(p.aid2, 0, N_ACTIONS * sizeof(int64_t));
        if (g.state == ST_SHOP && !g.targeting)
            for (int j = 0; j < (int)g.jokers.size() && j < MAX_JOKERS; j++)
                for (int i = 0; i < MAX_SHOP; i++)
                    if (p.mask[A_REPLACE + i * MAX_JOKERS + j]) p.aid2[A_REPLACE + i * MAX_JOKERS + j] = p.jid[j];
        for (int i = 0; i < N_SUB; i++)
            for (int q = 0; q < 5; q++) {
                p.subs[(size_t)i * 5 + q] = plays[i][q];
                p.subs[(size_t)(N_SUB + i) * 5 + q] = discs[i][q];
            }
    }

    // ---------------------------------------------------------------- stepping
    std::vector<int> subset_of(int a) const {
        std::vector<int> out;
        if (a < A_DISC) { for (int q = 0; q < play_len[a - A_PLAY]; q++) out.push_back(plays[a - A_PLAY][q]); }
        else { for (int q = 0; q < disc_len[a - A_DISC]; q++) out.push_back(discs[a - A_DISC][q]); }
        return out;
    }
    void apply_action(int a) {
        if (g.targeting) {
            if (!(A_PLAY <= a && a < A_DISC)) throw std::runtime_error("a consumable is waiting for its target cards");
            g.apply_targets(subset_of(a));
            return;
        }
        if (A_PLAY <= a && a < A_DISC) g.play(subset_of(a));
        else if (a < A_SELECT) g.discard(subset_of(a));
        else if (a == A_SELECT) g.select_blind();
        else if (a == A_SKIP) g.skip_blind();
        else if (a == A_REROLL_BOSS) g.reroll_boss();
        else if (a < A_BUY_PACK) g.buy_card(a - A_BUY);
        else if (a < A_VOUCHER) g.buy_pack(a - A_BUY_PACK);
        else if (a < A_REROLL) g.buy_voucher(a - A_VOUCHER);
        else if (a == A_REROLL) g.reroll();
        else if (a == A_LEAVE) g.leave_shop();
        else if (a < A_SELL_C) g.sell_joker(a - A_SELL_J);
        else if (a < A_USE_C) g.sell_consumable(a - A_SELL_C);
        else if (a < A_PICK) g.use_consumable(a - A_USE_C, true);
        else if (a < A_PSKIP) g.pack_pick(a - A_PICK, true);
        else if (a == A_PSKIP) g.pack_skip();
        else if (a < A_SWAP_END) g.swap_jokers(a - A_SWAP);
        else if (a < N_ACTIONS) {
            int i = (a - A_REPLACE) / MAX_JOKERS, j = (a - A_REPLACE) % MAX_JOKERS;
            g.sell_joker(j);
            if (i < (int)g.shop.size() && g.can_buy(g.shop[i])) g.buy_card(i);
        }
        else throw std::runtime_error("bad action");
    }
    StepResult step(int a) {
        if (a < 0 || a >= N_ACTIONS || !p.mask[a]) throw std::runtime_error("illegal action");
        int before = g.furthest_blind;
        uint8_t prev = g.state;
        bool verdant = g.boss_active() == B_VERDANT_LEAF && g.f_verdant;
        apply_action(a);
        steps += 1;
        if (a == A_REROLL) rerolls += 1;
        if (a >= A_SWAP && a < A_SWAP_END) swaps += 1;
        if (g.state != prev) {
            if (g.state == ST_SHOP && prev != ST_PACK) rerolls = 0;
            bool shop_pack = (prev == ST_SHOP && g.state == ST_PACK) || (prev == ST_PACK && g.state == ST_SHOP);
            if (!shop_pack) swaps = 0;
        }
        double r = 0.0;
        for (int b = before; b < g.furthest_blind; b++) r += weights[std::min(b, 23)];
        if (verdant && a >= A_SELL_J && a < A_SELL_C) r += verdant_bonus;
        if (g.state == ST_WON) {
            r += win_bonus;
        } else if (g.state == ST_GAME_OVER && g.target > 0) {
            int lost = 3 * (g.ante - 1) + g.blind_idx;
            if (lost + 1 > g.furthest_blind) r += 0.5 * weights[std::min(lost, 23)] * std::min(1.0, g.chips / g.target);
        }
        bool done = g.done() || steps >= MAX_STEPS;
        if (!done) {
            encode();
            bool any = false;
            for (int i = 0; i < N_ACTIONS; i++) if (p.mask[i]) { any = true; break; }
            if (!any) { g.state = ST_GAME_OVER; done = true; }
        }
        if (shape_chips != 0.0) {
            double ph = done ? 0.0 : potential();
            r += shape_gamma * ph - phi;
            phi = ph;
        }
        StepResult res;
        res.reward = r;
        res.done = done;
        return res;
    }
};

// ------------------------------------------------------------------ many environments
// Auto-reset as vec_env.VecEnv does: environment i plays seeds seed0 + i + n * episode. After step(), the
// observation of a finished environment is the first one of its next episode; rew / done / ante / won /
// blinds describe the step just taken (the finished episode's outcome when done).
struct VecEnv {
    int n;
    uint64_t seed0;
    std::vector<Env*> envs;
    std::vector<uint64_t> episode;
    // batch buffers
    std::vector<float> b_g, b_cards, b_jf, b_af;
    std::vector<int64_t> b_jid, b_cid, b_aid, b_subs, b_aid2;
    std::vector<uint8_t> b_mask;
    // thread pool
    int threads;
    std::vector<std::thread> pool;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    uint64_t generation = 0;
    int pending = 0;
    bool stop = false;
    std::atomic<int> next{0};
    const int64_t* job_actions = nullptr;
    float* job_rew = nullptr;
    bool* job_done = nullptr;
    int32_t* job_ante = nullptr;
    bool* job_won = nullptr;
    int32_t* job_blinds = nullptr;
    std::string error;

    VecEnv(int n_, int deck, int stake, uint64_t seed0_, int threads_, double ante_weight, double win_bonus,
           int win_ante, double shape_chips, double shape_gamma, double start_xmult = 0.0,
           double start_any = 0.0, double verdant_bonus = 0.0)
        : n(n_), seed0(seed0_) {
        b_g.resize((size_t)n * F_GLOBAL);
        b_cards.resize((size_t)n * MAX_HAND * F_CARD);
        b_jid.resize((size_t)n * MAX_JOKERS);
        b_jf.resize((size_t)n * MAX_JOKERS * F_JOKER);
        b_cid.resize((size_t)n * MAX_CONSUMABLES);
        b_aid.resize((size_t)n * N_ACTIONS);
        b_af.resize((size_t)n * N_ACTIONS * F_ACT);
        b_mask.resize((size_t)n * N_ACTIONS);
        b_subs.resize((size_t)n * 2 * N_SUB * 5);
        b_aid2.resize((size_t)n * N_ACTIONS);
        episode.assign(n, 0);
        for (int i = 0; i < n; i++) {
            Env* e = new Env(deck, stake, ante_weight, win_bonus, win_ante, shape_chips, shape_gamma);
            e->start_xmult = start_xmult;
            e->start_any = start_any;
            e->verdant_bonus = verdant_bonus;
            ObsPtr ptr;
            ptr.g = b_g.data() + (size_t)i * F_GLOBAL;
            ptr.cards = b_cards.data() + (size_t)i * MAX_HAND * F_CARD;
            ptr.jid = b_jid.data() + (size_t)i * MAX_JOKERS;
            ptr.jf = b_jf.data() + (size_t)i * MAX_JOKERS * F_JOKER;
            ptr.cid = b_cid.data() + (size_t)i * MAX_CONSUMABLES;
            ptr.aid = b_aid.data() + (size_t)i * N_ACTIONS;
            ptr.af = b_af.data() + (size_t)i * N_ACTIONS * F_ACT;
            ptr.mask = (bool*)(b_mask.data() + (size_t)i * N_ACTIONS);
            ptr.subs = b_subs.data() + (size_t)i * 2 * N_SUB * 5;
            ptr.aid2 = b_aid2.data() + (size_t)i * N_ACTIONS;
            e->bind(ptr);
            envs.push_back(e);
        }
        threads = threads_ > 0 ? threads_ : (int)std::max(1u, std::thread::hardware_concurrency());
        if (threads > n) threads = n;
        for (int i = 0; i < n; i++) new_episode(i);
        if (threads > 1)
            for (int t = 0; t < threads; t++) pool.emplace_back([this]() { worker(); });
    }
    ~VecEnv() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        for (std::thread& t : pool) t.join();
        for (Env* e : envs) delete e;
    }
    // head-start shares for the games that begin from now on (the trainer fades them over a run)
    void set_start(double xmult, double any) {
        for (Env* e : envs) { e->start_xmult = xmult; e->start_any = any; }
    }
    void new_episode(int i) {
        envs[i]->reset(seed0 + (uint64_t)i + (uint64_t)n * episode[i]);
        episode[i] += 1;
    }
    void step_one(int i) {
        Env& e = *envs[i];
        StepResult r = e.step((int)job_actions[i]);
        job_rew[i] = (float)r.reward;
        job_done[i] = r.done;
        job_ante[i] = e.g.ante;
        job_won[i] = e.g.state == ST_WON;
        job_blinds[i] = e.g.furthest_blind;
        if (r.done) new_episode(i);
    }
    void run_all() {
        int i;
        while ((i = next.fetch_add(1)) < n) {
            try {
                step_one(i);
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(mu);
                if (error.empty()) error = std::string("env ") + std::to_string(i) + ": " + ex.what();
            }
        }
    }
    void worker() {
        uint64_t seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [&]() { return stop || generation != seen; });
                if (stop) return;
                seen = generation;
            }
            run_all();
            {
                std::lock_guard<std::mutex> lk(mu);
                pending -= 1;
            }
            cv_done.notify_one();
        }
    }
    void step(const int64_t* actions, float* rew, bool* done, int32_t* ante, bool* won, int32_t* blinds) {
        job_actions = actions; job_rew = rew; job_done = done; job_ante = ante; job_won = won; job_blinds = blinds;
        error.clear();
        next.store(0);
        if (threads <= 1) {
            run_all();
        } else {
            {
                std::lock_guard<std::mutex> lk(mu);
                pending = threads;
                generation += 1;
            }
            cv_work.notify_all();
            std::unique_lock<std::mutex> lk(mu);
            cv_done.wait(lk, [&]() { return pending == 0; });
        }
        if (!error.empty()) throw std::runtime_error(error);
    }
};

}  // namespace bal
