#include <unordered_set>
// The RL environment on the C++ game: a port of balatro_rl/env.py (action layout, candidate plays and
// discards, legal-action mask, observation and action features, rewards). The observation arrays are the
// ones env.encode() makes, value for value (tests/test_engine.py), written into caller-provided memory so a
// batch of environments fills one set of arrays (VecEnv).
#pragma once
#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "game.hpp"

namespace bal {

// ------------------------------------------------------------------ action layout (env.py)
static const int BASE_HAND = 8;
static const int N_SUB = 218;
static const int MAX_ENUM_CARDS = 10;
// POT_MODE 1: the growth potential from the growth model's measured per-round rates (balatro_rl/growth.py), over POT_ROUNDS rounds
static int POT_MODE = 0, POT_ROUNDS = 6;
static inline bool pot2_rate(int k, double& rate, double& v0) {
    switch (k) {
        case J_GREEN_JOKER: rate = 1.51; v0 = 0.0; return true;
        case J_RIDE_THE_BUS: rate = 1.0; v0 = 0.0; return true;
        case J_RED_CARD: rate = 1.5; v0 = 0.0; return true;
        case J_FLASH: rate = 0.8; v0 = 0.0; return true;
        case J_RUNNER: rate = 4.08; v0 = 0.0; return true;
        case J_SQUARE: rate = 3.0; v0 = 0.0; return true;
        case J_CASTLE: rate = 4.12; v0 = 0.0; return true;
        case J_WEE: rate = 5.6; v0 = 0.0; return true;
        case J_TROUSERS: rate = 1.46; v0 = 0.0; return true;
        case J_CONSTELLATION: rate = 0.045; v0 = 1.0; return true;
        case J_HOLOGRAM: rate = 0.078; v0 = 1.0; return true;
        case J_VAMPIRE: rate = 0.006; v0 = 1.0; return true;
        case J_OBELISK: rate = 0.1; v0 = 1.0; return true;
        case J_LUCKY_CAT: rate = 0.005; v0 = 1.0; return true;
        case J_GLASS: rate = 0.0064; v0 = 1.0; return true;
        case J_CAMPFIRE: rate = 0.089; v0 = 1.0; return true;
        case J_MADNESS: rate = 0.333; v0 = 1.0; return true;
        case J_CEREMONIAL: rate = 1.6; v0 = 0.0; return true;
        case J_CAINO: rate = 0.027; v0 = 1.0; return true;
        case J_HIT_THE_ROAD: rate = -1.0; v0 = 1.0; return true;      // resets each round: a fixed delta
        default: return false;
    }
}
static inline bool pot_on(int k) { double r, v; return POT_MODE >= 1 ? pot2_rate(k, r, v) : JOKER_POT_ON[k]; }

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
// the arrays carry the marginal-value columns (MARG_*, marginal.py) after the base ones: zeros unless Env::marginal
static const int F_JOKER_BASE = 1 + 5 + 7;                            // 13
static const int F_JOKER_M = F_JOKER_BASE + 2;                        // (the potential column follows the marginal ones)
static const int F_JOKER = F_JOKER_M + 1;
static const int F_ACT_BASE = N_ACTION_TYPES + 1 + N_HANDS + 5 + 6 + 13 + 8 + 2;   // 63
static const int F_ACT_M = F_ACT_BASE + 3;
static const int F_ACT = F_ACT_M + 1;
static const int F_GLOBAL_BASE = 205;
static const int F_GLOBAL_M = F_GLOBAL_BASE + 4;
static const int F_GLOBAL = F_GLOBAL_M + 1;
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

// the privileged observation: [0,200) next shop 61, next boss 28, top 12 cards of the deck x 9; [200, 2000) the future list: for each of the
// next FUT_K antes the jokers its shop and 5 rerolls will show (1 - 0.15 x the reroll), its voucher, its packs and its boss
static const int FUT_K = 8, FUT_ROLLS = 5, FUT_W = J_COUNT + V_COUNT + N_PACKS + B_COUNT;
static const int F_PRIV = 200 + FUT_K * FUT_W;

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
    float* priv;         // [F_PRIV]: what only a critic may see (Env::encode_priv)
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
    float priv[F_PRIV];
    ObsPtr ptr() { return ObsPtr{g, cards, jid, jf, cid, aid, af, mask, subs, aid2, priv}; }
};

struct StepResult {
    double reward = 0.0;
    bool done = false;
    double sparse = 0.0;     // the part of `reward` that is the sparse stream: the win bonus, the ante-8 bonus, the blind rewards of blinds >= sparse_from_blind
    bool trunc = false;      // the step ended the episode by the step limit (not a loss or a win): the value of the state it was in is still worth something
};

inline double log_feat(double x, double scale = 6.0) { return std::log10((x > 0 ? x : 0) + 1) / scale; }

// multiplier jokers a training game may start with (Env::start_xmult): no legendaries, nothing that eats jokers
static const int16_t START_XMULT[] = {
    J_DUO, J_TRIO, J_FAMILY, J_ORDER, J_TRIBE, J_BARON, J_ANCIENT, J_OBELISK, J_CAVENDISH, J_CARD_SHARP,
    J_LOYALTY_CARD, J_ACROBAT, J_BLACKBOARD, J_FLOWER_POT, J_SEEING_DOUBLE, J_IDOL, J_CONSTELLATION, J_RAMEN,
    J_STENCIL, J_PHOTOGRAPH, J_BLOODSTONE, J_VAMPIRE, J_HOLOGRAM, J_STEEL_JOKER, J_LUCKY_CAT, J_BASEBALL,
    J_CAMPFIRE, J_THROWBACK, J_GLASS, J_HIT_THE_ROAD, J_DRIVERS_LICENSE};
static const int N_START_XMULT = (int)(sizeof(START_XMULT) / sizeof(START_XMULT[0]));
// jokers that grow over a run by their own counter (Env::start_grow): no legendaries, nothing that eats jokers
static const int16_t START_GROW[] = {
    J_RUNNER, J_SQUARE, J_CASTLE, J_WEE, J_GREEN_JOKER, J_RIDE_THE_BUS, J_RED_CARD, J_TROUSERS, J_FLASH,
    J_CONSTELLATION, J_HOLOGRAM, J_VAMPIRE, J_OBELISK, J_LUCKY_CAT, J_GLASS, J_CAMPFIRE, J_HIT_THE_ROAD};
// multiplier jokers that scale by their own counter and keep it (Env::start_xscale): the counter can be handed over pre-grown
static const int16_t START_XSCALE[] = {J_CONSTELLATION, J_HOLOGRAM, J_VAMPIRE, J_LUCKY_CAT, J_GLASS};
static const int N_START_XSCALE = (int)(sizeof(START_XSCALE) / sizeof(START_XSCALE[0]));
static const int N_START_GROW = (int)(sizeof(START_GROW) / sizeof(START_GROW[0]));

// ---- scenarios (Env::scn_p): a game that starts from a sampled build instead of from nothing
// what a joker asks of the deck: a deck theme (Env::apply_scenario) and its parameter
enum ScnTheme : int { TH_NONE = 0, TH_SUIT, TH_FACE, TH_RANKS, TH_ENH, TH_PAIRS, TH_THIN, TH_EDITION, TH_COUNT };
struct ScnNeed { int16_t key; int theme, param; };
static const ScnNeed SCN_NEED[] = {
    {J_GREEDY_JOKER, TH_SUIT, 3}, {J_LUSTY_JOKER, TH_SUIT, 1}, {J_WRATHFUL_JOKER, TH_SUIT, 0}, {J_GLUTTENOUS_JOKER, TH_SUIT, 2},
    {J_ROUGH_GEM, TH_SUIT, 3}, {J_BLOODSTONE, TH_SUIT, 1}, {J_ARROWHEAD, TH_SUIT, 0}, {J_ONYX_AGATE, TH_SUIT, 2},
    {J_SEEING_DOUBLE, TH_SUIT, 2}, {J_BLACKBOARD, TH_SUIT, 0},
    {J_SCARY_FACE, TH_FACE, 0}, {J_SMILEY, TH_FACE, 0}, {J_PHOTOGRAPH, TH_FACE, 0}, {J_SOCK_AND_BUSKIN, TH_FACE, 0},
    {J_PAREIDOLIA, TH_FACE, 0}, {J_BUSINESS, TH_FACE, 0}, {J_MIDAS_MASK, TH_FACE, 0}, {J_RESERVED_PARKING, TH_FACE, 0},
    {J_FACELESS, TH_FACE, 0}, {J_BARON, TH_FACE, 0}, {J_TRIBOULET, TH_FACE, 0}, {J_SHOOT_THE_MOON, TH_FACE, 0},
    {J_EVEN_STEVEN, TH_RANKS, 0}, {J_ODD_TODD, TH_RANKS, 1}, {J_SCHOLAR, TH_RANKS, 2}, {J_HACK, TH_RANKS, 3},
    {J_FIBONACCI, TH_RANKS, 4}, {J_WALKIE_TALKIE, TH_RANKS, 5}, {J_WEE, TH_RANKS, 6}, {J_8_BALL, TH_RANKS, 7},
    {J_STEEL_JOKER, TH_ENH, E_STEEL}, {J_GLASS, TH_ENH, E_GLASS}, {J_LUCKY_CAT, TH_ENH, E_LUCKY}, {J_STONE, TH_ENH, E_STONE},
    {J_VAMPIRE, TH_ENH, E_MULT}, {J_DRIVERS_LICENSE, TH_ENH, E_BONUS},
    {J_JOLLY, TH_PAIRS, 0}, {J_SLY, TH_PAIRS, 0}, {J_DUO, TH_PAIRS, 0}, {J_TRIO, TH_PAIRS, 0}, {J_FAMILY, TH_PAIRS, 0},
    {J_ZANY, TH_PAIRS, 0}, {J_WILY, TH_PAIRS, 0}, {J_MAD, TH_PAIRS, 0}, {J_CLEVER, TH_PAIRS, 0},
    {J_EROSION, TH_THIN, 0}};
static const int N_SCN_NEED = (int)(sizeof(SCN_NEED) / sizeof(SCN_NEED[0]));

// ---- the state archive (train.py --archive-*): games saved at the start of a blind, by stake and blind (a reservoir of `cap` for each); a later
// game may start from one of them (the late game, which a policy otherwise reaches in a handful of games, gets thousands of attempts)
struct ArchSnap { Game g; bool restricted = false, level_capped = false; };
struct Archive {
    std::mutex mu;
    int cap = 64;
    std::vector<ArchSnap> slot[9][24];
    uint64_t seen[9][24] = {};
    uint64_t starts[24] = {};                       // games started from each blind since the last read (archive_starts)
    uint64_t ends[24] = {}, wins[24] = {};          // games that ended / were won, by the blind they started from (archive_outcomes)
    uint64_t state = 0x9e3779b97f4a7c15ull;
    uint64_t next() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
};

static inline int popcount32(uint64_t x) { int n = 0; while (x) { n += (int)(x & 1u); x >>= 1; } return n; }

static uint64_t FP_CRN_SEED = 0;                         // > 0: fast_power's redrawn hands come from this seed for every state (common random numbers across candidates)
static bool CURVE_GENERIC_BOSS = false;                   // projected_share faces a generic boss (x2, no effect) instead of a random one
struct Env {
    int deck, stake, win_ante;
    double start_xscale = 0.0;       // share of games that start holding one of the scaling multiplier jokers ...
    double start_xscale_val = 1.0;   // ... with its counter (its multiplier) already at this value (a training aid, faded by the trainer)
    double start_xmult = 0.0;        // share of games that start holding one random multiplier joker (training aid)
    double start_any = 0.0;          // share that start holding one random joker of any kind but legendary:
                                     //   rarity common 30% / uncommon 40% / rare 30% (the ones a run seldom meets)
    double start_any2 = 0.0;
    bool start_any_eternal = false;  // the random starting joker(s) cannot be sold (eternal)         // of those, the share that also get a second one (drawn the same way, not the same joker)
    double verdant_bonus = 0.0;      // reward, once, for the sale that lifts Verdant Leaf's debuff
    // the joker school (train.py --school-share): in school_p of the games one growing joker, the student, is given at
    // the start as an eternal with a kit, and every new shop of the game carries one more item its chain needs
    std::vector<int16_t> school_ids;
    double school_p = 0.0;
    int school_student = -1;               // this game's student (a joker id), -1: not a school game
    int school_shop_seen = -1;
    double sc_n[J_COUNT] = {};             // finished school games and the sum of log(final counter), per student
    double sc_sumlog[J_COUNT] = {};
    double death_pen = 0.0, death_cap = 100.0;   // a lost game costs death_pen * min(money, death_cap) / death_cap
    static int standard_pack_index() {
        static int idx = -1;
        if (idx < 0) for (int i = 0; i < N_PACKS; i++) if (std::strcmp(PACK_DEFS[i].key, "p_standard_normal") == 0) idx = i;
        return idx;
    }
    // the kit: up to two consumables at the start
    static int school_kit(int student, int* out) {
        switch (student) {
            case J_VAMPIRE: out[0] = C_EMPRESS; out[1] = C_HEIROPHANT; return 2;
            case J_LUCKY_CAT: out[0] = C_MAGICIAN; out[1] = C_MAGICIAN; return 2;
            case J_GLASS: out[0] = C_JUSTICE; out[1] = C_JUSTICE; return 2;
            case J_CONSTELLATION: out[0] = C_MARS; out[1] = C_EARTH; return 2;
            default: return 0;                  // Hologram: its chain is the Standard pack on the shelf
        }
    }
    // a new shop of a school game: one more item the student's chain needs (a tarot or planet, a Standard pack)
    void school_supply() {
        if (school_student < 0 || g.state != ST_SHOP || g.shops_seen == school_shop_seen) return;
        school_shop_seen = g.shops_seen;
        int cons = -1, kind = SK_TAROT;
        switch (school_student) {
            case J_VAMPIRE: { static const int c[3] = {C_EMPRESS, C_HEIROPHANT, C_MAGICIAN}; cons = c[g.shops_seen % 3]; break; }
            case J_LUCKY_CAT: cons = C_MAGICIAN; break;
            case J_GLASS: cons = C_JUSTICE; break;
            case J_CONSTELLATION: { static const int c[4] = {C_MARS, C_EARTH, C_JUPITER, C_SATURN}; cons = c[g.shops_seen % 4]; kind = SK_PLANET; break; }
            default: break;
        }
        if (school_student == J_HOLOGRAM) {
            ShopItem b;
            b.kind = SK_PACK;
            b.key = standard_pack_index();
            b.cost = g.price(4);
            if (b.key >= 0) {
                if (g.shop_packs.empty()) g.shop_packs.push_back(b); else g.shop_packs.back() = b;
            }
        } else if (cons >= 0) {
            ShopItem it;
            it.kind = (uint8_t)kind;
            it.key = cons;
            it.cost = g.price(3);
            if ((int)g.shop.size() < g.shop_slots()) g.shop.push_back(it); else g.shop.back() = it;
        }
    }
    void school_note_end() {
        if (school_student < 0) return;
        double val = 1.0;
        for (const Joker& j : g.jokers) if (j.key == school_student) val = j.st.get(1.0);
        sc_n[school_student] += 1.0;
        sc_sumlog[school_student] += std::log(val > 1.0 ? val : 1.0);
    }
    // a shifting world (train.py --ban-top, --tilt-share): in ban_p of games the jokers in `banned` are out of
    // the pool; in tilt_p of games one random hand type (High Card .. Straight Flush) starts tilt_levels up
    std::vector<int16_t> banned;
    double ban_p = 0.0, tilt_p = 0.0;
    // the joker cap (train.py --cap-jokers, --cap-max): at most cap_max of the jokers in cap_set held at once; buying, swapping in or
    // picking from a pack one more is not a legal move (cap_max < 0: no cap)
    // (several groups, each with its own maximum: cap_set[key] = the joker's group, 1-based, 0 = none)
    std::vector<uint8_t> cap_set;
    std::vector<int> cap_maxes;
    int cap_max = -1;                      // >= 0 when any group is set
    int cap_group(int key) const { return (cap_max >= 0 && key >= 0 && key < (int)cap_set.size()) ? cap_set[key] : 0; }
    int cap_held(int grp) const { int n = 0; for (const Joker& j : g.jokers) n += cap_group(j.key) == grp; return n; }
    bool exact = false;                    // exact-seed mode: the game's named streams, the 8-character seed made from the integer
    int tilt_levels = 2;
    bool restricted = false;         // this game has the ban
    // the level cap (train.py --level-cap): in level_cap_p of the games, until ante level_cap_ante at most level_cap_n planet
    // cards may be bought, picked or used (a training aid that pushes the agent from hand levels toward joker builds)
    double level_cap_p = 0.0;
    int level_cap_n = 2, level_cap_ante = 4;
    bool level_capped_game = false;
    bool part_credit = true;         // false: a lost blind earns nothing for the chips scored in it, so the blind rewards are the potential W(furthest blind beaten) and nothing else (their sum over a game telescopes to it)
    bool log_partial = false;        // the part credit for a lost blind is on a log scale of the chips reached (log10(1 + chips) / log10(1 + target)), not linear: a smooth reward for
                                     // scoring more, whichever way the score was made (the late-game sandbox, where a linear share of a 200k target is ~0)
    int level_cap_mode = 0;          // 0: past level_cap_n planet uses planets can no longer be bought, picked or used; 1: they can, but raise no hand level past level_cap_n level-ups
    int pack_restock = 0;            // the sandbox: a pack bought is replaced by another, this many times per shop visit
    double pack_bias[5] = {1, 1, 1, 1, 1};     // ... and the weights of the pack kinds (arcana, celestial, standard, buffoon, spectral) are multiplied by these
    bool level_capped_now() const { return level_cap_mode == 0 && level_capped_game && g.ante < level_cap_ante && g.planet_uses >= level_cap_n; }
    double start_grow = 0.0;         // share of games that start holding one random growing joker, eternal: it
                                     //   cannot be sold, so the game has to be played around it (training aid)
    // training aid: a share of games start with extra money, so the agent can afford a strong build and sees
    // the late antes. Two tiers, each a share of games and the amount added (a game gets at most one)
    // scenarios (train.py --scenario-*): in scn_p of the games the game starts from a sampled build and ends where the scenario does
    double scn_p = 0.0;              // share of games that are scenarios (the rest are played from the start, as ever)
    double scn_w[3] = {1, 0, 0};     // the mix of kinds: 0 one blind, 1 three to six blinds with shops (rerolls capped), 2 a full run from the sampled ante
    int scn_min_ante = 1, scn_max_ante = 7, scn_reroll_cap = 3;
    double scn_couple = 0.5;         // share of scenarios whose deck theme and jokers are made to fit each other
    std::shared_ptr<Archive> arch;   // the state archive (shared by the environments of a VecEnv), null: none
    double arch_p = 0.0;             // share of games that start from an archived state
    double arch_store = 0.0;         // chance that the start of a blind (of a game that is not a synthetic scenario) is offered to the archive
    int arch_lo = 9, arch_hi = 20, arch_min = 8;       // the blinds (0..23) it starts from and stores; a blind needs arch_min snapshots to be used
    uint64_t arch_rng = 0x2545f4914f6cdd1dull;
    bool arch_winners = false;       // snapshots are kept only from games that went on to win (they wait in arch_pending until the game ends)
    std::vector<ArchSnap> arch_pending;
    int arch_start_blind = -1;       // the blind this game started from (an archive game), else -1
    bool autoplay = false;           // see step_auto
    int autoplay_nodes = 400;        // the solver's node cap
    bool autoplay_force = false;     // autoplay: the ordinary select too hands the round to the solver (every round is solver-played)
    double blind_mult[48] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};     // --immortal-fade: each blind's reward times this
    bool no_stickers = false, small_blind_pay = false;
    bool auto_target = false;         // a consumable that needs target cards is applied at once with Game::auto_targets (no targeting step): used in a round or picked from a pack
    bool shop_only = false;           // every blind before the final one is won without being played (paid as if won with one hand left; the counters of jokers that grow by playing advance one round); the final boss is played by the exact round solver
    int shop_only_nodes = 1000;       // the solver's node cap for that final boss
    double partial_w = 0.5;           // the part credit of a lost blind: partial_w x its weight x the share of the target reached (0.5: the original)
    int so_rounds = 0;                // shop-only: rounds won without play so far this game (a deterministic cursor for the simulated card scorings)   // rule changes: see Game::no_stickers / small_blind_pay
    int menu_slots = 0;              // see Game::menu_slots
    double catalogue_p = 0.0;        // share of games with the catalogue shop (Game::catalogue)
    bool immortal_pay = false;       // see Game::immortal_pay
    bool immortal = false;           // see Game::immortal (train.py --immortal): the reward is the blinds actually beaten, each once, and the win
    uint64_t last_bits = 0;          // the blinds beaten in the last game that ended in this environment (survives the auto-reset)
    double scn_power_lo = 1.0, scn_power_hi = 1.0;   // a scenario's build power is log-uniform in [lo, hi]: it multiplies the hand level-ups and the rounds its counters have grown, and above ~1.5
                                                      // fills the jokers (4-5) mostly with ones that score (+mult / xMult / growing), with more editions: builds near the pass / fail line of late antes
    uint64_t scn_build_seed = 0;     // != 0: the build (jokers, deck, levels, money) of a scenario is drawn from this seed, not from the game's: the same build meets new draws
    int scn_force_blind = -1;        // 0 / 1 / 2: a one-blind scenario faces this blind (small, big, boss); -1: any
    double scn_level_conc = 0.0;     // share of scenarios whose hand levels pile up on one or two hand types (as a played-out run's do) instead of spreading
    int scn_kind = -1;               // this game's kind (-1: a real game from the start)
    double last_score = 0.0;         // the chips the last step's play scored (0: the step was no play); survives the auto-reset of a finished game
    int scn_limit = 0;               // the furthest_blind at which the scenario is cut off (a truncation, not a loss), 0: never
    int reroll_cap = MAX_REROLLS_PER_SHOP;
    int reroll_cap_base = MAX_REROLLS_PER_SHOP;   // the cap a normal game has (a few-blinds scenario sets its own)
    int flash_cap_cfg = 0;                        // Game::flash_cap of every game
    int hand_level_cap_cfg = 0;                   // Game::hand_level_cap of every game
    int scaled_start = 0;            // the sandbox: up to this many growing jokers held from the start (1..n, random), their counters grown scaled_rounds rounds
    bool scaled_shop = false;        // ... and a growing joker made for a shop or a pack arrives with its counter grown the same
    int scaled_rounds = 30;
    int start_ante = 1;              // > 1: every game begins in the shop of this ante (blinds before it count as beaten): a sandbox for the late game
    double start_money_p = 0.0, start_money2_p = 0.0;
    int start_money = 0, start_money2 = 0;
    double weights[48];
    std::vector<long long> blind_bases_cfg;   // (see Game::blind_bases) given to every new game
    double win_bonus, shape_chips, shape_gamma;
    double blind_scale = 1.0;        // the reward for each blind beaten (and the part-credit for a lost one) times this
    bool verdant_rel = false;        // the Verdant Leaf bonus is times the weight of the blind being faced
    double ante8_bonus = 0.0;        // paid once, when the ante-7 boss is beaten and ante 8 begins
    int sparse_from_blind = 24;      // blinds from this index on (3 per ante) pay into the sparse stream (24: none; the win and ante-8 bonuses always do)
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
        set_weights(ante_weight, 1.0);
        own = new Obs();
        p = own->ptr();
    }
    Env(const Env&) = delete;
    Env& operator=(const Env&) = delete;
    // the game being played (and its counters) copied from another environment: a search plays candidate actions on copies
    // scripted rounds (judge data, the shop-only game): the select action plays the whole round with scripted_action(), which sees only the hand it holds (no foresight of the
    // draws: unlike --autoplay's exact solver). Rule: play the best predicted-score play if it clears the blind, if it is the last hand, if no discard is left, or if it is on pace
    // (best x hands left >= sp_pace x chips needed); else discard the largest set of cards outside it (the lowest-valued ones among them).
    bool auto_round = false;
    double sp_pace = 1.5;
    int sp_mode = 0;                         // 0: score as much as possible; 1: scaler-friendly: no discards for Green Joker, no face cards for Ride the Bus, not the most-played hand for Obelisk (it for
                                             // Supernova), 4-card plays for Square, straights for Runner: each only up to a cost in score, and always a play that clears the blind if there is one
    bool lite = false;                       // inside a scripted round: no mask check, and only the candidate plays are made after a step (the observation is made when the round is over)
    double rr_ratio = 0.0, rr_money = 0.0;   // the last scripted round: best share of the target reached, money gained
    int rr_hands = 0, rr_disc = 0, rr_clear = 0;
    double rr_c0[MAX_JOKERS], rr_c1[MAX_JOKERS];   // each joker's counter before and after it
    bool unchecked = false;                  // step() trusts the action (a search copies a state and plays an action known to be legal)
    void copy_config_from(const Env& o) {
        marginal = o.marginal; exact = o.exact;
        for (int i = 0; i < 48; i++) weights[i] = o.weights[i];
        blind_bases_cfg = o.blind_bases_cfg;
        win_bonus = o.win_bonus; blind_scale = o.blind_scale; ante8_bonus = o.ante8_bonus; sparse_from_blind = o.sparse_from_blind;
        death_pen = o.death_pen; death_cap = o.death_cap; verdant_bonus = o.verdant_bonus;
        cap_set = o.cap_set; cap_maxes = o.cap_maxes; cap_max = o.cap_max;
    }
    // the unseen future redrawn: every named stream restarts from the seed of `salt` (what is on the table stays; shops, packs, bosses, tags and
    // shuffles still to come, and every random roll, are new draws)
    // the honest round player's copies: the cards not yet seen (the draw pile) in a new order and the rolls still to come redrawn, so a solve on the copy
    // cannot use the real future
    void hide_future(uint64_t salt) {
        if (g.exact) reseed_future(salt); else g.rng.seed_u64(salt);
        PyRandom r(salt ^ 0x51afd7ed558ccd25ull);
        for (int i = (int)g.deck.size() - 1; i > 0; i--) std::swap(g.deck[i], g.deck[r.choice_index((size_t)(i + 1))]);
        encode();
    }
    void reseed_future(uint64_t salt) {
        g.sg.init(seed_string(salt));
        marg_valid = false;
        priv_shop_valid = false;
        fut_ante = -1;
    }
    void copy_state_from(const Env& o, bool enc = true) {
        g = o.g;
        steps = o.steps; rerolls = o.rerolls; swaps = o.swaps; phi = o.phi;
        restricted = o.restricted; level_capped_game = o.level_capped_game;
        school_student = o.school_student; school_shop_seen = o.school_shop_seen;
        marg_valid = false;
        priv_shop_valid = false;
        fut_ante = -1;
        if (enc) {
            encode();
        } else {                                  // the candidates of the copied state (what an action index means)
            std::memcpy(plays, o.plays, sizeof plays); std::memcpy(discs, o.discs, sizeof discs);
            std::memcpy(play_len, o.play_len, sizeof play_len); std::memcpy(disc_len, o.disc_len, sizeof disc_len);
            std::memcpy(pred_sc, o.pred_sc, sizeof pred_sc); std::memcpy(pred_hand, o.pred_hand, sizeof pred_hand);
        }
    }
    ~Env() { delete own; }
    void bind(const ObsPtr& ptr) { p = ptr; }
    Obs& obs_ref() { return *own; }

    double potential() const {
        if (g.done()) return 0.0;
        double ph = 0.0;
        if (shape_chips != 0.0 && g.state == ST_SELECTING_HAND && g.target > 0) {
            int b = 3 * (g.ante - 1) + g.blind_idx;
            if (b + 1 > g.furthest_blind) ph += shape_chips * weights[std::min(b, 47)] * std::min(1.0, g.chips / g.target);
        }
        return ph;
    }
    // the reward for each of the 24 blinds: 1 at ante 1 up to ante_weight at ante 8, along (ante / 7) ** power
    // (power 1: linear, the original; above 1: convex), scaled so that they add up to 24
    void set_weights(double ante_weight, double power) {           // (a game of win_ante antes: 3 x win_ante blinds summing to that many; the ante weight reached at the last ante)
        const int NB = std::min(48, 3 * std::max(1, win_ante)), NA = std::max(2, NB / 3);
        double raw[48], sum = 0.0;
        for (int b = 0; b < NB; b++) {
            if (power == 1.0) raw[b] = 1 + (ante_weight - 1) * (b / 3) / (double)(NA - 1);
            else raw[b] = 1 + (ante_weight - 1) * std::pow((b / 3) / (double)(NA - 1), power);
            sum += raw[b];
        }
        double scale = NB / sum;
        for (int b = 0; b < 48; b++) weights[b] = b < NB ? raw[b] * scale : 0.0;
    }
    // blind rewards that rise by the same factor each ante (1, ratio, ratio^2 ...), summing to 24
    void set_weights_geom(double ratio) {
        const int NB = std::min(48, 3 * std::max(1, win_ante));
        double raw[48], sum = 0.0;
        for (int b = 0; b < NB; b++) { raw[b] = std::pow(ratio, b / 3); sum += raw[b]; }
        for (int b = 0; b < 48; b++) weights[b] = b < NB ? raw[b] * NB / sum : 0.0;
    }
    void reset(uint64_t seed) {
        so_rounds = 0;
        g = Game(seed, deck, stake, win_ante, nullptr, exact ? seed_string(seed).c_str() : nullptr);
        marg_valid = false;
        priv_shop_valid = false;
        fut_ante = -1;
        school_student = -1;
        school_shop_seen = -1;
        restricted = false;
        level_capped_game = false;
        if (level_cap_p > 0.0) {
            PyRandom r(seed ^ 0x7f4a7c159e3779b9ull);
            level_capped_game = r.random() < level_cap_p;
        }
        if (level_cap_mode == 1 && level_capped_game) g.planet_level_cap = level_cap_n;
        g.pack_restock = pack_restock;
        g.immortal = immortal;
        g.menu_slots = menu_slots;
        g.blind_bases = blind_bases_cfg;
        if (catalogue_p > 0.0) { PyRandom rc(seed ^ 0x2545f4914f6cdd1dull); g.catalogue = rc.random() < catalogue_p; }
        g.no_stickers = no_stickers;
        g.small_blind_pay = small_blind_pay;
        g.immortal_pay = immortal_pay;
        for (int k = 0; k < 5; k++) g.pack_w_mult[k] = pack_bias[k];
        if (ban_p > 0.0 && !banned.empty()) {
            PyRandom r(seed ^ 0x5851f42d4c957f2dull);
            if (r.random() < ban_p) {
                restricted = true;
                g.has_pool = true;
                for (int k = 0; k < J_COUNT; k++) g.pool[k] = true;
                for (int16_t k : banned) if (k >= 0 && k < J_COUNT) g.pool[k] = false;
            }
        }
        if (tilt_p > 0.0) {
            PyRandom r(seed ^ 0x14057b7ef767814full);
            if (r.random() < tilt_p) g.hand_levels[r.choice_index((size_t)9)] += tilt_levels;
        }
        if (start_xmult > 0.0) {
            PyRandom r(seed ^ 0x9e3779b97f4a7c15ull);       // its own stream: the choice does not shift the game's draws
            if (r.random() < start_xmult) {
                Joker j;
                j.key = START_XMULT[r.choice_index((size_t)N_START_XMULT)];
                j.base_cost = j.d().cost;
                g.add_joker(j);
            }
        }
        if (start_xscale > 0.0) {
            PyRandom r(seed ^ 0x3c6ef372fe94f82bull);
            if (r.random() < start_xscale) {
                Joker j;
                j.key = START_XSCALE[r.choice_index((size_t)N_START_XSCALE)];
                j.base_cost = j.d().cost;
                j.st.set(start_xscale_val > 1.0 ? start_xscale_val : 1.0);
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
                j.eternal = start_any_eternal;
                g.add_joker(j);
                if (start_any2 > 0.0 && r.random() < start_any2) {
                    double x2 = r.random();
                    int rarity2 = x2 < 0.3 ? 1 : (x2 < 0.7 ? 2 : 3);
                    int n2 = 0;
                    for (int k = 0; k < J_COUNT; k++) if (JOKER_DEFS[k].rarity == rarity2 && k != j.key) opts[n2++] = (int16_t)k;
                    if (n2 > 0) {
                        Joker j2;
                        j2.key = opts[r.choice_index((size_t)n2)];
                        j2.base_cost = j2.d().cost;
                        j2.eternal = start_any_eternal;
                        g.add_joker(j2);
                    }
                }
            }
        }
        if (school_p > 0.0 && !school_ids.empty()) {
            PyRandom r(seed ^ 0x6a09e667f3bcc909ull);
            if (r.random() < school_p) {
                school_student = school_ids[r.choice_index(school_ids.size())];
                Joker j;
                j.key = (int16_t)school_student;
                j.base_cost = j.d().cost;
                j.eternal = true;
                g.add_joker(j);
                int kit[2];
                int n = school_kit(school_student, kit);
                for (int i = 0; i < n && (int)g.consumables.size() < g.consumable_slots; i++)
                    g.consumables.push_back(Game::make_cons(kit[i]));
            }
        }
        if (start_grow > 0.0) {
            PyRandom r(seed ^ 0x27d4eb2f165667c5ull);
            if (r.random() < start_grow) {
                Joker j;
                j.key = START_GROW[r.choice_index((size_t)N_START_GROW)];
                j.base_cost = j.d().cost;
                j.eternal = true;
                g.add_joker(j);
            }
        }
        if (start_money_p > 0.0 || start_money2_p > 0.0) {
            PyRandom r(seed ^ 0x165667b19e3779f9ull);
            double u = r.random();
            if (u < start_money2_p) g.money += start_money2;
            else if (u < start_money2_p + start_money_p) g.money += start_money;
        }
        g.scaled_shop = scaled_shop;
        g.scaled_rounds = scaled_rounds;
        if (scaled_start > 0) {
            PyRandom r(seed ^ 0x8a5cd789635d2dffull);
            int16_t opts[J_COUNT];
            int n = 0;
            for (const ScnGrow& sg : SCN_GROW)
                if (sg.per_round > 0 && !(g.has_pool && !g.pool[sg.key])) opts[n++] = sg.key;
            int k = 1 + (int)r.choice_index((size_t)scaled_start);
            for (int i = 0; i < k && n > 0; i++) {
                int q = (int)r.choice_index((size_t)n);
                Joker j;
                j.key = opts[q];
                j.base_cost = j.d().cost;
                double v;
                if (scaled_counter(j.key, scaled_rounds, v)) j.st.set(v);
                g.add_joker(j);
                opts[q] = opts[--n];
            }
        }
        if (start_ante > 1) {                           // the late-game sandbox: the shop of the ante, the blinds before it behind us
            g.ante = start_ante;
            g.blind_idx = 0;
            g.new_ante();
            g.furthest_blind = 3 * (start_ante - 1);
            g.open_shop();
        }
        scn_kind = -1; scn_limit = 0; reroll_cap = reroll_cap_base;
        g.hand_level_cap = hand_level_cap_cfg;
        g.flash_cap = flash_cap_cfg;
        if (scn_p > 0.0) {
            PyRandom r(seed ^ 0xd6e8feb86659fd93ull);
            if (r.random() < scn_p) apply_scenario(seed);
        }
        arch_pending.clear(); arch_start_blind = -1;
        if (arch && arch_p > 0.0 && scn_kind < 0) {
            PyRandom r(seed ^ 0xa24baed4963ee407ull);
            if (r.random() < arch_p) restore_from_archive(r, seed);
        }
        steps = 0; rerolls = 0; swaps = 0;
        encode();
        phi = potential();
    }
    // a game started from an archived state: the blind to be selected, the build and the deck as they were; the future (draws, shops) is drawn anew
    bool restore_from_archive(PyRandom& r, uint64_t seed) {
        std::lock_guard<std::mutex> lk(arch->mu);
        int cand[24], nc = 0;
        for (int b = arch_lo; b <= arch_hi && b < 24; b++)
            if ((int)arch->slot[stake][b].size() >= arch_min) cand[nc++] = b;
        if (!nc) return false;
        int b = cand[r.choice_index((size_t)nc)];
        const std::vector<ArchSnap>& v = arch->slot[stake][b];
        const ArchSnap& s = v[r.choice_index(v.size())];
        arch->starts[b]++;
        arch_start_blind = b;
        g = s.g;
        g.rng = PyRandom(seed ^ 0x6c8e9cf570932bd5ull);
        restricted = s.restricted; level_capped_game = s.level_capped;
        scn_kind = 3; scn_limit = 0;
        marg_valid = false; priv_shop_valid = false; fut_ante = -1;
        return true;
    }
    void reservoir_insert(int b, const ArchSnap& snap) {
        std::lock_guard<std::mutex> lk(arch->mu);
        std::vector<ArchSnap>& v = arch->slot[stake][b];
        uint64_t n = ++arch->seen[stake][b];
        if ((int)v.size() < arch->cap) { v.push_back(snap); return; }
        uint64_t j = arch->next() % n;                              // (reservoir sampling: every state seen has the same chance of being kept)
        if ((int)j < arch->cap) v[j] = snap;
    }
    void archive_commit() {                                          // the game was won: the blind starts it went through are kept
        for (const ArchSnap& s : arch_pending) {
            int b = 3 * (s.g.ante - 1) + s.g.blind_idx;
            if (b < 0 || b > 23) continue;
            arch_rng ^= arch_rng << 13; arch_rng ^= arch_rng >> 7; arch_rng ^= arch_rng << 17;
            if ((double)(arch_rng >> 11) * (1.0 / 9007199254740992.0) < arch_store) reservoir_insert(b, s);
        }
    }
    void archive_offer() {
        int b = 3 * (g.ante - 1) + g.blind_idx;
        if (b < 0 || b > 23) return;                        // (every blind is kept; the range arch_lo..arch_hi is only where games start from)
        if (arch_winners) { if (b >= arch_lo && arch_pending.size() < 32) arch_pending.push_back(ArchSnap{g, restricted, level_capped_game}); return; }   // (winners only, from arch_lo: the late antes)
        arch_rng ^= arch_rng << 13; arch_rng ^= arch_rng >> 7; arch_rng ^= arch_rng << 17;
        if ((double)(arch_rng >> 11) * (1.0 / 9007199254740992.0) >= arch_store) return;
        std::lock_guard<std::mutex> lk(arch->mu);
        std::vector<ArchSnap>& v = arch->slot[stake][b];
        uint64_t n = ++arch->seen[stake][b];
        if ((int)v.size() < arch->cap) { v.push_back(ArchSnap{g, restricted, level_capped_game}); return; }
        uint64_t j = arch->next() % n;                              // (reservoir sampling: every state seen has the same chance of being kept)
        if ((int)j < arch->cap) v[j] = ArchSnap{g, restricted, level_capped_game};
    }
    // A scenario: a build (2-5 jokers with counters as if owned for a number of rounds, a deck with a theme, hand levels, money) at a sampled
    // ante, and a length (one blind / a few blinds with shops / the rest of the run). It is the same game, the same observations and rewards:
    // only the start differs, and a cut-off scenario is a truncation (the value of where it stood is bootstrapped).
    void apply_scenario(uint64_t seed) {
        PyRandom r((scn_build_seed ? scn_build_seed : seed) ^ 0x2545f4914f6cdd1dull);
        double wsum = scn_w[0] + scn_w[1] + scn_w[2];
        double u = r.random() * (wsum > 0 ? wsum : 1.0);
        scn_kind = u < scn_w[0] ? 0 : (u < scn_w[0] + scn_w[1] ? 1 : 2);
        int span = std::max(1, scn_max_ante - scn_min_ante + 1);
        int ante = scn_min_ante + (int)r.choice_index((size_t)span);
        g.jokers.clear();
        g.consumables.clear();
        g.joker_slots = 5 + (deck == D_BLACK ? 1 : 0) - (deck == D_PAINTED ? 1 : 0);
        // the jokers
        double power = scn_power_hi > scn_power_lo ? scn_power_lo * std::exp(r.random() * std::log(scn_power_hi / scn_power_lo)) : scn_power_lo;
        int nj = 2 + (int)r.choice_index(4);
        if (power > 1.5) nj = std::min(5, nj + 1 + (power > 3.0 ? 1 : 0));
        double want_scoring = power > 1.5 ? std::min(0.9, 0.3 * power) : 0.0;
        bool couple = r.random() < scn_couple;
        bool deck_from_jokers = couple && r.random() < 0.5;       // coupled: the deck follows the jokers, or the jokers follow the deck
        std::vector<int> picked;
        double rare_p = std::min(0.45, 0.05 + 0.05 * ante), unc_p = 0.30;
        auto draw_joker = [&](int need_theme, int need_param) -> int {
            bool scoring_only = r.random() < want_scoring;
            for (int tries = 0; tries < 200; tries++) {
                double x = r.random();
                int rarity = x < rare_p ? 3 : (x < rare_p + unc_p ? 2 : 1);
                int16_t opts[J_COUNT];
                int n = 0;
                for (int k = 0; k < J_COUNT; k++) {
                    if (JOKER_DEFS[k].rarity != rarity || (g.has_pool && !g.pool[k])) continue;
                    if (k == J_CEREMONIAL || k == J_MADNESS) continue;
                    if (std::find(picked.begin(), picked.end(), k) != picked.end()) continue;
                    if (scoring_only && JOKER_SWAP_ROLE[k] < 1) {
                        bool grows = false;
                        for (const ScnGrow& sg : SCN_GROW) if (sg.key == k && sg.per_round > 0) grows = true;
                        if (!grows) continue;
                    }
                    if (need_theme != TH_NONE) {
                        bool ok = false;
                        for (int q = 0; q < N_SCN_NEED; q++)
                            if (SCN_NEED[q].key == k && SCN_NEED[q].theme == need_theme && SCN_NEED[q].param == need_param) ok = true;
                        if (!ok) continue;
                    }
                    opts[n++] = (int16_t)k;
                }
                if (n) return opts[r.choice_index((size_t)n)];
            }
            return -1;
        };
        // the deck theme: from a need of one of the jokers, or chosen (and then maybe a joker that needs it)
        int theme = TH_NONE, tparam = 0;
        std::vector<int> js;
        for (int i = 0; i < nj; i++) { int k = draw_joker(TH_NONE, 0); if (k >= 0) { picked.push_back(k); js.push_back(k); } }
        if (couple && deck_from_jokers) {
            std::vector<int> needs;
            for (int q = 0; q < N_SCN_NEED; q++)
                for (int k : js) if (SCN_NEED[q].key == k) needs.push_back(q);
            if (!needs.empty()) { int q = needs[r.choice_index(needs.size())]; theme = SCN_NEED[q].theme; tparam = SCN_NEED[q].param; }
        }
        if (theme == TH_NONE) {
            theme = (int)r.choice_index((size_t)TH_COUNT);
            if (theme == TH_SUIT) tparam = (int)r.choice_index(4);
            else if (theme == TH_RANKS) tparam = (int)r.choice_index(8);
            else if (theme == TH_ENH) { static const int E[6] = {E_BONUS, E_MULT, E_GLASS, E_STEEL, E_LUCKY, E_GOLD}; tparam = E[r.choice_index(6)]; }
            if (couple && !deck_from_jokers && theme != TH_NONE && theme != TH_EDITION && !js.empty()) {
                int k = draw_joker(theme, tparam);                     // a joker that needs this deck replaces one of the drawn ones
                if (k >= 0) { picked.push_back(k); js[r.choice_index(js.size())] = k; }
            }
        }
        for (int k : js) {
            Joker j;
            j.key = (int16_t)k;
            j.base_cost = j.d().cost;
            double e = r.random();
            double ed_p = power > 1.5 ? 0.35 : 0.15;
            if (e < 0.03) j.edition = ED_NEG;
            else if (e < ed_p) j.edition = (uint8_t)(ED_FOIL + r.choice_index(3));
            int rounds = (int)std::lround(std::exp(r.random() * std::log(3.0 * ante + 4.0)) * power);      // log-uniform, 1 .. about the rounds played so far (times the build power)
            if (rounds < 1) rounds = 1;
            for (const ScnGrow& sg : SCN_GROW) {
                if (sg.key != k) continue;
                double v = sg.init + sg.per_round * rounds * (0.5 + r.random());
                j.st.set(std::min(sg.hi, std::max(sg.lo, v)));
            }
            g.add_joker(j);
        }
        // the deck: the 52 cards, edited to the theme with a strength in [0, 1]
        double strength = r.random();
        int n_edit = (int)std::lround(strength * (6 + 4 * ante));
        std::vector<int>& fd = g.full_deck;
        auto rand_card = [&]() { return fd[r.choice_index(fd.size())]; };
        static const int RANKSETS[8][5] = {{2, 4, 6, 8, 10}, {3, 5, 7, 9, 14}, {14, 14, 14, 14, 14}, {2, 3, 4, 5, 2},
                                           {14, 2, 3, 5, 8}, {10, 4, 10, 4, 10}, {2, 2, 2, 2, 2}, {8, 8, 8, 8, 8}};
        for (int i = 0; i < n_edit && !fd.empty(); i++) {
            int cid = rand_card();
            Card& c = g.cards[cid];
            switch (theme) {
                case TH_SUIT: c.suit = (int8_t)tparam; break;
                case TH_FACE: c.rank = (int8_t)(11 + r.choice_index(3)); break;
                case TH_RANKS: c.rank = (int8_t)RANKSETS[tparam][r.choice_index(5)]; break;
                case TH_ENH: c.enh = (uint8_t)tparam; break;
                case TH_PAIRS: { int id = g.copy_card(cid); fd.push_back(id); break; }
                case TH_THIN: if (fd.size() > 20) fd.erase(std::find(fd.begin(), fd.end(), cid)); break;
                case TH_EDITION:
                    if (r.random() < 0.7) c.ed = (uint8_t)(ED_FOIL + r.choice_index(3));
                    else c.seal = (uint8_t)(S_RED + r.choice_index(4));
                    break;
                default: break;
            }
        }
        for (int i = 0, n = (int)r.choice_index((size_t)ante + 1); i < n && !fd.empty(); i++) {     // some random legal edits whatever the theme
            Card& c = g.cards[rand_card()];
            double t = r.random();
            if (t < 0.25) c.suit = (int8_t)r.choice_index(4);
            else if (t < 0.5) c.rank = (int8_t)(2 + r.choice_index(13));
            else if (t < 0.75) c.enh = (uint8_t)(E_BONUS + r.choice_index(5));
            else c.ed = (uint8_t)(ED_FOIL + r.choice_index(3));
        }
        // hand levels and money
        int lv = (int)std::floor(r.random() * r.random() * (2.0 * ante + 1.0) * power);
        if (r.random() < scn_level_conc) {                       // a played-out run piles its levels on the hands it plays: ~60% on one, ~25% on a second
            static const int HW[9] = {1, 6, 3, 2, 2, 3, 1, 1, 1};   // High Card .. Straight Flush: Pair, Two Pair, Flush, Straight, Three are the usual homes
            double tot = 0; for (int h = 0; h < 9; h++) tot += HW[h];
            auto pick = [&]() { double x = r.random() * tot; for (int h = 0; h < 9; h++) { x -= HW[h]; if (x <= 0) return h; } return 1; };
            int p1 = pick(), p2 = pick();
            for (int i = 0; i < lv; i++) { double x = r.random(); g.hand_levels[x < 0.6 ? p1 : (x < 0.85 ? p2 : (int)r.choice_index(9))] += 1; }
        } else {
            for (int i = 0; i < lv; i++) g.hand_levels[r.choice_index(9)] += 1;
        }
        g.money = (long long)(std::exp(r.random() * std::log(301.0)) - 1.0);
        // where and how long
        g.ante = ante;
        g.blind_idx = 0;
        g.state = ST_BLIND_SELECT;
        if (scn_kind == 0) {
            g.blind_idx = (int)r.choice_index(3);
            if (scn_force_blind >= 0) g.blind_idx = scn_force_blind;
            g.new_ante();
            g.furthest_blind = 3 * (ante - 1) + g.blind_idx;
            scn_limit = g.furthest_blind + 1;
        } else {
            g.new_ante();
            g.furthest_blind = 3 * (ante - 1);
            g.open_shop();
            if (scn_kind == 1) {
                scn_limit = g.furthest_blind + 3 + (int)r.choice_index(4);
                reroll_cap = scn_reroll_cap;
            }
        }
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
    // The cards of a play score in the order they are played; that changes the score only with these jokers or
    // a Glass / Polychrome card in the hand (env.py order_matters)
    bool order_matters() const {
        for (const Joker& j : g.jokers)
            if (!j.debuffed && (j.key == J_PHOTOGRAPH || j.key == J_HANGING_CHAD || j.key == J_BLOODSTONE ||
                                j.key == J_TRIBOULET || j.key == J_IDOL || j.key == J_ANCIENT)) return true;
        for (int id : g.hand) {
            const Card& c = g.cards[id];
            if (c.enh == E_GLASS || c.ed == ED_POLY) return true;
        }
        return false;
    }
    // The best of a few orders of the play s (k cards, scoring `base` as given): reversed, each card moved to
    // the front, each card moved to the back (env.py best_orders). s is rewritten; -> its score.
    double best_order(int8_t* s, int k, double base, const Plan& plan, Game::View& view) {
        if (k < 2) return base;
        int hand = 0, p[5], rev[5], rot[5];
        for (int q = 0; q < k; q++) rev[q] = s[k - 1 - q];
        for (int q = 0; q < k - 1; q++) rot[q] = s[q + 1];
        rot[k - 1] = s[0];
        double r1 = g.predict(rev, k, plan, view, hand), r2 = g.predict(rot, k, plan, view, hand);
        if (r1 == base && r2 == base) return base;
        double best = base;
        int bp[5];
        for (int q = 0; q < k; q++) bp[q] = s[q];
        auto consider = [&](const int* o, double sc) {
            if (sc > best) { best = sc; for (int q = 0; q < k; q++) bp[q] = o[q]; }
        };
        consider(rev, r1);
        for (int j = 1; j < k; j++) {                        // card j to the front
            int w = 0;
            p[w++] = s[j];
            for (int q = 0; q < k; q++) if (q != j) p[w++] = s[q];
            consider(p, g.predict(p, k, plan, view, hand));
        }
        for (int j = 0; j < k - 1; j++) {                    // card j to the back
            int w = 0;
            for (int q = 0; q < k; q++) if (q != j) p[w++] = s[q];
            p[w++] = s[j];
            consider(p, j == 0 ? r2 : g.predict(p, k, plan, view, hand));
        }
        for (int q = 0; q < k; q++) s[q] = (int8_t)bp[q];
        return best;
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
        const bool ordered = order_matters();
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
                if (ordered) pred_sc[i] = best_order(plays[i], len, pred_sc[i], plan, view);
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
        struct Scored { double neg; int len; int8_t s[5]; double sc; int hand; int8_t o[5]; };
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
            for (int q = 0; q < 5; q++) x.o[q] = x.s[q];
            if (ordered) x.sc = best_order(x.o, k, x.sc, plan, view);
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
            for (int q = 0; q < 5; q++) plays[i][q] = scored[i].o[q];
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
        f[k++] = (float)(rerolls / (double)reroll_cap);
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
        for (int st = 0; st < N_STAKES; st++) f[k++] = g.stake == st ? 1.0f : 0.0f;
        for (int q = 0; q < F_GLOBAL - F_GLOBAL_BASE; q++) f[k++] = 0.0f;      // marginal_fill
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
            m[A_SKIP] = g.blind_idx < 2 && scn_kind != 0;       // (a one-blind scenario ends where its blind is beaten: no skipping it)
            m[A_REROLL_BOSS] = g.can_reroll_boss();
            m[A_LEAVE] = autoplay;
        } else if (st == ST_SHOP) {
            for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) m[A_BUY + i] = g.can_buy(g.shop[i]);
            for (int i = 0; i < (int)g.shop_packs.size() && i < 2; i++) m[A_BUY_PACK + i] = g.can_afford(g.shop_packs[i].cost);
            for (int i = 0; i < (int)g.shop_vouchers.size() && i < MAX_VOUCHERS; i++)
                m[A_VOUCHER + i] = g.can_afford(g.shop_vouchers[i].cost);
            m[A_REROLL] = rerolls < reroll_cap && (g.free_rerolls > 0 || g.can_afford(g.reroll_cost) || (g.catalogue && !g.exact));
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
            if (cap_max >= 0) {                                   // the joker caps: no buy or swap that would hold more than a group's maximum
                for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) {
                    const ShopItem& it = g.shop[i];
                    int grp = it.kind == SK_JOKER ? cap_group(it.joker.key) : 0;
                    if (!grp) continue;
                    int held = cap_held(grp), mx = cap_maxes[grp - 1];
                    if (held >= mx) m[A_BUY + i] = false;
                    for (int j = 0; j < (int)g.jokers.size() && j < MAX_JOKERS; j++)
                        if (held - (int)(cap_group(g.jokers[j].key) == grp) >= mx) m[A_REPLACE + i * MAX_JOKERS + j] = false;
                }
            }
        } else if (st == ST_PACK) {
            for (int i = 0; i < (int)g.pack_cards.size() && i < MAX_PACK; i++) {
                int grp = g.pack_cards[i].kind == PI_JOKER ? cap_group(g.pack_cards[i].joker.key) : 0;
                m[A_PICK + i] = g.pack_pick_ok(i) && !(grp && cap_held(grp) >= cap_maxes[grp - 1]);
            }
            m[A_PSKIP] = true;
        }
        if (level_capped_now()) {
            if (st == ST_SHOP) {
                for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) if (g.shop[i].kind == SK_PLANET) m[A_BUY + i] = false;
                for (int i = 0; i < (int)g.shop_packs.size() && i < 2; i++)
                    if (PACK_DEFS[g.shop_packs[i].key].kind == PK_CELESTIAL) m[A_BUY_PACK + i] = false;
            } else if (st == ST_PACK) {
                for (int i = 0; i < (int)g.pack_cards.size() && i < MAX_PACK; i++)
                    if (g.pack_cards[i].kind == PI_CONS && g.pack_cards[i].cons.kind() == CK_PLANET) m[A_PICK + i] = false;
            }
        }
        if (st == ST_SHOP || st == ST_SELECTING_HAND || st == ST_PACK) {      // selling: also with a pack open
            int nj = std::min((int)g.jokers.size(), MAX_JOKERS);
            for (int i = 0; i < nj; i++) m[A_SELL_J + i] = !g.jokers[i].eternal;
            for (int i = 0; i < (int)g.consumables.size() && i < MAX_CONSUMABLES; i++) m[A_SELL_C + i] = true;
        }
        if (st == ST_SHOP || st == ST_SELECTING_HAND) {
            int nj = std::min((int)g.jokers.size(), MAX_JOKERS);
            for (int i = 0; i < (int)g.consumables.size() && i < MAX_CONSUMABLES; i++)
                m[A_USE_C + i] = g.consumable_usable(g.consumables[i], st == ST_SELECTING_HAND ? g.hand : none)
                                 && !(level_capped_now() && g.consumables[i].kind() == CK_PLANET);
            if (swaps < MAX_SWAPS_PER_PHASE && g.jokers.size() >= 2 && g.swaps_matter())
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

    // ---------------------------------------------------------------- marginal-value inputs (marginal.py)
    bool marg_blind_only = false;                           // the marginal-value columns only at a blind select (the judge reads them there: no cost at a shop step)
    int marginal = 0;                                        // 0 off, 1 marginal values, 2 also the growth potential
    std::unique_ptr<Game> mscratch;
    static const int MARG_K = 4;
    struct MargCtx {
        int n = 0;                                   // cards per sampled hand
        std::vector<int> hand[MARG_K], rest[MARG_K];  // the sampled hands, and the rest of the deck
        std::vector<int16_t> reps[MARG_K];           // per hand: the best play (subset index) of each hand type
    };
    static double mclip(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
    void marg_setup(Game& s, MargCtx& m) {
        m.n = std::min(std::min(8, g.effective_hand_size()), (int)g.full_deck.size());
        s = g;
        s.state = ST_SELECTING_HAND;
        s.blind_idx = 0;
        s.hands_left = s.round_hands();
        s.discards_left = s.round_discards();
        s.discards_used_round = 0;
        for (int h = 0; h < N_HANDS; h++) s.hand_played_round[h] = 0;
        s.round_hand_types = 0;
        int nd = (int)g.full_deck.size();
        for (int k = 0; k < MARG_K; k++) {
            std::vector<int> idx(nd);
            for (int i = 0; i < nd; i++) idx[i] = i;
            uint64_t x = (0x2545F491ull + 7919ull * (uint64_t)k) & 0x7FFFFFFFull;
            for (int i = 0; i < m.n; i++) {
                x = (x * 1103515245ull + 12345ull) & 0x7FFFFFFFull;
                int j = i + (int)(x % (uint64_t)(nd - i));
                std::swap(idx[i], idx[j]);
            }
            std::vector<char> taken(nd, 0);
            m.hand[k].clear(); m.rest[k].clear();
            for (int i = 0; i < m.n; i++) { taken[idx[i]] = 1; m.hand[k].push_back(g.full_deck[idx[i]]); }
            for (int i = 0; i < nd; i++) if (!taken[i]) m.rest[k].push_back(g.full_deck[i]);
        }
    }
    // the best play of the k-th sampled hand with the scratch game as it is (collect: all plays, and remember the
    // best of each hand type; otherwise only those)
    double marg_best(Game& s, MargCtx& m, int k, bool collect) {
        s.hand = m.hand[k];
        s.deck = m.rest[k];
        Plan plan;
        s.make_plan(plan);
        Game::View view;
        s.hand_view(view);
        const SubsetTable& T = subset_table();
        double top = 0.0;
        if (collect) {
            double tb[N_HANDS];
            int tq[N_HANDS];
            for (int h = 0; h < N_HANDS; h++) { tb[h] = -1.0; tq[h] = -1; }
            for (int q = 0; q < N_SUB; q++) {
                int len = T.len[q];
                if (T.s[q][len - 1] >= m.n) continue;
                int pos[5];
                for (int t = 0; t < len; t++) pos[t] = T.s[q][t];
                int h = 0;
                double sc = s.predict(pos, len, plan, view, h);
                if (tq[h] < 0 || sc > tb[h]) { tb[h] = sc; tq[h] = q; }
            }
            m.reps[k].clear();
            for (int h = 0; h < N_HANDS; h++)
                if (tq[h] >= 0) { m.reps[k].push_back((int16_t)tq[h]); top = std::max(top, tb[h]); }
        } else {
            for (int16_t q : m.reps[k]) {
                int len = T.len[q];
                int pos[5];
                for (int t = 0; t < len; t++) pos[t] = T.s[q][t];
                int h = 0;
                top = std::max(top, s.predict(pos, len, plan, view, h));
            }
        }
        return top;
    }
    double marg_mean(Game& s, MargCtx& m) {
        double total = 0.0;
        for (int k = 0; k < MARG_K; k++) total += marg_best(s, m, k, false);
        return total / MARG_K;
    }
    static void marg_gain(float* row, double base, double mean, double cost) {
        double lr = std::log10((mean + 1.0) / (base + 1.0));
        row[0] = (float)(mclip(lr, -1.0, 3.0) / 3.0);
        row[1] = (float)(mclip(mean / base - 1.0, -1.0, 5.0) / 5.0);
        row[2] = (float)mclip(lr / std::max(1.0, cost), -1.0, 1.0);
    }
    // a hash of everything the scoring of a hand reads (not the money, unless Bull / Bootstraps is held: marginal_fill
    // sets the scratch game's money for the items it evaluates)
    static void mix(uint64_t& h, uint64_t v) { h = (h ^ v) * 0x100000001B3ull; h ^= h >> 29; }
    static uint64_t dbits(double v) { uint64_t u; std::memcpy(&u, &v, sizeof u); return u; }
    // ---- the state key (exact-seed games, at a blind select): a hash of everything that determines the future EXCEPT the money, and the money apart.
    // States with the same key and the same money are the same state (the future is a function of the actions from here); with the same key and more money one dominates the other.
    // The RNG stream positions are in it: two states with the same jokers but different reroll counts will see different shops. Only for exact-seed games (the plain generator's state is not readable).
    static uint64_t kmix(uint64_t h, uint64_t v) {
        v += 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
        return h ^ (v ^ (v >> 31));
    }
    static uint64_t kstr(const std::string& s) { uint64_t h = 1469598103934665603ull; for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; } return h; }
    uint64_t state_key() const {
        uint64_t h = 0x1234567ull;
        auto M = [&](uint64_t v) { h = kmix(h, v); };
        M(g.ante); M(g.blind_idx); M(g.state); M(g.joker_slots); M(g.consumable_slots); M(g.hand_size); M(g.base_hands); M(g.base_discards);
        M(g.vouchers); M(g.planets_used); M(g.planet_uses); M(g.tarots_used); M((uint64_t)(int64_t)g.last_consumable); M(g.rerolls_total); M(g.blinds_skipped);
        M(g.used_bosses); M(g.boss); M(g.tags_offered[0]); M(g.tags_offered[1]); M(g.f_verdant); M(g.f_gros_extinct); M(g.f_seen_soul); M(g.f_seen_black_hole);
        M((uint64_t)(int64_t)g.f_voucher_bought_ante); M(g.f_ecto); M(g.boss_rerolled_ante); M(g.planet_level_gains); M(g.first_shop_buffoon);
        for (int i = 0; i < N_HANDS; i++) { M((uint64_t)g.hand_levels[i]); M((uint64_t)g.hand_played[i]); }
        for (int8_t t : g.pending_tags) M((uint64_t)t);
        M(g.pending_tags.size());
        for (int i = 0; i < 3; i++) M((uint64_t)g.orbital_choice[i]);
        M(g.tgt_idol_rank); M(g.tgt_idol_suit); M(g.tgt_mail); M((uint64_t)(int64_t)g.tgt_ancient); M(g.tgt_castle);
        for (int i = 0; i < B_COUNT; i++) M((uint64_t)g.boss_counts[i]);
        // jokers, in order (and the order of their creation, which decides some random picks)
        std::vector<int> uid_rank(g.jokers.size());
        for (size_t i = 0; i < g.jokers.size(); i++) { int r = 0; for (size_t j = 0; j < g.jokers.size(); j++) if (g.jokers[j].uid < g.jokers[i].uid) r++; uid_rank[i] = r; }
        M(g.jokers.size());
        for (size_t i = 0; i < g.jokers.size(); i++) {
            const Joker& j = g.jokers[i];
            M(j.key); M(j.edition); M(j.eternal); M(j.rental); M(j.debuffed); M(j.hidden); M(j.has_perish); M((uint64_t)j.perishable); M((uint64_t)j.base_cost); M((uint64_t)j.sell_bonus);
            M(j.has_cost); M((uint64_t)j.cost); M((uint64_t)uid_rank[i]);
            M(j.st.has_val); M(LuaRand::d2u(j.st.val)); M((uint64_t)(int64_t)j.st.rank); M((uint64_t)(int64_t)j.st.suit); M((uint64_t)(int64_t)j.st.hand); M((uint64_t)(int64_t)j.st.left);
        }
        M(g.consumables.size());
        for (const Cons& c : g.consumables) { M((uint64_t)(int64_t)c.id); M(c.negative); M((uint64_t)c.sell_bonus); }
        // the deck: its cards in the order they were made
        std::vector<int> ids(g.full_deck.begin(), g.full_deck.end());
        std::sort(ids.begin(), ids.end());
        M(ids.size());
        for (int id : ids) { const Card& c = g.cards[id]; M(c.rank); M(c.suit); M(c.enh); M(c.ed); M(c.seal); M(c.deb); M((uint64_t)c.extra); }
        for (size_t i = 0; i < g.ante_played.size(); i++) M(g.ante_played[i]);
        // the RNG: every named stream's position, the global generator
        if (g.exact) {
            uint64_t acc = 0;
            for (const auto& kv : g.sg.st) acc += kmix(kstr(kv.first), LuaRand::d2u(kv.second));
            M(acc); M(g.sg.st.size());
            for (int i = 0; i < 4; i++) M(g.sg.rng.u[i]);
        }
        return h;
    }
    uint64_t round_key() const {
        uint64_t h = 0x9e3779b97f4a7c15ull;
        auto M = [&](uint64_t v) { h = kmix(h, v); };
        M(g.ante); M(g.blind_idx); M(g.state); M(g.hand_size); M(g.hands_left); M(g.discards_left); M(g.discards_used_round);
        M(LuaRand::d2u(g.chips)); M(LuaRand::d2u(g.target)); M(g.boss); M(g.boss_disabled); M(g.round_hand_types); M((uint64_t)(int64_t)g.mouth_hand);
        M((uint64_t)(int64_t)g.ox_hand); M((uint64_t)(int64_t)g.crimson_disabled); M((uint64_t)(int64_t)g.forced_uid); M(g.hand_size_override); M(g.juggle);
        for (int i = 0; i < N_HANDS; i++) { M((uint64_t)g.hand_levels[i]); M((uint64_t)g.hand_played[i]); M((uint64_t)g.hand_played_round[i]); }
        M(g.jokers.size());
        for (const Joker& j : g.jokers) {
            M(j.key); M(j.edition); M(j.debuffed); M(j.hidden); M((uint64_t)j.perishable);
            M(j.st.has_val); M(LuaRand::d2u(j.st.val)); M((uint64_t)(int64_t)j.st.rank); M((uint64_t)(int64_t)j.st.suit); M((uint64_t)(int64_t)j.st.hand); M((uint64_t)(int64_t)j.st.left);
        }
        auto card = [&](int id) { const Card& c = g.cards[id]; M(c.rank); M(c.suit); M(c.enh); M(c.ed); M(c.seal); M(c.deb); M((uint64_t)c.extra); };
        M(g.hand.size()); for (int id : g.hand) card(id);
        M(g.deck.size()); for (int id : g.deck) card(id);
        M(g.discard_pile.size());
        for (size_t i = 0; i < g.ante_played.size(); i++) M(g.ante_played[i]);
        return h;
    }
    // the names of the scalar members of the game that differ from another's (a debugging aid for the state key)
    std::vector<std::string> diff_names(const Env& o) const {
        std::vector<std::string> out;
        const Game& a = g; const Game& b = o.g;
        if (a.win_ante != b.win_ante) out.push_back("win_ante");
        if (a.has_pool != b.has_pool) out.push_back("has_pool");
        if (a.deck_type != b.deck_type) out.push_back("deck_type");
        if (a.stake != b.stake) out.push_back("stake");
        if (a.ante != b.ante) out.push_back("ante");
        if (a.blind_idx != b.blind_idx) out.push_back("blind_idx");
        if (a.state != b.state) out.push_back("state");
        if (a.money != b.money) out.push_back("money");
        if (a.starting_deck_size != b.starting_deck_size) out.push_back("starting_deck_size");
        if (a.joker_slots != b.joker_slots) out.push_back("joker_slots");
        if (a.consumable_slots != b.consumable_slots) out.push_back("consumable_slots");
        if (a.hand_size != b.hand_size) out.push_back("hand_size");
        if (a.base_hands != b.base_hands) out.push_back("base_hands");
        if (a.base_discards != b.base_discards) out.push_back("base_discards");
        if (a.vouchers != b.vouchers) out.push_back("vouchers");
        if (a.f_verdant != b.f_verdant) out.push_back("f_verdant");
        if (a.f_gros_extinct != b.f_gros_extinct) out.push_back("f_gros_extinct");
        if (a.f_seen_soul != b.f_seen_soul) out.push_back("f_seen_soul");
        if (a.f_seen_black_hole != b.f_seen_black_hole) out.push_back("f_seen_black_hole");
        if (a.f_voucher_bought_ante != b.f_voucher_bought_ante) out.push_back("f_voucher_bought_ante");
        if (a.f_ecto != b.f_ecto) out.push_back("f_ecto");
        if (a.f_trading_destroy != b.f_trading_destroy) out.push_back("f_trading_destroy");
        if (a.planets_used != b.planets_used) out.push_back("planets_used");
        if (a.planet_uses != b.planet_uses) out.push_back("planet_uses");
        if (a.planet_level_cap != b.planet_level_cap) out.push_back("planet_level_cap");
        if (a.planet_level_gains != b.planet_level_gains) out.push_back("planet_level_gains");
        if (a.flash_cap != b.flash_cap) out.push_back("flash_cap");
        if (a.rerolls_total != b.rerolls_total) out.push_back("rerolls_total");
        if (a.hand_level_cap != b.hand_level_cap) out.push_back("hand_level_cap");
        if (a.pack_restock != b.pack_restock) out.push_back("pack_restock");
        if (a.pack_restocks_used != b.pack_restocks_used) out.push_back("pack_restocks_used");
        if (a.tarots_used != b.tarots_used) out.push_back("tarots_used");
        if (a.last_consumable != b.last_consumable) out.push_back("last_consumable");
        if (a.used_bosses != b.used_bosses) out.push_back("used_bosses");
        if (a.exact != b.exact) out.push_back("exact");
        if (a.first_shop_buffoon != b.first_shop_buffoon) out.push_back("first_shop_buffoon");
        if (a.tgt_idol_rank != b.tgt_idol_rank) out.push_back("tgt_idol_rank");
        if (a.tgt_idol_suit != b.tgt_idol_suit) out.push_back("tgt_idol_suit");
        if (a.tgt_mail != b.tgt_mail) out.push_back("tgt_mail");
        if (a.tgt_ancient != b.tgt_ancient) out.push_back("tgt_ancient");
        if (a.tgt_castle != b.tgt_castle) out.push_back("tgt_castle");
        if (a.chips != b.chips) out.push_back("chips");
        if (a.target != b.target) out.push_back("target");
        if (a.hands_left != b.hands_left) out.push_back("hands_left");
        if (a.discards_left != b.discards_left) out.push_back("discards_left");
        if (a.discards_used_round != b.discards_used_round) out.push_back("discards_used_round");
        if (a.crimson_disabled != b.crimson_disabled) out.push_back("crimson_disabled");
        if (a.forced_uid != b.forced_uid) out.push_back("forced_uid");
        if (a.round_hand_types != b.round_hand_types) out.push_back("round_hand_types");
        if (a.mouth_hand != b.mouth_hand) out.push_back("mouth_hand");
        if (a.boss_disabled != b.boss_disabled) out.push_back("boss_disabled");
        if (a.juggle != b.juggle) out.push_back("juggle");
        if (a.hand_size_override != b.hand_size_override) out.push_back("hand_size_override");
        if (a.targeting != b.targeting) out.push_back("targeting");
        if (a.targeting_from_pack != b.targeting_from_pack) out.push_back("targeting_from_pack");
        if (a.first_draw_done != b.first_draw_done) out.push_back("first_draw_done");
        if (a.boss_rerolled_ante != b.boss_rerolled_ante) out.push_back("boss_rerolled_ante");
        if (a.ox_hand != b.ox_hand) out.push_back("ox_hand");
        if (a.has_ante_voucher != b.has_ante_voucher) out.push_back("has_ante_voucher");
        if (a.reroll_cost != b.reroll_cost) out.push_back("reroll_cost");
        if (a.free_rerolls != b.free_rerolls) out.push_back("free_rerolls");
        if (a.shops_seen != b.shops_seen) out.push_back("shops_seen");
        if (a.pack_kind != b.pack_kind) out.push_back("pack_kind");
        if (a.pack_picks != b.pack_picks) out.push_back("pack_picks");
        if (a.pack_return != b.pack_return) out.push_back("pack_return");
        if (a.boss != b.boss) out.push_back("boss");
        if (a.next_joker_uid != b.next_joker_uid) out.push_back("next_joker_uid");
        if (a.scaled_shop != b.scaled_shop) out.push_back("scaled_shop");
        if (a.scaled_rounds != b.scaled_rounds) out.push_back("scaled_rounds");
        if (a.immortal_pay != b.immortal_pay) out.push_back("immortal_pay");
        if (a.menu_slots != b.menu_slots) out.push_back("menu_slots");
        if (a.immortal != b.immortal) out.push_back("immortal");
        if (a.cleared_bits != b.cleared_bits) out.push_back("cleared_bits");
        if (a.last_play_score != b.last_play_score) out.push_back("last_play_score");
        if (std::memcmp(&a.pool, &b.pool, sizeof(a.pool)) != 0) out.push_back("pool[]");
        if (std::memcmp(&a.hand_levels, &b.hand_levels, sizeof(a.hand_levels)) != 0) out.push_back("hand_levels[]");
        if (std::memcmp(&a.hand_played, &b.hand_played, sizeof(a.hand_played)) != 0) out.push_back("hand_played[]");
        if (std::memcmp(&a.hand_played_round, &b.hand_played_round, sizeof(a.hand_played_round)) != 0) out.push_back("hand_played_round[]");
        if (std::memcmp(&a.pack_w_mult, &b.pack_w_mult, sizeof(a.pack_w_mult)) != 0) out.push_back("pack_w_mult[]");
        if (std::memcmp(&a.boss_counts, &b.boss_counts, sizeof(a.boss_counts)) != 0) out.push_back("boss_counts[]");
        if (std::memcmp(&a.orbital_choice, &b.orbital_choice, sizeof(a.orbital_choice)) != 0) out.push_back("orbital_choice[]");
        if (std::memcmp(&a.tags_offered, &b.tags_offered, sizeof(a.tags_offered)) != 0) out.push_back("tags_offered[]");
        if (a.deck != b.deck) out.push_back("deck(order)");
        if (a.hand != b.hand) out.push_back("hand");
        if (a.discard_pile != b.discard_pile) out.push_back("discard_pile");
        if (a.pack_hand != b.pack_hand) out.push_back("pack_hand");
        if (a.cards.size() == b.cards.size()) for (size_t i = 0; i < a.cards.size(); i++) {
            const Card& x = a.cards[i]; const Card& y = b.cards[i];
            if (x.rank != y.rank || x.suit != y.suit || x.enh != y.enh || x.ed != y.ed || x.seal != y.seal || x.deb != y.deb || x.hidden != y.hidden || x.extra != y.extra || x.uid != y.uid) { out.push_back("cards[" + std::to_string(i) + "]"); break; }
        }
        if (a.jokers.size() == b.jokers.size()) for (size_t i = 0; i < a.jokers.size(); i++) {
            const Joker& x = a.jokers[i]; const Joker& y = b.jokers[i];
            if (x.key != y.key || x.edition != y.edition || x.uid != y.uid || x.base_cost != y.base_cost || x.sell_bonus != y.sell_bonus || x.cost != y.cost || x.has_cost != y.has_cost || x.eternal != y.eternal
                || x.rental != y.rental || x.debuffed != y.debuffed || x.hidden != y.hidden || x.perishable != y.perishable || x.has_perish != y.has_perish || x.st.has_val != y.st.has_val || x.st.val != y.st.val
                || x.st.rank != y.st.rank || x.st.suit != y.st.suit || x.st.hand != y.st.hand || x.st.left != y.st.left) { out.push_back("jokers[" + std::to_string(i) + "]"); break; }
        }
        if (a.shop.size() != b.shop.size()) out.push_back("shop.size");
        if (a.shop_packs.size() != b.shop_packs.size()) out.push_back("shop_packs.size");
        if (a.shop_vouchers.size() != b.shop_vouchers.size()) out.push_back("shop_vouchers.size");
        if (a.has_ante_voucher != b.has_ante_voucher) out.push_back("has_ante_voucher");
        if (a.graveyard.size() != b.graveyard.size()) out.push_back("graveyard.size");
        if (a.ante_played != b.ante_played) out.push_back("ante_played");
        if (a.sg.st != b.sg.st) out.push_back("sg.st(streams)");
        for (const auto& kv : a.sg.st) { auto it = b.sg.st.find(kv.first); if (it == b.sg.st.end()) out.push_back("st-only-in-first:" + kv.first); else if (it->second != kv.second) out.push_back("st-differs:" + kv.first); }
        for (const auto& kv : b.sg.st) if (!a.sg.st.count(kv.first)) out.push_back("st-only-in-second:" + kv.first);
        if (a.sg.seed != b.sg.seed) out.push_back("sg.seed");
        if (a.sg.hashed_seed != b.sg.hashed_seed) out.push_back("sg.hashed_seed");
        if (a.rng.index != b.rng.index || std::memcmp(a.rng.mt, b.rng.mt, sizeof a.rng.mt) != 0) out.push_back("g.rng(PyRandom)");
        if (std::memcmp(a.sg.rng.u, b.sg.rng.u, sizeof a.sg.rng.u) != 0) out.push_back("sg.rng");
        if (a.exact != b.exact) out.push_back("exact");
        if (!a.exact) out.push_back("NOT-EXACT");
        if (a.jokers.size() != b.jokers.size()) out.push_back("jokers.size");
        if (a.cards.size() != b.cards.size()) out.push_back("cards.size");
        if (a.full_deck != b.full_deck) out.push_back("full_deck(ids)");
        if (a.deck.size() != b.deck.size()) out.push_back("deck.size");
        if (a.consumables.size() != b.consumables.size()) out.push_back("consumables.size");
        return out;
    }
    // the scalar members of the ENVIRONMENT (not the game) that differ from another's: a debugging aid
    std::vector<std::string> diff_env_names(const Env& o) const {
        std::vector<std::string> out;
        if (deck != o.deck) out.push_back("deck");
        if (stake != o.stake) out.push_back("stake");
        if (win_ante != o.win_ante) out.push_back("win_ante");
        if (start_xscale != o.start_xscale) out.push_back("start_xscale");
        if (start_xscale_val != o.start_xscale_val) out.push_back("start_xscale_val");
        if (start_xmult != o.start_xmult) out.push_back("start_xmult");
        if (start_any != o.start_any) out.push_back("start_any");
        if (verdant_bonus != o.verdant_bonus) out.push_back("verdant_bonus");
        if (school_p != o.school_p) out.push_back("school_p");
        if (school_student != o.school_student) out.push_back("school_student");
        if (school_shop_seen != o.school_shop_seen) out.push_back("school_shop_seen");
        if (death_pen != o.death_pen) out.push_back("death_pen");
        if (death_cap != o.death_cap) out.push_back("death_cap");
        if (ban_p != o.ban_p) out.push_back("ban_p");
        if (tilt_p != o.tilt_p) out.push_back("tilt_p");
        if (exact != o.exact) out.push_back("exact");
        if (tilt_levels != o.tilt_levels) out.push_back("tilt_levels");
        if (restricted != o.restricted) out.push_back("restricted");
        if (level_cap_p != o.level_cap_p) out.push_back("level_cap_p");
        if (level_cap_n != o.level_cap_n) out.push_back("level_cap_n");
        if (level_cap_ante != o.level_cap_ante) out.push_back("level_cap_ante");
        if (level_capped_game != o.level_capped_game) out.push_back("level_capped_game");
        if (part_credit != o.part_credit) out.push_back("part_credit");
        if (log_partial != o.log_partial) out.push_back("log_partial");
        if (level_cap_mode != o.level_cap_mode) out.push_back("level_cap_mode");
        if (pack_restock != o.pack_restock) out.push_back("pack_restock");
        if (start_grow != o.start_grow) out.push_back("start_grow");
        if (scn_p != o.scn_p) out.push_back("scn_p");
        if (scn_min_ante != o.scn_min_ante) out.push_back("scn_min_ante");
        if (scn_max_ante != o.scn_max_ante) out.push_back("scn_max_ante");
        if (scn_reroll_cap != o.scn_reroll_cap) out.push_back("scn_reroll_cap");
        if (scn_couple != o.scn_couple) out.push_back("scn_couple");
        if (arch_p != o.arch_p) out.push_back("arch_p");
        if (arch_store != o.arch_store) out.push_back("arch_store");
        if (arch_lo != o.arch_lo) out.push_back("arch_lo");
        if (arch_hi != o.arch_hi) out.push_back("arch_hi");
        if (arch_min != o.arch_min) out.push_back("arch_min");
        if (arch_rng != o.arch_rng) out.push_back("arch_rng");
        if (arch_winners != o.arch_winners) out.push_back("arch_winners");
        if (arch_start_blind != o.arch_start_blind) out.push_back("arch_start_blind");
        if (autoplay != o.autoplay) out.push_back("autoplay");
        if (autoplay_nodes != o.autoplay_nodes) out.push_back("autoplay_nodes");
        if (menu_slots != o.menu_slots) out.push_back("menu_slots");
        if (immortal_pay != o.immortal_pay) out.push_back("immortal_pay");
        if (immortal != o.immortal) out.push_back("immortal");
        if (last_bits != o.last_bits) out.push_back("last_bits");
        if (scn_power_lo != o.scn_power_lo) out.push_back("scn_power_lo");
        if (scn_power_hi != o.scn_power_hi) out.push_back("scn_power_hi");
        if (scn_build_seed != o.scn_build_seed) out.push_back("scn_build_seed");
        if (scn_force_blind != o.scn_force_blind) out.push_back("scn_force_blind");
        if (scn_level_conc != o.scn_level_conc) out.push_back("scn_level_conc");
        if (scn_kind != o.scn_kind) out.push_back("scn_kind");
        if (last_score != o.last_score) out.push_back("last_score");
        if (scn_limit != o.scn_limit) out.push_back("scn_limit");
        if (reroll_cap != o.reroll_cap) out.push_back("reroll_cap");
        if (reroll_cap_base != o.reroll_cap_base) out.push_back("reroll_cap_base");
        if (flash_cap_cfg != o.flash_cap_cfg) out.push_back("flash_cap_cfg");
        if (hand_level_cap_cfg != o.hand_level_cap_cfg) out.push_back("hand_level_cap_cfg");
        if (scaled_start != o.scaled_start) out.push_back("scaled_start");
        if (scaled_shop != o.scaled_shop) out.push_back("scaled_shop");
        if (scaled_rounds != o.scaled_rounds) out.push_back("scaled_rounds");
        if (start_ante != o.start_ante) out.push_back("start_ante");
        if (start_money_p != o.start_money_p) out.push_back("start_money_p");
        if (start_money2_p != o.start_money2_p) out.push_back("start_money2_p");
        if (start_money != o.start_money) out.push_back("start_money");
        if (start_money2 != o.start_money2) out.push_back("start_money2");
        if (win_bonus != o.win_bonus) out.push_back("win_bonus");
        if (shape_chips != o.shape_chips) out.push_back("shape_chips");
        if (shape_gamma != o.shape_gamma) out.push_back("shape_gamma");
        if (blind_scale != o.blind_scale) out.push_back("blind_scale");
        if (verdant_rel != o.verdant_rel) out.push_back("verdant_rel");
        if (ante8_bonus != o.ante8_bonus) out.push_back("ante8_bonus");
        if (sparse_from_blind != o.sparse_from_blind) out.push_back("sparse_from_blind");
        if (steps != o.steps) out.push_back("steps");
        if (rerolls != o.rerolls) out.push_back("rerolls");
        if (swaps != o.swaps) out.push_back("swaps");
        if (phi != o.phi) out.push_back("phi");
        return out;
    }
    uint64_t marg_signature() const {
        uint64_t h = 0xCBF29CE484222325ull;
        mix(h, (uint64_t)g.ante); mix(h, (uint64_t)g.boss); mix(h, (uint64_t)g.deck_type); mix(h, (uint64_t)g.stake);
        mix(h, g.vouchers); mix(h, (uint64_t)g.joker_slots); mix(h, (uint64_t)g.hand_size);
        mix(h, (uint64_t)g.base_hands); mix(h, (uint64_t)g.base_discards); mix(h, (uint64_t)g.juggle);
        mix(h, (uint64_t)g.hand_size_override); mix(h, (uint64_t)g.starting_deck_size);
        mix(h, (uint64_t)g.blinds_skipped); mix(h, (uint64_t)g.tarots_used);
        for (int q = 0; q < N_HANDS; q++) { mix(h, (uint64_t)g.hand_levels[q]); mix(h, (uint64_t)g.hand_played[q]); }
        bool money_matters = false;
        for (const Joker& j : g.jokers) {
            mix(h, (uint64_t)j.key); mix(h, (uint64_t)j.edition); mix(h, (uint64_t)j.debuffed | ((uint64_t)j.hidden << 1));
            mix(h, (uint64_t)j.st.has_val); mix(h, dbits(j.st.val));
            mix(h, (uint64_t)(int64_t)j.st.rank); mix(h, (uint64_t)(int64_t)j.st.suit);
            mix(h, (uint64_t)(int64_t)j.st.hand); mix(h, (uint64_t)(int64_t)j.st.left);
            if (j.key == J_BULL || j.key == J_BOOTSTRAPS) money_matters = true;
        }
        mix(h, (uint64_t)g.jokers.size());
        if (money_matters) mix(h, (uint64_t)g.money);
        if (g.has_voucher(V_OBSERVATORY))                      // the only scoring that reads the consumables (held planets)
            for (const Cons& c : g.consumables) mix(h, (uint64_t)(int64_t)c.id);
        for (int id : g.full_deck) {
            const Card& c = g.cards[id];
            mix(h, (uint64_t)id);
            mix(h, (uint64_t)(uint8_t)c.rank | ((uint64_t)(uint8_t)c.suit << 8) | ((uint64_t)c.enh << 16) |
                   ((uint64_t)c.ed << 24) | ((uint64_t)c.seal << 32) | ((uint64_t)c.deb << 40) | ((uint64_t)c.hidden << 41));
            mix(h, (uint64_t)(int64_t)c.extra);
        }
        mix(h, (uint64_t)g.full_deck.size());
        return h;
    }
    struct MargBuild {
        MargCtx m;
        double base = 0.0;
        double rm[MAX_JOKERS];
        double pj[MAX_JOKERS];                       // the score with that held joker grown to its potential
        double pall = 0.0;                           // ... with all growing jokers grown
        bool any_grower = false;
    };
    // the counter of a growing joker after its potential growth
    static void grow_to_potential(Joker& j, const Game* s = nullptr) {
        if (POT_MODE == 2 && s && (j.key == J_LUCKY_CAT || j.key == J_GLASS || j.key == J_VAMPIRE)) {
            const double scored = 8.0;                                   // cards scored per round: about 2 plays of 4
            int lucky = 0, glass = 0, enh = 0, n = 0;
            for (int id : s->full_deck) {
                const Card& c = s->cards[id];
                n++;
                if (c.enh == E_LUCKY) lucky++;
                if (c.enh == E_GLASS) glass++;
                if (c.enh != E_NONE) enh++;
            }
            n = std::max(n, 1);
            double rate = j.key == J_LUCKY_CAT ? scored * lucky / n * 0.253 * 0.25
                        : j.key == J_GLASS ? scored * glass / n * 0.25 * 0.75 : scored * enh / n * 0.1;
            j.st.set(j.st.get(1.0) + rate * POT_ROUNDS);
            return;
        }
        if (POT_MODE == 1 || POT_MODE == 2) {
            double rate, v0;
            if (!pot2_rate(j.key, rate, v0)) return;
            double d = rate < 0 ? 1.0 : rate * POT_ROUNDS;
            j.st.set(j.st.get(v0) + d);
            return;
        }
        j.st.set(j.st.get(JOKER_POT_VAL0[j.key]) + JOKER_POT_DELTA[j.key]);
    }
    // the build-dependent part: sampled hands, the best play of each hand type, the base score, each held joker removed
    void marg_build(Game& s, MargBuild& b) {
        marg_setup(s, b.m);
        double base = 0.0;
        for (int k = 0; k < MARG_K; k++) base += marg_best(s, b.m, k, true);
        b.base = base / MARG_K;
        const std::vector<Joker> jokers0 = s.jokers;
        int nj = std::min((int)jokers0.size(), MAX_JOKERS);
        for (int i = 0; i < nj; i++) {
            s.jokers = jokers0;
            s.jokers.erase(s.jokers.begin() + i);
            b.rm[i] = marg_mean(s, b.m);
        }
        b.any_grower = false;
        b.pall = b.base;
        for (int i = 0; i < MAX_JOKERS; i++) b.pj[i] = b.base;
        if (marginal >= 2) {
            s.jokers = jokers0;
            for (int i = 0; i < nj; i++) {
                if (!pot_on(jokers0[i].key)) continue;
                b.any_grower = true;
                s.jokers = jokers0;
                grow_to_potential(s.jokers[i], &s);
                b.pj[i] = marg_mean(s, b.m);
            }
            if (b.any_grower) {
                s.jokers = jokers0;
                for (int i = 0; i < nj; i++) if (pot_on(s.jokers[i].key)) grow_to_potential(s.jokers[i], &s);
                b.pall = marg_mean(s, b.m);
            }
        }
        s.jokers = jokers0;
    }
    bool marg_valid = false, marg_verify = false;
    long long marg_hits = 0, marg_misses = 0;                // cache statistics
    std::vector<std::pair<int, double>> marg_dbg;           // (row, mean) of the item rows of the last fill, for debugging
    double marg_dbg_base = 0.0;
    uint64_t marg_sig = 0;
    MargBuild marg_cache;
    void marginal_fill() {
        if (!mscratch) mscratch.reset(new Game());
        Game& s = *mscratch;
        uint64_t sig = marg_signature();
        if (!(marg_valid && sig == marg_sig)) {
            marg_misses++;
            marg_build(s, marg_cache);
            marg_sig = sig;
            marg_valid = true;
        } else {
            marg_hits++;
            if (marg_verify) {
                MargBuild fresh;
                Game chk;
                marg_build(chk, fresh);
                bool same = fresh.base == marg_cache.base;
                for (int i = 0; i < std::min((int)g.jokers.size(), MAX_JOKERS); i++)
                    same = same && fresh.rm[i] == marg_cache.rm[i] && fresh.pj[i] == marg_cache.pj[i];
                same = same && fresh.pall == marg_cache.pall;
                if (!same) throw std::runtime_error("marginal cache: the cached build values differ from a fresh computation");
            }
        }
        MargCtx& m = marg_cache.m;
        const double base = marg_cache.base;
        marg_dbg.clear();
        marg_dbg_base = base;
        s.money = g.money;
        const std::vector<Joker> jokers0 = s.jokers;
        p.g[F_GLOBAL_BASE] = (float)(std::log10(1.0 + base) / 7.0);
        double hands = g.round_hands();
        for (int i = 0; i < 3; i++)
            p.g[F_GLOBAL_BASE + 1 + i] = (float)std::min(5.0, hands * base / std::max(1.0, (double)g.blind_target(i)));
        auto with_jokers = [&](int remove, const Joker* add, bool potential = false) -> double {
            int slots = s.joker_slots;                       // (a Negative joker adds a slot; its init may draw random numbers)
            s.jokers = jokers0;
            if (remove >= 0) s.jokers.erase(s.jokers.begin() + remove);
            if (add) {
                Joker j = *add;
                j.uid = 0;
                s.rng = PyRandom(12345);
                s.add_joker(j);
                s.order_jokers();
                if (potential && pot_on(j.key)) {
                    int uid = s.next_joker_uid - 1;                  // the joker just added
                    for (Joker& x : s.jokers) if (x.uid == uid) grow_to_potential(x, &s);
                }
            }
            double v = marg_mean(s, m);
            s.jokers = jokers0;
            s.joker_slots = slots;
            return v;
        };
        auto with_level = [&](int h) -> double {
            s.hand_levels[h] += 1;
            double v = marg_mean(s, m);
            s.hand_levels[h] -= 1;
            return v;
        };
        auto row = [&](int a) { return p.af + (size_t)a * F_ACT + F_ACT_BASE; };
        int nj = std::min((int)g.jokers.size(), MAX_JOKERS);
        for (int i = 0; i < nj; i++) {
            double v = marg_cache.rm[i];
            float* jr = p.jf + (size_t)i * F_JOKER + F_JOKER_BASE;
            jr[0] = (float)mclip(1.0 - v / base, -1.0, 1.0);
            jr[1] = (float)(mclip(std::log10((base + 1.0) / (v + 1.0)), -1.0, 3.0) / 3.0);
        }
        auto potv = [&](double mean) { return (float)(mclip(std::log10((mean + 1.0) / (base + 1.0)), -1.0, 3.0) / 3.0); };
        const bool pot = marginal >= 2;
        auto prow = [&](int a) { return p.af + (size_t)a * F_ACT + F_ACT_M; };
        if (pot) {
            for (int i = 0; i < nj; i++)
                if (pot_on(g.jokers[i].key)) p.jf[(size_t)i * F_JOKER + F_JOKER_M] = potv(marg_cache.pj[i]);
            if (marg_cache.any_grower) p.g[F_GLOBAL_M] = potv(marg_cache.pall);
        }
        if (g.targeting) return;
        if (g.state == ST_SHOP) {
            for (int i = 0; i < (int)g.shop.size() && i < MAX_SHOP; i++) {
                const ShopItem& it = g.shop[i];
                if (it.kind == SK_JOKER) {
                    double mv0 = with_jokers(-1, &it.joker);
                    marg_dbg.push_back({A_BUY + i, mv0});
                    marg_gain(row(A_BUY + i), base, mv0, it.cost);
                    const bool grows = pot && pot_on(it.joker.key);
                    if (grows) *prow(A_BUY + i) = potv(with_jokers(-1, &it.joker, true));
                    for (int j = 0; j < (int)g.jokers.size() && j < MAX_JOKERS; j++) {
                        int a = A_REPLACE + i * MAX_JOKERS + j;
                        if (p.mask[a]) {
                            marg_gain(row(a), base, with_jokers(j, &it.joker), (double)it.cost - g.jokers[j].sell_value());
                            if (grows) *prow(a) = potv(with_jokers(j, &it.joker, true));
                        }
                    }
                } else if (it.kind == SK_PLANET) {
                    marg_gain(row(A_BUY + i), base, with_level(CONS_DEFS[it.key].planet_hand), it.cost);
                }
            }
        }
        if (g.state == ST_PACK) {
            for (int i = 0; i < (int)g.pack_cards.size() && i < MAX_PACK; i++) {
                const PackItem& x = g.pack_cards[i];
                if (x.kind == PI_JOKER) {
                    marg_gain(row(A_PICK + i), base, with_jokers(-1, &x.joker), 0);
                    if (pot && pot_on(x.joker.key)) *prow(A_PICK + i) = potv(with_jokers(-1, &x.joker, true));
                }
                else if (x.kind == PI_CONS && CONS_DEFS[x.cons.id].kind == CK_PLANET)
                    marg_gain(row(A_PICK + i), base, with_level(CONS_DEFS[x.cons.id].planet_hand), 0);
            }
        }
    }

    bool priv_on = false;                    // fill the privileged observation (VecEnv.set_priv)
    float priv_shop[61];                     // the next shop's features: recomputed outside blinds, kept inside them
    bool priv_shop_valid = false;
    float priv_fut[FUT_K * FUT_W];           // the future list, recomputed when the ante changes (what it shows depends on the actions little)
    int fut_ante = -1;
    void encode_future() {
        std::fill(priv_fut, priv_fut + FUT_K * FUT_W, 0.0f);
        try {
            Game g2 = g;
            g2.money = 100000;
            for (int k = 0; k < FUT_K; k++) {
                int a = g.ante + k;
                if (a > WIN_ANTE) break;
                float* f = priv_fut + (size_t)k * FUT_W;
                g2.ante = a;
                g2.blind_idx = 0;
                if (k > 0) g2.new_ante();
                if (g2.boss >= 0 && g2.boss < B_COUNT) f[J_COUNT + V_COUNT + N_PACKS + g2.boss] = 1.0f;
                g2.open_shop();
                auto note = [&](float val) {
                    for (const ShopItem& it : g2.shop)
                        if (it.kind == SK_JOKER && it.key >= 0 && it.key < J_COUNT) f[it.key] = std::max(f[it.key], val);
                };
                note(1.0f);
                for (const ShopItem& it : g2.shop_packs) if (it.key >= 0 && it.key < N_PACKS) f[J_COUNT + V_COUNT + it.key] = 1.0f;
                if (!g2.shop_vouchers.empty() && g2.shop_vouchers[0].key < V_COUNT) f[J_COUNT + g2.shop_vouchers[0].key] = 1.0f;
                for (int r = 1; r <= FUT_ROLLS; r++) {
                    g2.free_rerolls = 1;
                    g2.reroll();
                    note(1.0f - 0.15f * r);
                }
            }
        } catch (...) {}
    }
    void encode_priv() {
        float* v = p.priv;
        std::fill(v, v + F_PRIV, 0.0f);
        // --- the next shop, on a copy of the game: the shop that opens after the blind now in front of us (or, from a shop, after the next one)
        if (!priv_shop_valid || g.state != ST_SELECTING_HAND) {
            std::fill(priv_shop, priv_shop + 61, 0.0f);
            if (g.exact) {
                Game g2 = g;
                if (g2.state == ST_SHOP) {                       // leaving: the next blind comes first
                    if (g2.blind_idx == 2) { g2.ante += 1; g2.blind_idx = 0; g2.new_ante(); } else g2.blind_idx += 1;
                } else if (g2.blind_idx == 2) {                  // the boss of this ante: the shop is the next ante's
                    g2.ante += 1; g2.blind_idx = 0; g2.new_ante();
                }
                if (g2.state == ST_PACK) g2.state = ST_SHOP;
                g2.open_shop();
                int o = 0;
                for (int i = 0; i < 4; i++, o += 11) {
                    if (i >= (int)g2.shop.size()) continue;
                    const ShopItem& it = g2.shop[i];
                    float* f = priv_shop + o;
                    f[std::min<int>(it.kind, 4)] = 1.0f;
                    f[5] = std::min(it.cost, 30) / 10.0f;
                    if (it.kind == SK_JOKER) {
                        f[6] = JOKER_DEFS[it.key].rarity / 4.0f;
                        int role = JOKER_SWAP_ROLE[it.key];
                        if (role >= 0 && role < 4) f[7 + role] = 1.0f;
                    }
                }
                for (int i = 0; i < 2 && i < (int)g2.shop_packs.size(); i++, o += 7) {
                    float* f = priv_shop + o;
                    const PackDef& pd = PACK_DEFS[g2.shop_packs[i].key];
                    f[std::min<int>(pd.kind, 4)] = 1.0f;
                    f[5] = pd.size / 2.0f;
                    f[6] = std::min(g2.shop_packs[i].cost, 30) / 10.0f;
                }
                if (!g2.shop_vouchers.empty()) {
                    priv_shop[58] = 1.0f;
                    priv_shop[59] = g2.shop_vouchers[0].key / (float)V_COUNT;
                    priv_shop[60] = std::min(g2.shop_vouchers[0].cost, 30) / 20.0f;
                }
            }
            priv_shop_valid = true;
        }
        for (int i = 0; i < 61; i++) v[i] = priv_shop[i];
        // --- the boss of the next ante (the counts of the bosses used decide it)
        if (g.exact && g.state != ST_GAME_OVER && g.state != ST_WON) {
            Game g3 = g;
            g3.ante = g.blind_idx == 2 && g.state != ST_BLIND_SELECT ? g.ante + 2 : g.ante + 1;
            int b = g3.exact_boss();
            if (b >= 0 && b < B_COUNT && b < 28) v[61 + b] = 1.0f;
        }
        // --- the cards on top of the deck: the hand drawn next. Inside a blind it is the deck as it is; before one, the deck as the blind's
        // shuffle ('nr' + ante) will make it
        std::vector<int> deck_now;
        if (g.state == ST_SELECTING_HAND || !g.exact) {
            deck_now = g.deck;
        } else {
            deck_now = g.full_deck;
            std::stable_sort(deck_now.begin(), deck_now.end());
            Streams s2 = g.sg;
            int a = g.state == ST_SHOP && g.blind_idx == 2 ? g.ante + 1 : g.ante;
            if (g.state == ST_BLIND_SELECT || g.state == ST_PACK) a = g.ante;
            s2.shuffle(deck_now, "nr" + std::to_string(a));
        }
        for (int i = 0; i < 12 && i < (int)deck_now.size(); i++) {
            const Card& c = g.cards[deck_now[(int)deck_now.size() - 1 - i]];
            float* f = v + 89 + i * 9;
            f[0] = c.is_stone() ? 0.0f : c.rank / 14.0f;
            if (c.suit >= 0 && c.suit < 4) f[1 + c.suit] = 1.0f;
            f[5] = c.enh != E_NONE ? 1.0f : 0.0f;
            f[6] = c.ed != ED_NONE ? 1.0f : 0.0f;
            f[7] = c.seal != S_NONE ? 1.0f : 0.0f;
            f[8] = c.is_stone() ? 1.0f : 0.0f;
        }
        if (g.exact && g.state != ST_GAME_OVER && g.state != ST_WON) {
            if (fut_ante != g.ante) { encode_future(); fut_ante = g.ante; }
            std::memcpy(v + 200, priv_fut, sizeof priv_fut);
        }
    }
    void encode() {
        encode_core();
        if (priv_on) encode_priv();
    }
    void encode_core() {
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
        if (marginal && g.state != ST_SELECTING_HAND && (!marg_blind_only || g.state == ST_BLIND_SELECT)) marginal_fill();
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
        else if (a < A_PICK) g.use_consumable(a - A_USE_C, !auto_target);
        else if (a < A_PSKIP) g.pack_pick(a - A_PICK, !auto_target);
        else if (a == A_PSKIP) g.pack_skip();
        else if (a < A_SWAP_END) g.swap_jokers(a - A_SWAP);
        else if (a < N_ACTIONS) {
            int i = (a - A_REPLACE) / MAX_JOKERS, j = (a - A_REPLACE) % MAX_JOKERS;
            g.sell_joker(j);
            if (i < (int)g.shop.size() && g.can_buy(g.shop[i])) g.buy_card(i);
        }
        else throw std::runtime_error("bad action");
    }
    // --autoplay: at a blind select the action "leave shop" (never legal there otherwise) means: select the blind and let the exact round solver play it (the plan that
    // clears it with the most money left). The network keeps the final say: the ordinary select plays the hands itself. No plan found: the blind is selected and play goes on by hand.
    StepResult step_auto();
    // ---- scripted mode 2: how safe the blind is decides how much to invest in the build's growth.
    //   margin = P10(round score over redraws) / what is still needed; the redraws come from the known multiset of the hand and the draw pile (never the order the game will deal);
    //   a round's score = the sum over the hands left of a best single-hand score (every candidate play scored by the exact scorer), drawn at random from the redraws.
    //   lambda = clamp((margin - 1) / (m_full - 1), 0, 1), m_full larger on a boss and in later antes. Each candidate play is scored
    //       Score(a) = Power_now(a) + lambda x Potential(a),   Power_now = the play's exact score,
    //       Potential(a) = sum over the growing jokers of (their counter growth by a, in natural rounds) x (the score one natural round of growth adds per hand, measured on the
    //       redrawn hands with the exact scorer) x (hands left in the run to ante 8).  The later in the run, the fewer hands remain, so the same growth is worth less.
    //   A play that clears the blind is always preferred to one that does not.
    double sp_mfull = 1.6;
    double sp_force_lambda = -1.0;                     // >= 0: lambda is this whatever the margin (the growth sweep: only the growth matters)
    int sp_samples = 24;
    std::vector<double> sp_pool;                       // best single-hand score of each redrawn hand (this round)
    std::vector<int> sp_gain_uid;                      // growing jokers held (by uid) ...
    std::vector<double> sp_gain_val, sp_gain_rate;     // ... the score a natural round of growth adds per hand, and the counter's natural rate
    uint64_t sp_rng = 0x9E3779B97F4A7C15ull;
    double rr_margin = 0.0, rr_lambda = 0.0;           // the margin and lambda at the first decision of the last scripted round
    bool sp_first = true;
    uint64_t sp_next() { sp_rng ^= sp_rng << 13; sp_rng ^= sp_rng >> 7; sp_rng ^= sp_rng << 17; return sp_rng; }
    Env& sp_tmp() {
        thread_local std::unique_ptr<Env> t;
        if (!t) t = std::make_unique<Env>(deck, stake, 1.0, 10.0, win_ante);
        t->copy_config_from(*this);
        t->marginal = 0;
        t->unchecked = true;
        return *t;
    }
    static double best_pred_of(const Env& e) {
        double b = 0.0;
        for (int i = 0; i < N_SUB; i++) if (e.play_len[i] > 0 && e.pred_sc[i] > b) b = e.pred_sc[i];
        return b;
    }
    void sp_build_pool() {
        sp_pool.clear(); sp_gain_uid.clear(); sp_gain_val.clear(); sp_gain_rate.clear();
        Env& t = sp_tmp();
        std::vector<int> all(g.hand.begin(), g.hand.end());
        all.insert(all.end(), g.deck.begin(), g.deck.end());
        const int nh = (int)g.hand.size();
        std::vector<int> gi;                           // indices of the growing jokers
        for (int j = 0; j < (int)g.jokers.size(); j++) {
            const ScnGrow* sg = grow_entry(g.jokers[j].key);
            if (sg && !g.jokers[j].debuffed) { gi.push_back(j); sp_gain_uid.push_back(g.jokers[j].uid); sp_gain_val.push_back(0.0); sp_gain_rate.push_back(sg->per_round); }
        }
        std::vector<int> pool;
        for (int k = 0; k < sp_samples; k++) {
            pool = all;
            for (int i = 0; i < nh && i < (int)pool.size(); i++) { int j = i + (int)(sp_next() % (uint64_t)(pool.size() - i)); std::swap(pool[i], pool[j]); }
            t.copy_state_from(*this, false);
            t.g.hand.assign(pool.begin(), pool.begin() + std::min((int)pool.size(), nh));
            t.g.deck.assign(pool.begin() + std::min((int)pool.size(), nh), pool.end());
            t.candidates();
            double best = best_pred_of(t);
            sp_pool.push_back(best);
            for (size_t q = 0; q < gi.size(); q++) {
                Joker& jj = t.g.jokers[gi[q]];
                const ScnGrow* sg = grow_entry(jj.key);
                double c0 = jj.st.get(sg->init);
                jj.st.set(c0 + sg->per_round);
                t.candidates();
                sp_gain_val[q] += std::max(0.0, best_pred_of(t) - best) / sp_samples;
                jj.st.set(c0);
            }
        }
    }
    // a fast measure of a build's power at the start of a blind, without searching the round: `samples` hands redrawn from the hand and the draw pile, each one's best
    // single play scored by the exact scorer (every joker, level, enhancement). -> {mean best play, max best play}. The redraws use their own generator, seeded from the state.
    std::pair<double, double> fast_power(int samples) {
        Env& t = sp_tmp();
        uint64_t rs = FP_CRN_SEED ? FP_CRN_SEED * 2654435761ull + 7ull
                                  : (exact ? state_key() : (uint64_t)(g.ante * 7919 + g.blind_idx * 104729 + g.money + 31 * (int)g.full_deck.size())) * 2654435761ull + 7ull;
        auto nxt = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
        // in a round: the hand and the draw pile; outside one (shop, blind select): a full hand from the whole deck (the hand is empty there: before, this
        // measured one-card hands and every caller outside a round, the heuristic bot's "power gap" among them, got a meaningless number)
        const bool inr = g.state == ST_SELECTING_HAND && !g.hand.empty();
        std::vector<int> all;
        if (inr) { all.assign(g.hand.begin(), g.hand.end()); all.insert(all.end(), g.deck.begin(), g.deck.end()); if (FP_CRN_SEED) std::sort(all.begin(), all.end()); }
        else { all.assign(g.full_deck.begin(), g.full_deck.end()); if (FP_CRN_SEED) std::sort(all.begin(), all.end()); }     // (common random numbers: a canonical order)
        const int nh = inr ? (int)g.hand.size() : std::max(1, std::min(MAX_HAND, g.effective_hand_size()));
        double sum = 0.0, mx = 0.0;
        std::vector<int> pool;
        for (int k = 0; k < samples; k++) {
            pool = all;
            for (int i = 0; i < nh && i < (int)pool.size(); i++) { int j = i + (int)(nxt() % (uint64_t)(pool.size() - i)); std::swap(pool[i], pool[j]); }
            t.copy_state_from(*this, false);
            t.g.hand.assign(pool.begin(), pool.begin() + std::min((int)pool.size(), nh));
            t.g.deck.assign(pool.begin() + std::min((int)pool.size(), nh), pool.end());
            t.candidates();
            double b = best_pred_of(t);
            sum += b;
            mx = std::max(mx, b);
        }
        return {sum / std::max(1, samples), mx};
    }
    // the build as it will be at the ante-8 boss: every joker in SCN_GROW moved by its natural rate over the rounds left (growing ones up, decaying ones down, kept in
    // [lo, hi]), then faced with an ante-8 boss and measured by fast_power. -> hands x mean best play / target (a share: >= 1 the build would clear it). Outside a round only.
    double projected_power(int samples) { return projected_share(0, samples); }
    // the build projected to the boss of ante `to_ante` (0: the win ante): projected_power's growth over the rounds until then, the boss of that ante faced with
    // fast_power. -> hands x mean best play / target. Outside a round only; an ante already passed gives 0.
    double projected_share(int to_ante, int samples) {
        thread_local std::unique_ptr<Env> pt;
        if (!pt) pt = std::make_unique<Env>(deck, stake, 1.0, 10.0, win_ante);
        pt->copy_config_from(*this);
        pt->marginal = 0;
        pt->unchecked = true;
        pt->auto_round = false;
        pt->shop_only = false;
        pt->copy_state_from(*this, false);
        Game& h = pt->g;
        if (h.done()) return 0.0;
        const int ta = to_ante > 0 ? std::min(to_ante, h.win_ante) : h.win_ante;
        if (ta < h.ante) return 0.0;
        int rounds_left = std::max(0, 3 * (ta - h.ante) + (3 - std::min(h.blind_idx, 3)));
        for (Joker& j : h.jokers)
            for (const ScnGrow& sgr : SCN_GROW)
                if (sgr.key == j.key) {
                    double v = j.st.has_val ? j.st.val : sgr.init;
                    j.st.set(std::min(std::max(v + sgr.per_round * rounds_left, sgr.lo), sgr.hi));
                }
        // multiplier scalers that are not in SCN_GROW: Madness +0.5 at each small / big blind selected, Yorick +1 per 23 cards discarded (~10 a round),
        // Throwback +0.25 per blind skipped (the run's skip rate so far, carried over the rounds left)
        int done_blinds = std::max(1, 3 * (h.ante - 1) + h.blind_idx);
        double skip_rate = (double)h.blinds_skipped / done_blinds;
        for (Joker& j : h.jokers) {
            if (j.key == J_MADNESS) j.st.set(j.st.get(1.0) + 0.5 * rounds_left * 2.0 / 3.0);
            else if (j.key == J_YORICK) j.st.set(j.st.get(1.0) + rounds_left * 10.0 / 23.0);
        }
        h.blinds_skipped += (int)std::lround(skip_rate * rounds_left);
        h.ante = ta;
        h.blind_idx = 2;
        h.new_ante();
        h.blind_idx = 2;
        if (CURVE_GENERIC_BOSS) h.boss = B_NONE;                       // (a generic x2 boss: the same target for every candidate, no boss effect)
        h.state = ST_BLIND_SELECT;
        pt->encode();
        pt->step_core(A_SELECT);
        if (h.state != ST_SELECTING_HAND) return 0.0;
        auto pr = pt->fast_power(samples);
        return std::max(1, h.hands_left) * pr.first / std::max(1.0, h.target);
    }
    double sp_margin_now() {
        const int H = std::max(1, g.hands_left), T = 200;
        std::vector<double> sums(T);
        for (int t = 0; t < T; t++) { double sm = 0.0; for (int h = 0; h < H; h++) sm += sp_pool[sp_next() % sp_pool.size()]; sums[t] = sm; }
        std::sort(sums.begin(), sums.end());
        return sums[T / 10] / std::max(1.0, g.target - g.chips);
    }
    int scripted_action2() {
        if (sp_pool.empty()) sp_build_pool();
        double margin = sp_margin_now();
        double mfull = sp_mfull + 0.05 * (g.ante - 1) + (g.blind_idx == 2 ? 0.2 : 0.0);
        double lambda = std::min(1.0, std::max(0.0, (margin - 1.0) / (mfull - 1.0)));
        if (sp_force_lambda >= 0.0) lambda = sp_force_lambda;
        if (sp_first) { rr_margin = margin; rr_lambda = lambda; sp_first = false; }
        int b = -1;
        double bs = -1.0;
        for (int i = 0; i < N_SUB; i++) if (play_len[i] > 0 && pred_sc[i] > bs) { bs = pred_sc[i]; b = i; }
        if (b < 0) {
            for (int i = 0; i < N_SUB; i++) if (disc_len[i] > 0 && g.discards_left > 0) return A_DISC + i;
            return -1;
        }
        const double need = std::max(1.0, g.target - g.chips);
        const int hands = std::max(1, g.hands_left);
        int chosen = b;
        if (lambda > 0.0 && !sp_gain_uid.empty()) {
            const int rr = std::max(1, 3 * (8 - g.ante) + (2 - g.blind_idx));
            const double H = rr * 3.5;                                           // hands left in the run, about
            std::vector<int> idx;
            for (int i = 0; i < N_SUB; i++) if (play_len[i] > 0) idx.push_back(i);
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int c) { return pred_sc[a] > pred_sc[c]; });
            if (idx.size() > 24) idx.resize(24);
            Env& t = sp_tmp();
            std::vector<double> c0(sp_gain_uid.size(), 0.0);
            for (size_t q = 0; q < sp_gain_uid.size(); q++) for (const Joker& j : g.jokers) if (j.uid == sp_gain_uid[q]) { const ScnGrow* sg = grow_entry(j.key); c0[q] = j.st.get(sg->init); }
            double best_score = -1e300;
            bool any_clear = false;
            for (int i : idx) if (pred_sc[i] >= need) any_clear = true;
            for (int i : idx) {
                if (any_clear && pred_sc[i] < need) continue;
                t.copy_state_from(*this, false);
                t.lite = true;
                t.step_core(A_PLAY + i);
                double pot = 0.0;
                for (size_t q = 0; q < sp_gain_uid.size(); q++)
                    for (const Joker& j : t.g.jokers) if (j.uid == sp_gain_uid[q]) {
                        const ScnGrow* sg = grow_entry(j.key);
                        double d = j.st.get(sg->init) - c0[q];
                        pot += d / sp_gain_rate[q] * sp_gain_val[q] * H;                // (a counter lost counts against the play as much as one gained counts for it)
                    }
                double sc = pred_sc[i] + lambda * pot;
                if (sc > best_score) { best_score = sc; chosen = i; }
            }
        }
        bool green_keep = g.has(J_GREEN_JOKER) && lambda >= 0.5;
        if (bs >= need || hands <= 1 || g.discards_left <= 0 || bs * hands >= need * (green_keep ? 1.0 : sp_pace)) return A_PLAY + chosen;
        bool used[16] = {false};
        for (int k = 0; k < play_len[b]; k++) used[plays[b][k] & 15] = true;
        int bi = -1, blen = 0;
        double bval = 1e18;
        for (int i = 0; i < N_SUB; i++) {
            if (disc_len[i] == 0) continue;
            bool ok = true;
            double val = 0.0;
            for (int k = 0; k < disc_len[i]; k++) {
                int pos = discs[i][k];
                if (pos < 0 || pos >= (int)g.hand.size() || used[pos & 15]) { ok = false; break; }
                val += g.card_value(g.cards[g.hand[pos]]);
            }
            if (!ok) continue;
            if (disc_len[i] > blen || (disc_len[i] == blen && val < bval)) { bi = i; blen = disc_len[i]; bval = val; }
        }
        return bi >= 0 ? A_DISC + bi : A_PLAY + chosen;
    }
    int scripted_action() {
        if (sp_mode == 2) return scripted_action2();
        int b = -1;
        double bs = -1.0;
        for (int i = 0; i < N_SUB; i++) if (play_len[i] > 0 && pred_sc[i] > bs) { bs = pred_sc[i]; b = i; }
        bool green_keep = false;
        if (sp_mode == 1 && b >= 0) {
            const bool green = g.has(J_GREEN_JOKER), bus = g.has(J_RIDE_THE_BUS), obel = g.has(J_OBELISK), super_ = g.has(J_SUPERNOVA), sq = g.has(J_SQUARE), run = g.has(J_RUNNER);
            const bool par = g.has(J_PAREIDOLIA);
            green_keep = green;
            auto ctr = [&](int key) { for (const Joker& j : g.jokers) if (j.key == key && !j.debuffed) return j.st.get(0.0); return 0.0; };
            const double c_bus = bus ? ctr(J_RIDE_THE_BUS) : 0.0, c_obel = obel ? ctr(J_OBELISK) : 1.0;
            const double f_bus = std::max(0.1, 1.0 / (1.0 + 0.1 * c_bus));              // a counter worth protecting is worth more than a play's score: the bigger it is, the harder the face cards are avoided
            const double f_obel = std::max(0.05, 1.0 / (1.0 + 1.5 * std::max(0.0, c_obel - 1.0)));
            if (bus || obel || super_ || sq || run) {
                int mp = -1, mc = 0;
                for (int h = 0; h < 9; h++) if (g.hand_played[h] > mc) { mc = g.hand_played[h]; mp = h; }
                double need0 = std::max(1.0, g.target - g.chips), bestadj = -1.0, bestclear = -1.0;
                int bi = -1, bc = -1;
                for (int i = 0; i < N_SUB; i++) {
                    if (play_len[i] <= 0) continue;
                    double f = 1.0;
                    if (bus) { bool face = false; for (int k = 0; k < play_len[i]; k++) if (g.cards[g.hand[plays[i][k]]].is_face(par)) face = true; if (face) f *= f_bus; }
                    if (obel && mp >= 0 && pred_hand[i] == mp) f *= f_obel;
                    if (super_ && !obel && mp >= 0 && pred_hand[i] == mp) f *= 1.2;
                    if (sq && play_len[i] == 4) f *= 1.25;
                    if (run && (pred_hand[i] == 4 || pred_hand[i] == 8)) f *= 1.25;
                    double adj = pred_sc[i] * f;
                    if (adj > bestadj) { bestadj = adj; bi = i; }
                    if (pred_sc[i] >= need0 && adj > bestclear) { bestclear = adj; bc = i; }
                }
                b = bc >= 0 ? bc : bi;
                bs = pred_sc[b];
            }
        }
        if (b < 0) {
            for (int i = 0; i < N_SUB; i++) if (disc_len[i] > 0 && g.discards_left > 0) return A_DISC + i;
            return -1;
        }
        double need = std::max(1.0, g.target - g.chips);
        int hands = std::max(1, g.hands_left);
        if (bs >= need || hands <= 1 || g.discards_left <= 0 || bs * hands >= need * (green_keep ? 1.0 : sp_pace)) return A_PLAY + b;
        bool used[16] = {false};
        for (int k = 0; k < play_len[b]; k++) used[plays[b][k] & 15] = true;
        int bi = -1, blen = 0;
        double bval = 1e18;
        for (int i = 0; i < N_SUB; i++) {
            if (disc_len[i] == 0) continue;
            bool ok = true;
            double val = 0.0;
            for (int k = 0; k < disc_len[i]; k++) {
                int pos = discs[i][k];
                if (pos < 0 || pos >= (int)g.hand.size() || used[pos & 15]) { ok = false; break; }
                val += g.card_value(g.cards[g.hand[pos]]);
            }
            if (!ok) continue;
            if (disc_len[i] > blen || (disc_len[i] == blen && val < bval)) { bi = i; blen = disc_len[i]; bval = val; }
        }
        return bi >= 0 ? A_DISC + bi : A_PLAY + b;
    }
    StepResult step(int a) {
        StepResult res = step_core(a);
        if (shop_only && a == A_SELECT && !res.done && g.state == ST_SELECTING_HAND) return shop_only_final(res);
        if (!auto_round || a != A_SELECT || res.done || g.state != ST_SELECTING_HAND) return res;
        rr_hands = 0; rr_disc = 0; rr_ratio = 0.0; rr_clear = 0;
        sp_pool.clear(); sp_first = true; rr_margin = 0.0; rr_lambda = 0.0;
        sp_rng = (exact ? state_key() : (uint64_t)(g.ante * 7919 + g.blind_idx * 104729 + g.money)) * 2654435761ull + 1ull;      // a round is a function of the game state: its sampling does not depend on how the game got here
        double money0 = (double)g.money;
        int fb = g.furthest_blind;
        for (int i = 0; i < MAX_JOKERS; i++) { rr_c0[i] = i < (int)g.jokers.size() && g.jokers[i].st.has_val ? g.jokers[i].st.val : 0.0; rr_c1[i] = rr_c0[i]; }
        lite = true;
        bool first = true;
        for (int guard = 0; guard < 64 && !res.done && g.state == ST_SELECTING_HAND && !g.targeting; guard++) {
            if (!first) candidates();
            first = false;
            int act = scripted_action();
            if (act < 0) break;
            double c0 = g.chips;
            StepResult r2 = step_core(act);
            if (act < A_DISC) { rr_hands++; if (g.target > 0) rr_ratio = std::max(rr_ratio, (c0 + last_score) / g.target); }
            else rr_disc++;
            res.reward += r2.reward; res.sparse += r2.sparse; res.done = r2.done; res.trunc = r2.trunc;
        }
        lite = false;
        rr_clear = g.furthest_blind > fb ? 1 : 0;
        rr_money = (double)g.money - money0;
        for (int i = 0; i < MAX_JOKERS; i++) rr_c1[i] = i < (int)g.jokers.size() && g.jokers[i].st.has_val ? g.jokers[i].st.val : 0.0;
        if (!res.done) {
            encode();
            bool any = false;
            for (int i = 0; i < N_ACTIONS; i++) if (p.mask[i]) { any = true; break; }
            if (!any) { g.state = ST_GAME_OVER; res.done = true; }
        }
        return res;
    }
    // --shop-only: a blind before the final one is won without play. Jokers that grow by playing cards get one round of growth (SCN_GROW); the others
    // (shop-driven: Red Card, Flash, Constellation, Hologram, Campfire; end-of-round ones) grow by themselves. Paid as a win with one hand left.
    void shop_only_win() {
        static const int16_t PLAY_GROW[] = {J_RIDE_THE_BUS, J_GREEN_JOKER, J_RUNNER, J_SQUARE, J_TROUSERS, J_WEE, J_CASTLE, J_OBELISK, J_VAMPIRE,
                                            J_LUCKY_CAT, J_GLASS, J_HIT_THE_ROAD, J_CAINO, J_ICE_CREAM, J_RAMEN, J_SELZER};
        for (Joker& j : g.jokers) {
            if (j.debuffed) continue;
            bool grows = false;
            for (int16_t k : PLAY_GROW) if (j.key == k) grows = true;
            if (!grows) continue;
            for (const ScnGrow& sgr : SCN_GROW)
                if (sgr.key == j.key) {
                    double v = j.st.has_val ? j.st.val : sgr.init;
                    j.st.set(std::min(std::max(v + sgr.per_round, sgr.lo), sgr.hi));
                }
        }
        // jokers that change with the hands played but are not in SCN_GROW: a round taken as ~3 hands of the build's main hand (its highest level, then most played)
        // and ~2 discards of 5 cards
        int h = 0;
        for (int k = 1; k < N_HANDS; k++)
            if (g.hand_levels[k] > g.hand_levels[h] || (g.hand_levels[k] == g.hand_levels[h] && g.hand_played[k] > g.hand_played[h])) h = k;
        if (g.hand_levels[h] <= 1 && g.hand_played[h] == 0) h = 1;                      // nothing chosen yet: a Pair
        g.hand_played[h] += 3;                                                          // Supernova, and every count of hands played
        int nd = (int)g.full_deck.size();
        for (Joker& j : g.jokers) {
            if (j.debuffed) continue;
            if (j.key == J_HIKER && nd > 0) {                                           // 15 card scorings, +5 chips each, spread over the deck
                for (int t = 0; t < 15; t++) g.cards[g.full_deck[(so_rounds * 15 + t) % nd]].extra += 5;
            } else if (j.key == J_MIDAS_MASK) {                                         // face cards scored turn Gold: up to 3 a round
                int made = 0;
                for (int id : g.full_deck) {
                    Card& c = g.cards[id];
                    if (made >= 3) break;
                    if (c.is_face() && !c.is_stone() && c.enh != E_GOLD) { c.enh = E_GOLD; made++; }
                }
            } else if (j.key == J_YORICK) {                                             // 10 cards discarded toward each +1 xMult per 23
                int left = (j.st.left >= 0 ? j.st.left : 23) - 10;
                while (left <= 0) { j.st.set(j.st.get(1.0) + 1); left += 23; }
                j.st.left = left;
            } else if (j.key == J_LOYALTY_CARD) {                                       // its 6-hand cycle advances 3 hands
                double v = j.st.has_val ? j.st.val : 5;
                j.st.set(std::fmod(v - 3 + 60, 6.0));
            } else if (j.key == J_SPACE) {                                              // 1 in 4 per hand to level the hand played: every 4th hand, exactly
                int acc = (j.st.left >= 0 ? j.st.left : 0) + 3;
                while (acc >= 4) { g.hand_levels[h] += 1; acc -= 4; }
                j.st.left = acc;
            }
        }
        so_rounds += 1;
        g.chips = g.target;
        g.hands_left = std::min(g.hands_left, 1);
        g.win_round(false);
    }
    StepResult shop_only_final(StepResult res);
    std::pair<double, double> final_boss_clear(int samples, int node_cap);     // the ante-8 oracle (defined after RoundSolver)                 // the final boss of the shop-only game (defined after RoundSolver)
    StepResult step_core(int a) {
        if (autoplay && a == A_LEAVE && g.state == ST_BLIND_SELECT && (unchecked || p.mask[A_LEAVE])) return step_auto();
        if (autoplay && autoplay_force && a == A_SELECT && g.state == ST_BLIND_SELECT) return step_auto();
        if (a < 0 || a >= N_ACTIONS || (!unchecked && !lite && !p.mask[a])) throw std::runtime_error("illegal action");
        int before = g.furthest_blind;
        uint64_t bits_before = g.cleared_bits;
        uint8_t prev = g.state;
        bool verdant = g.boss_active() == B_VERDANT_LEAF && g.f_verdant;
        int vb = std::min(3 * (g.ante - 1) + g.blind_idx, 23);
        g.last_play_score = 0.0;
        apply_action(a);
        last_score = g.last_play_score;
        if (shop_only && a == A_SELECT && g.state == ST_SELECTING_HAND && !(g.ante >= g.win_ante && g.blind_idx == 2)) shop_only_win();
        g.order_jokers();
        school_supply();
        steps += 1;
        if (a == A_REROLL) rerolls += 1;
        if (a >= A_SWAP && a < A_SWAP_END) swaps += 1;
        if (g.state != prev) {
            if (g.state == ST_SHOP && prev != ST_PACK) rerolls = 0;
            bool shop_pack = (prev == ST_SHOP && g.state == ST_PACK) || (prev == ST_PACK && g.state == ST_SHOP);
            if (!shop_pack) swaps = 0;
        }
        if (arch && arch_store > 0.0 && g.state == ST_BLIND_SELECT && prev != ST_BLIND_SELECT && (scn_kind < 0 || scn_kind == 3)) archive_offer();
        double r = 0.0, sp = 0.0;
        if (g.immortal) {                                  // each blind actually beaten pays its weight, once (a skipped-over failed blind pays nothing)
            uint64_t nb = g.cleared_bits & ~bits_before;
            for (int b = 0; b < 48; b++)
                if ((nb >> b) & 1ull) { double w = weights[b] * blind_scale * blind_mult[b]; r += w; if (b >= sparse_from_blind) sp += w; }
        } else if (!shop_only)                            // (the shop-only game: blinds won without play pay nothing; only the final boss counts)
        for (int b = before; b < g.furthest_blind; b++) {
            double w = weights[std::min(b, 47)] * blind_scale;
            r += w;
            if (b >= sparse_from_blind) sp += w;
        }
        const int last_ante_start = 3 * (std::max(1, g.win_ante) - 1);          // (the final ante begins: ante 8 in the normal game)
        if (before < last_ante_start && g.furthest_blind >= last_ante_start) { r += ante8_bonus; sp += ante8_bonus; }
        if (verdant && a >= A_SELL_J && a < A_SELL_C) r += verdant_rel ? verdant_bonus * weights[vb] * blind_scale : verdant_bonus;
        if (g.state == ST_WON) {
            r += win_bonus;
            sp += win_bonus;
        } else if (g.state == ST_GAME_OVER && g.target > 0 && !g.immortal && part_credit) {
            int lost = 3 * (g.ante - 1) + g.blind_idx;
            if (lost + 1 > g.furthest_blind) {
                double frac = log_partial ? (g.chips > 0 ? std::min(1.0, std::log10(1.0 + (double)g.chips) / std::log10(1.0 + (double)g.target)) : 0.0)
                                          : std::min(1.0, g.chips / g.target);
                r += partial_w * weights[std::min(lost, 23)] * blind_scale * frac;
            }
        }
        if (death_pen > 0.0 && g.state == ST_GAME_OVER)
            r -= death_pen * std::min(std::fabs((double)g.money), death_cap) / death_cap;      // debt counts like money
        bool scn_end = scn_limit > 0 && !g.done() && g.furthest_blind >= scn_limit;
        bool trunc = !g.done() && (steps >= MAX_STEPS || scn_end);
        bool done = g.done() || steps >= MAX_STEPS || scn_end;
        if (done) school_note_end();
        if (!done && !lite) {
            encode();
            bool any = false;
            for (int i = 0; i < N_ACTIONS; i++) if (p.mask[i]) { any = true; break; }
            if (!any) { g.state = ST_GAME_OVER; done = true; }
        }
        if (done) last_bits = g.cleared_bits;
        if (done && arch) {
            if (arch_start_blind >= 0) {
                std::lock_guard<std::mutex> lk(arch->mu);
                arch->ends[arch_start_blind]++;
                if (g.state == ST_WON) arch->wins[arch_start_blind]++;
            }
            if (arch_winners) { if (g.state == ST_WON) archive_commit(); arch_pending.clear(); }
        }
        if (shape_chips != 0.0) {
            double ph = done ? 0.0 : potential();
            r += shape_gamma * ph - phi;
            phi = ph;
        }
        StepResult res;
        res.reward = r;
        res.done = done;
        res.sparse = sp;
        res.trunc = trunc && done && !g.done();
        return res;
    }
};

// ------------------------------------------------------------------ many environments
// Auto-reset as vec_env.VecEnv does: environment i plays seeds seed0 + i + n * episode. After step(), the
// observation of a finished environment is the first one of its next episode; rew / done / ante / won /
// blinds describe the step just taken (the finished episode's outcome when done).
// An exact solver for one blind: the cards to play and discard that clear it. The deck order and every roll are known, so a state can be
// copied and a play tried on the copy. Depth-first over the best few plays by predicted score and a few discards (the complement of a
// good play); the clearing lines are kept by the money they leave (leftover hands pay).
// the exact round solver's preference between lines that clear the same blind: ending money, plus ROUND_FEED_W dollars for every "natural round of growth"
// (the growth of a joker's counter in the line divided by its per-round rate in SCN_GROW) the line gives the growing jokers the build holds. 0: money only.
static double ROUND_FEED_W = 0.0;
static bool SOLVER_HAND_BOUND = false;                  // the round solver: once a clearing line with H hands left is found, a branch that has fewer hands left
                                                          // than that cannot end richer (a hand left pays $1) and is not searched; discards are tried before plays
static int SOLVER_EXTRA = -1;
static bool SOLVER_SMART_ORDER = false;
static bool SOLVER_TT = false;                            // the round solver skips a position it has already reached by another order of moves (round_key + money + consumables)                   // the round solver tries discards first where the best play now, made on every hand left, would not reach the target
static bool SOLVER_TAROTS = false;                       // the round solver also tries a held consumable that needs target cards (Chariot, Death, Strength ...), on its best targets
static int SOLVER_TARGETS = 3;                            // ... the target choices tried per consumable (the auto-targeting heuristic's picks first)
static bool SOLVER_LITE = false;                         // the round solver's nodes: a step makes only the candidate plays (no full observation)
static bool SOLVER_CACHE = false;                        // the beam's levels share solved rounds between beam states (round_key), each cached line replayed to check it
static double SOLVER_DEAD_F = 0.0;                       // > 0: the round solver skips a round whose target exceeds hands left x the best play from all cards left x this margin
static std::atomic<long long> SOLVER_CACHE_HITS{0}, SOLVER_CACHE_MISSES{0}, SOLVER_CACHE_FALLBACKS{0};                            // the round solver: >= 0 = stop this many nodes after the first clearing line is found (-1: search on to node_cap)
static int SEARCH_REROLL_CAP = 0;
static bool SEARCH_PACK_MACRO = false;                   // the beam's shop search opens a bought pack at once: every pick (and the skip) is one child of the buy,
                                                          // so a pack does not use up the search's depth and width before its value shows                        // > 0: the beam's shop search tries at most this many rerolls per shop visit
static int SEARCH_PRUNE = 0;                             // the beam's shop search (ShopSearch::run): bit 1 = a swap / sell must raise the ranking score (else the child is pruned);
                                                          // bit 2 = money dominance: of two states with the same future (state_key), only the one with more money is kept
static int LEVEL_SCRIPTED = -1;                          // >= 0: the beam's levels play each blind with the scripted player of this mode (shop-only search: the cards are not searched); -1: the exact round solver
static int SOLVER_NPLAYS = 4, SOLVER_NDISC = 3, SOLVER_FIRST_PLAYS = 0, SOLVER_FIRST_DISC = 0;          // the exact round solver tries this many plays (best by predicted score) and this many discards at each step
struct RoundPlan { std::vector<int> acts; int money = 0; int hands_left = 0; double score = 0.0; };



// ---------------------------------------------------------------- the evaluation network (balatro_rl/nnue.py): float inference in the engine.
// The feature extraction reads the encoded observation exactly as balatro_rl/nnue.py features() does (the column indices and the layout are
// handed over from Python at load time, so both sides use the same numbers).
struct NnueNet {
    int nf = 0, w = 0, h1 = 0, h2 = 0;
    std::vector<float> acc, acc_b, l1, l1_b, l2, l2_b, o, o_b;
    std::vector<int> cols, off;
    int J0 = 0, C0 = 0, NJ = 150, NC = 52, nboss = 28;
    bool ok = false;
};
static NnueNet NNUE_NET;
static double ROUND_EVAL_W = 0.0;          // > 0: the round solver scores a clearing line by its money + this x the blinds the network expects after it

enum { NC_ANTE, NC_BLIND, NC_STATE, NC_MONEY, NC_LOG_MONEY, NC_NEG, NC_LEVELS, NC_PLAYED, NC_RANKS, NC_SUITS, NC_ENH, NC_SEAL, NC_FULL, NC_VOUCH, NC_BOSS,
       NC_ROUND_HANDS, NC_ROUND_DISC, NC_HAND_SIZE, NC_JSLOTS, NC_N };
enum { NO_J_ID, NO_J_ED, NO_J_CNT, NO_J_RENT, NO_J_PERISH, NO_LVL, NO_PLAYED, NO_RANK, NO_SUIT, NO_ENH, NO_SEAL, NO_DECK, NO_VOUCH, NO_MONEY, NO_PHASE,
       NO_BOSS, NO_CONS, NO_HANDS, NO_DISC, NO_HSIZE, NO_JSLOTS, NO_FUT0, NO_FUT1, NO_FUT2, NO_FUT_LATER, NO_N };

static inline int nn_searchsorted_left(const double* edges, int n, double x) { int i = 0; while (i < n && edges[i] < x) i++; return i; }

// the active features of one observation (g, jid, jf with stride fj, cid, fut = the future list [FUT_K x FUT_W] or null). as_next_select: a state
// right after a round, judged as the blind select that follows (the network was trained on blind selects).
static int nnue_features(const float* g, const int64_t* jid, const float* jf, int fj, const int64_t* cid, const float* fut, bool as_next_select, int* out) {
    const NnueNet& N = NNUE_NET;
    const int* C = N.cols.data();
    const int* O = N.off.data();
    int n = 0;
    auto rint_ = [](float x) { return (int)std::nearbyint(x); };
    auto clampi = [](int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); };
    int ante = clampi(rint_(g[C[NC_ANTE]] * 8.0f), 1, 8);
    int blind = 0; for (int b = 1; b < 3; b++) if (g[C[NC_BLIND] + b] > g[C[NC_BLIND] + blind]) blind = b;
    int state = 0; for (int q = 1; q < 4; q++) if (g[C[NC_STATE] + q] > g[C[NC_STATE] + state]) state = q;
    if (as_next_select) { blind += 1; if (blind > 2) { blind = 0; ante = std::min(ante + 1, 8); } state = 0; }
    float mg = g[C[NC_MONEY]];
    double money = mg < 1.99f ? (double)(mg * 50.0f) : (double)(std::pow(10.0f, 3.0f * g[C[NC_LOG_MONEY]]) - 1.0f);
    if (g[C[NC_NEG]] > 0.5f) money = -1.0;
    for (int k = 0; k < MAX_JOKERS; k++) {
        int j = (int)jid[k] - N.J0;
        if (j < 0 || j >= N.NJ) continue;
        const float* f = jf + (size_t)k * fj;
        out[n++] = O[NO_J_ID] + j;
        for (int e = 0; e < 4; e++) if (f[2 + e] > 0.5f) out[n++] = O[NO_J_ED] + j * 4 + e;
        if (f[11] > 0.0f) out[n++] = O[NO_J_CNT] + j * 8 + std::min((int)(f[11] * 8.0f), 7);
        if (f[8] > 0.5f) out[n++] = O[NO_J_RENT] + j;
        if (f[7] > 0.05f) out[n++] = O[NO_J_PERISH] + j;
    }
    static const double PB[6] = {0, 2, 5, 10, 20, 1e9};
    for (int h = 0; h < N_HANDS; h++) {
        int lv = clampi(rint_(g[C[NC_LEVELS] + h] * 10.0f), 1, 10);
        if (lv > 1) out[n++] = O[NO_LVL] + h * 10 + lv - 1;
        float pl = g[C[NC_PLAYED] + h] * 30.0f;
        if (pl > 0.5f) out[n++] = O[NO_PLAYED] + h * 6 + nn_searchsorted_left(PB, 6, (double)pl);
    }
    for (int r = 0; r < 13; r++) out[n++] = O[NO_RANK] + r * 9 + clampi(rint_(g[C[NC_RANKS] + r] * 4.0f), 0, 8);
    for (int q = 0; q < 4; q++) out[n++] = O[NO_SUIT] + q * 12 + clampi(rint_(g[C[NC_SUITS] + q] * 13.0f / 3.0f), 0, 11);
    static const double EB[6] = {0, 1, 2, 4, 8, 1e9};
    for (int e = 0; e < 8; e++) { float x = (float)rint_(g[C[NC_ENH] + e] * 10.0f); if (x > 0) out[n++] = O[NO_ENH] + e * 6 + std::min(nn_searchsorted_left(EB, 6, x), 5); }
    for (int e = 0; e < 4; e++) { float x = (float)rint_(g[C[NC_SEAL] + e] * 10.0f); if (x > 0) out[n++] = O[NO_SEAL] + e * 6 + std::min(nn_searchsorted_left(EB, 6, x), 5); }
    {
        int full = rint_(g[C[NC_FULL]] * 52.0f);
        int d = (int)std::floor((full - 28) / 4.0);
        out[n++] = O[NO_DECK] + clampi(d, 0, 11);
    }
    for (int v = 0; v < 32; v++) if (g[C[NC_VOUCH] + v] > 0.5f) out[n++] = O[NO_VOUCH] + v;
    static const double ME[15] = {0, 2, 4, 7, 11, 15, 19, 24, 29, 39, 49, 74, 99, 149, 249};
    out[n++] = O[NO_MONEY] + std::min(nn_searchsorted_left(ME, 15, money), 15);
    out[n++] = O[NO_PHASE] + ((ante - 1) * 3 + blind) * 4 + state;
    { int bi = 0; float bm = g[C[NC_BOSS]]; for (int b = 1; b < N.nboss; b++) if (g[C[NC_BOSS] + b] > bm) { bm = g[C[NC_BOSS] + b]; bi = b; }
      if (bm > 0.5f) out[n++] = O[NO_BOSS] + bi; }
    for (int q = 0; q < MAX_CONSUMABLES; q++) { int c = (int)cid[q] - N.C0; if (c >= 0 && c < N.NC) out[n++] = O[NO_CONS] + c; }
    out[n++] = O[NO_HANDS] + clampi(rint_(g[C[NC_ROUND_HANDS]] * 5.0f), 0, 6);
    out[n++] = O[NO_DISC] + clampi(rint_(g[C[NC_ROUND_DISC]] * 5.0f), 0, 7);
    out[n++] = O[NO_HSIZE] + clampi(rint_(g[C[NC_HAND_SIZE]] * 8.0f) - 4, 0, 9);
    out[n++] = O[NO_JSLOTS] + clampi(rint_(g[C[NC_JSLOTS]] * 7.0f), 0, 7);
    if (fut) {
        for (int r = 0; r < 3; r++) for (int j = 0; j < N.NJ; j++) if (fut[(size_t)r * FUT_W + j] > 0.5f) out[n++] = O[NO_FUT0 + r] + j;
        for (int j = 0; j < N.NJ; j++) { bool any = false; for (int r = 3; r < FUT_K; r++) if (fut[(size_t)r * FUT_W + j] > 0.5f) any = true; if (any) out[n++] = O[NO_FUT_LATER] + j; }
    }
    return n;
}

static double nnue_forward(const int* feats, int n) {
    const NnueNet& N = NNUE_NET;
    std::vector<float> a(N.acc_b.begin(), N.acc_b.end());
    for (int q = 0; q < n; q++) { const float* col = N.acc.data() + (size_t)feats[q] * N.w; for (int i = 0; i < N.w; i++) a[i] += col[i]; }
    for (float& x : a) x = std::min(1.0f, std::max(0.0f, x));
    std::vector<float> x1(N.h1), x2(N.h2);
    for (int i = 0; i < N.h1; i++) { float s_ = N.l1_b[i]; const float* r = N.l1.data() + (size_t)i * N.w; for (int k = 0; k < N.w; k++) s_ += r[k] * a[k]; x1[i] = std::min(1.0f, std::max(0.0f, s_)); }
    for (int i = 0; i < N.h2; i++) { float s_ = N.l2_b[i]; const float* r = N.l2.data() + (size_t)i * N.h1; for (int k = 0; k < N.h1; k++) s_ += r[k] * x1[k]; x2[i] = std::min(1.0f, std::max(0.0f, s_)); }
    float z = N.o_b[0];
    for (int k = 0; k < N.h2; k++) z += N.o[k] * x2[k];
    return 24.0 / (1.0 + std::exp(-(double)z));                    // the blinds the run is expected to reach
}

struct RoundSolver {
    std::vector<std::unique_ptr<Env>> pool;
    std::unique_ptr<Env> ev;                                 // (scratch for the network's view of a line's end state)
    std::vector<float> fut;                                  // the seed's future list at the root (it barely depends on how the round is played)
    int nodes = 0, node_cap = 400, max_plans = 3, n_plays = 4, n_discards = 3, first_plan_nodes = -1;
    double best_share = 0.0;                                 // the highest (chips scored / target) any explored line ends the round with (>= 1: the blind is cleared)
    std::vector<RoundPlan> plans;
    std::vector<std::pair<int, double>> base_cnt;          // (joker uid, counter) of the growing jokers at the root
    const Env* cfg = nullptr;
    Env& at(int d) {
        while ((int)pool.size() <= d) {
            auto e = std::make_unique<Env>(cfg->deck, cfg->stake, 1.0, 10.0, cfg->win_ante);
            e->copy_config_from(*cfg);
            e->marginal = 0;                                   // the build values are not needed inside a blind
            e->unchecked = true;
            pool.push_back(std::move(e));
        }
        return *pool[d];
    }
    void record(const std::vector<int>& acts, const Env& c, int hands_left) {
        RoundPlan pl;
        if (first_plan_nodes < 0) first_plan_nodes = nodes;
        pl.acts = acts; pl.money = c.g.money; pl.hands_left = hands_left;
        pl.score = (double)c.g.money;
        if (NNUE_NET.ok && ROUND_EVAL_W > 0.0 && ev && c.g.state == ST_SHOP) {
            ev->copy_state_from(c, true);
            int feats[1024];
            int nfe = nnue_features(ev->p.g, ev->p.jid, ev->p.jf, F_JOKER, ev->p.cid, fut.empty() ? nullptr : fut.data(), true, feats);
            pl.score += ROUND_EVAL_W * nnue_forward(feats, nfe);
        }
        if (ROUND_FEED_W > 0.0) {
            double feed = 0.0;
            for (const Joker& j : c.g.jokers) {
                const ScnGrow* sg = grow_entry(j.key);
                if (!sg) continue;
                double b = sg->init;
                for (const auto& u : base_cnt) if (u.first == j.uid) { b = u.second; break; }
                double d = j.st.get(sg->init) - b;
                if (d > 0) feed += d / sg->per_round;
            }
            pl.score += ROUND_FEED_W * feed;
        }
        plans.push_back(pl);
        std::stable_sort(plans.begin(), plans.end(), [](const RoundPlan& a, const RoundPlan& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.acts.size() < b.acts.size();
        });
        if ((int)plans.size() > max_plans) plans.pop_back();
    }
    void dfs(int d, std::vector<int>& acts) {
        Env& e = *pool[d];
        std::vector<int> pl;
        for (int i = 0; i < N_SUB; i++) if (e.plays[i][0] >= 0 && e.play_len[i] > 0 && (SOLVER_LITE || e.p.mask[A_PLAY + i])) pl.push_back(i);
        std::stable_sort(pl.begin(), pl.end(), [&](int a, int b) { return e.pred_sc[a] > e.pred_sc[b]; });
        std::vector<int> cands;
        int np_ = ROUND_FEED_W > 0.0 ? std::max(n_plays, 8) : n_plays;
        if (d == 0 && SOLVER_FIRST_PLAYS > 0) np_ = std::max(np_, SOLVER_FIRST_PLAYS);        // the first step of a round tries more plays: the line that keeps the good cards starts there
        for (int k = 0; k < (int)pl.size() && k < np_; k++) cands.push_back(A_PLAY + pl[k]);
        if (e.g.discards_left > 0) {
            int nh = (int)e.g.hand.size();
            int made = 0;
            const int nd_ = (d == 0 && SOLVER_FIRST_DISC > 0) ? std::max(n_discards, SOLVER_FIRST_DISC) : n_discards;
            for (int k = 0; k < (int)pl.size() && made < nd_; k++) {
                int i = pl[k];
                bool in[32] = {false};
                for (int q = 0; q < e.play_len[i]; q++) in[e.plays[i][q]] = true;
                int comp[16], c = 0;
                for (int q = 0; q < nh && q < 16; q++) if (!in[q]) comp[c++] = q;
                if (c < 1 || c > 5) continue;
                for (int j = 0; j < N_SUB; j++) {
                    if (e.disc_len[j] != c || !(SOLVER_LITE || e.p.mask[A_DISC + j])) continue;
                    bool same = true;
                    for (int q = 0; q < c; q++) if (e.discs[j][q] != comp[q]) { same = false; break; }
                    if (same) { cands.push_back(A_DISC + j); made++; break; }
                }
            }
        }
        if (SOLVER_SMART_ORDER && !pl.empty() && e.g.discards_left > 0 && e.g.target > 0) {
            // (ordering only, nothing is pruned): the greedy line cannot clear -> set-up discards first, best play's discard first
            const double reach = e.g.chips + (double)std::max(1, e.g.hands_left) * e.pred_sc[pl[0]];
            if (reach < e.g.target) std::stable_partition(cands.begin(), cands.end(), [](int a) { return a >= A_DISC && a < A_SELECT; });
        }
        if (SOLVER_TAROTS)
            for (int k = 0; k < (int)e.g.consumables.size() && k < MAX_CONSUMABLES; k++)
                if (CONS_DEFS[e.g.consumables[k].id].need > 0 && (SOLVER_LITE || e.p.mask[A_USE_C + k])) cands.push_back(A_USE_C + k);
        if (SOLVER_HAND_BOUND && !plans.empty() && !(NNUE_NET.ok && ROUND_EVAL_W > 0.0)) {               // a clearing line exists: look for richer ones, discards first (the lines that save hands start with them)
            int best_left = 0;
            for (const RoundPlan& p : plans) best_left = std::max(best_left, p.hands_left);
            if (e.g.hands_left <= best_left) return;               // every play from here leaves no more hands than the best line already found
            std::stable_partition(cands.begin(), cands.end(), [](int a) { return a >= A_DISC && a < A_SELECT; });
        }
        for (int a : cands) {
            if (nodes >= node_cap * 4 || (nodes >= node_cap && !plans.empty())) return;
            if (SOLVER_EXTRA >= 0 && first_plan_nodes >= 0 && nodes >= first_plan_nodes + SOLVER_EXTRA) return;
            Env& c = at(d + 1);
            c.copy_state_from(e, false);
            nodes++;
            int hands_before = e.g.hands_left;
            if (SOLVER_LITE) {
                c.lite = true;
                c.step_core(a);
                c.lite = false;
                if (c.g.state == ST_SELECTING_HAND && !c.g.targeting) c.candidates();
            } else {
                c.step(a);
            }
            if (!fresh_pos(c)) continue;                               // (reached before by another order of moves)
            acts.push_back(a);
            if (c.g.targeting) {                                       // a consumable waits for its targets: the best few target choices, then search on
                use_targets(d + 1, acts);
                acts.pop_back();
                continue;
            }
            uint8_t st = c.g.state;
            if (st != ST_SELECTING_HAND && e.g.target > 0) best_share = std::max(best_share, (e.g.chips + c.last_score) / e.g.target);
            if (st == ST_SELECTING_HAND) dfs(d + 1, acts);
            else if (st == ST_SHOP || st == ST_WON || st == ST_BLIND_SELECT) record(acts, c, hands_before);
            acts.pop_back();
        }
    }
    // pool[d] waits for a consumable's targets: try the SOLVER_TARGETS target sets the auto-targeting heuristic likes best (most of their cards among its picks, larger sets first)
    void use_targets(int d, std::vector<int>& acts) {
        Env& c = *pool[d];
        if (SOLVER_LITE) c.target_candidates();
        const std::vector<int>& cs = c.g.target_cards();
        std::vector<int> autos = c.g.auto_targets(c.g.targeting_cons.id, cs);
        std::vector<std::pair<int, int>> opts;                    // (score, slot)
        for (int i = 0; i < N_SUB; i++) {
            if (c.play_len[i] <= 0) continue;
            int in = 0;
            for (int q = 0; q < c.play_len[i]; q++) in += Game::in_pile(autos, cs[c.plays[i][q]]) ? 1 : 0;
            opts.push_back({in * 8 + c.play_len[i], i});
        }
        std::stable_sort(opts.begin(), opts.end(), [](const std::pair<int, int>& x, const std::pair<int, int>& y) { return x.first > y.first; });
        for (int k = 0; k < (int)opts.size() && k < SOLVER_TARGETS; k++) {
            if (nodes >= node_cap * 4 || (nodes >= node_cap && !plans.empty())) return;
            const int a = A_PLAY + opts[k].second;
            Env& c2 = at(d + 1);
            c2.copy_state_from(c, false);
            nodes++;
            if (SOLVER_LITE) {
                c2.lite = true;
                c2.step_core(a);
                c2.lite = false;
                if (c2.g.state == ST_SELECTING_HAND && !c2.g.targeting) c2.candidates();
            } else {
                c2.step(a);
            }
            if (!fresh_pos(c2)) continue;
            acts.push_back(a);
            if (c2.g.state == ST_SELECTING_HAND && !c2.g.targeting) dfs(d + 1, acts);
            acts.pop_back();
        }
    }
    // root: an environment at the first decision of a blind (SELECTING_HAND)
    std::unordered_set<uint64_t> seen;                       // (SOLVER_TT) positions reached in this solve
    bool fresh_pos(const Env& c) {
        if (!SOLVER_TT || c.g.state != ST_SELECTING_HAND) return true;
        uint64_t k = c.round_key() ^ (0x9e3779b97f4a7c15ull * (uint64_t)(c.g.money + 1000)) ^ ((uint64_t)c.g.consumables.size() << 56);
        return seen.insert(k).second;
    }
    void solve(const Env& root) {
        cfg = &root;
        seen.clear();
        n_plays = SOLVER_NPLAYS; n_discards = SOLVER_NDISC;
        best_share = 0.0;
        plans.clear();
        nodes = 0;
        first_plan_nodes = -1;
        base_cnt.clear();
        for (const Joker& j : root.g.jokers) { const ScnGrow* sg = grow_entry(j.key); if (sg) base_cnt.push_back({j.uid, j.st.get(sg->init)}); }
        Env& r = at(0);
        r.copy_state_from(root, true);
        std::vector<int> acts;
        fut.clear();
        if (NNUE_NET.ok && ROUND_EVAL_W > 0.0) {
            if (!ev) { ev = std::make_unique<Env>(cfg->deck, cfg->stake, 1.0, 10.0, cfg->win_ante); ev->copy_config_from(*cfg); ev->marginal = 0; ev->unchecked = true; }
            if (root.g.exact) { ev->copy_state_from(root, false); ev->encode_future(); fut.assign(ev->priv_fut, ev->priv_fut + FUT_K * FUT_W); }
        }
        if (SOLVER_DEAD_F > 0.0 && r.g.state == ST_SELECTING_HAND && !r.g.targeting && r.g.target > 0) {
            // an impossible round: the best single play from all the cards left (hand and draw pile together, every held-card effect counted
            // for the whole pile), on every hand left, times the margin dead_f, still falls short of the target -> not searched
            Env& t = at(1);
            t.copy_state_from(r, false);
            std::vector<int> all(t.g.hand.begin(), t.g.hand.end());
            all.insert(all.end(), t.g.deck.begin(), t.g.deck.end());
            if ((int)all.size() > MAX_HAND) {                                  // the 16 most useful cards (the candidate generator's own measure, over the whole pile)
                int suit_cnt[4] = {0, 0, 0, 0}, rank_cnt[16] = {0};
                for (int id : all) { const Card& c = t.g.cards[id]; for (int q = 0; q < 4; q++) if (c.has_suit(q)) suit_cnt[q]++; rank_cnt[c.rank]++; }
                std::vector<std::pair<double, int>> u;
                for (int id : all) {
                    const Card& c = t.g.cards[id];
                    double v = t.g.card_value(c);
                    if (!c.is_stone()) v += 3 * suit_cnt[c.suit] + 8 * rank_cnt[c.rank];
                    u.push_back({-v, id});
                }
                std::stable_sort(u.begin(), u.end());
                all.clear();
                for (int k = 0; k < MAX_HAND; k++) all.push_back(u[k].second);
            }
            t.g.hand = all;
            t.g.deck.clear();
            t.g.forced_uid = -1;
            t.candidates();
            double bp = 0.0;
            for (int i = 0; i < N_SUB; i++) if (t.play_len[i] > 0) bp = std::max(bp, t.pred_sc[i]);
            if (r.g.chips + r.g.hands_left * bp * SOLVER_DEAD_F < r.g.target) {
                best_share = std::min(1.0, (r.g.chips + r.g.hands_left * bp) / r.g.target);
                return;
            }
        }
        if (r.g.state == ST_SELECTING_HAND) dfs(0, acts);
    }
};

inline StepResult Env::shop_only_final(StepResult res) {                 // the final boss: the exact round solver's best line (it sees the draws); no line that clears it: the run is lost
    thread_local RoundSolver rs;
    rs.max_plans = 1;
    rs.node_cap = shop_only_nodes;
    rs.solve(*this);
    if (rs.plans.empty()) {                                   // lost: part credit for the best share of the target any explored line reached
        double share = std::min(1.0, rs.best_share);
        if (part_credit && g.target > 0) {
            double chips = share * g.target;
            double frac = log_partial ? (chips > 0 ? std::min(1.0, std::log10(1.0 + chips) / std::log10(1.0 + (double)g.target)) : 0.0) : share;
            res.reward += partial_w * weights[23] * blind_scale * frac;
        }
        g.state = ST_GAME_OVER;
        res.done = true;
        last_bits = g.cleared_bits;
        return res;
    }
    const std::vector<int> plan = rs.plans[0].acts;
    for (int act : plan) {
        if (res.done || g.state != ST_SELECTING_HAND || !p.mask[act]) break;
        StepResult s2 = step_core(act);
        res.reward += s2.reward; res.sparse += s2.sparse; res.done = s2.done; res.trunc = s2.trunc;
    }
    if (!res.done && g.state == ST_SELECTING_HAND) { g.state = ST_GAME_OVER; res.done = true; last_bits = g.cleared_bits; }
    return res;
}

// The ante-8 oracle, for the value and the shaping (train.py --oracle-bonus / --oracle-inputs): this build with its scaling counters moved by their
// natural rates over the rounds left (as projected_power), facing the ante-8 boss (four of the five final bosses in turn: Verdant Leaf is left out,
// its debuff needs a joker sold), the round played by the exact round solver on `samples` redrawn decks
// -> (the share of the samples whose round clears the boss, the mean over the samples of min(1, the best share of the target any line reached))
inline std::pair<double, double> Env::final_boss_clear(int samples, int node_cap) {
    static const int FINALS[4] = {B_VIOLET_VESSEL, B_CRIMSON_HEART, B_AMBER_ACORN, B_CERULEAN_BELL};
    thread_local std::unique_ptr<Env> pt;
    thread_local RoundSolver rs;
    if (!pt) pt = std::make_unique<Env>(deck, stake, 1.0, 10.0, win_ante);
    int n = std::max(1, samples);
    double cleared = 0.0, share = 0.0;
    for (int s = 0; s < n; s++) {
        pt->copy_config_from(*this);
        pt->marginal = 0;
        pt->unchecked = true;
        pt->auto_round = false;
        pt->shop_only = false;
        pt->copy_state_from(*this, false);
        Game& h = pt->g;
        if (h.done()) return {0.0, 0.0};
        int rounds_left = std::max(0, 3 * (h.win_ante - h.ante) + (3 - std::min(h.blind_idx, 3)));
        for (Joker& j : h.jokers)
            for (const ScnGrow& sgr : SCN_GROW)
                if (sgr.key == j.key) {
                    double v = j.st.has_val ? j.st.val : sgr.init;
                    j.st.set(std::min(std::max(v + sgr.per_round * rounds_left, sgr.lo), sgr.hi));
                }
        int done_blinds = std::max(1, 3 * (h.ante - 1) + h.blind_idx);
        double skip_rate = (double)h.blinds_skipped / done_blinds;
        for (Joker& j : h.jokers) {
            if (j.key == J_MADNESS) j.st.set(j.st.get(1.0) + 0.5 * rounds_left * 2.0 / 3.0);
            else if (j.key == J_YORICK) j.st.set(j.st.get(1.0) + rounds_left * 10.0 / 23.0);
        }
        h.blinds_skipped += (int)std::lround(skip_rate * rounds_left);
        h.ante = h.win_ante;
        h.blind_idx = 2;
        h.new_ante();
        h.blind_idx = 2;
        h.boss = FINALS[s % 4];
        h.state = ST_BLIND_SELECT;
        uint64_t salt = 1 + 7919ull * (uint64_t)(s + 1) + 104729ull * (uint64_t)std::max(0, h.furthest_blind) + 31ull * (uint64_t)std::max<long long>(0, h.money);
        if (h.exact) pt->reseed_future(salt); else h.rng.seed_u64(salt);
        pt->encode();
        pt->step_core(A_SELECT);
        if (h.state != ST_SELECTING_HAND) continue;
        rs.max_plans = 1;
        rs.node_cap = node_cap;
        rs.solve(*pt);
        if (!rs.plans.empty()) cleared += 1.0;
        share += std::min(1.0, rs.best_share);
    }
    return {cleared / n, share / n};
}

inline StepResult Env::step_auto() {
    const bool force_was = autoplay_force;
    autoplay_force = false;                          // (the select below is the ordinary one: with force on it would hand the round back here)
    StepResult tot = step(A_SELECT);
    autoplay_force = force_was;
    if (tot.done || g.state != ST_SELECTING_HAND) return tot;
    thread_local RoundSolver rs;
    rs.max_plans = 1;
    rs.node_cap = autoplay_nodes;
    rs.solve(*this);
    if (!rs.plans.empty()) {
        const std::vector<int> plan = rs.plans[0].acts;
        for (int act : plan) {
            if (tot.done || g.state != ST_SELECTING_HAND || !p.mask[act]) break;
            StepResult s = step(act);
            tot.reward += s.reward; tot.sparse += s.sparse; tot.done = s.done; tot.trunc = s.trunc;
        }
    }
    // force: no clearing line was found (or the plan stopped short): the scripted player finishes the round, so the network never plays one
    for (int guard = 0; autoplay_force && guard < 64 && !tot.done && g.state == ST_SELECTING_HAND; guard++) {
        int act = scripted_action();
        if (act < 0 || act >= N_ACTIONS || !p.mask[act]) break;
        StepResult s = step(act);
        tot.reward += s.reward; tot.sparse += s.sparse; tot.done = s.done; tot.trunc = s.trunc;
    }
    return tot;
}

// One level of the game search: from a state at a blind select, every way to get to the next blind select. The blind is skipped (for its
// tag) or played (the exact RoundSolver finds the lines that clear it, each leads to the shop) and the shop is searched layer by layer:
// every legal action is tried on a copy, duplicate states are dropped, the best `sw` go on, and leaving the shop ends a line. The end
// states are ranked by the build's expected best-play score in decades (observation column F_GLOBAL_BASE) plus a little for money.
struct VecEnv;
struct LevelCand { std::vector<int> path; double score = 0.0; uint64_t sig = 0; int slot = -1; std::vector<int> jokers; };
struct LevelOut { bool win = false; std::vector<int> win_path; std::vector<LevelCand> cands; };

struct RoundCache {                                      // solved rounds shared by the threads of one expand_level call
    std::mutex mu;
    std::unordered_map<uint64_t, std::vector<RoundPlan>> map;
    bool get(uint64_t k, std::vector<RoundPlan>& out) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = map.find(k);
        if (it == map.end()) return false;
        out = it->second;
        return true;
    }
    void put(uint64_t k, const std::vector<RoundPlan>& v) {
        std::lock_guard<std::mutex> lk(mu);
        map[k] = v;
    }
};

struct ShopSearch {
    RoundSolver rs;
    RoundCache* cache = nullptr;
    std::vector<std::unique_ptr<Env>> A, B, E, PK;   // two layers of states, the kept end states, and the states inside an opened pack
    std::unique_ptr<Env> tmp, rp;
    double money_w = 0.01, pot_w = 0.0;
    std::vector<int> target;                 // joker ids a goal-directed search wants to own
    double target_w = 0.0;                   // the ranking's bonus for each one owned
    double scale_w = 0.0;                    // > 0: the scaling-aware ranking (survival of the next blind with a third of the growth, the boss with all of it, strength, growth)
    int sw = 32, sl = 7, ep = 8;
    long long nodes = 0;
    const Env* cfg = nullptr;
    std::unique_ptr<Env> make(const Env& c) {
        auto e = std::make_unique<Env>(c.deck, c.stake, 1.0, 10.0, c.win_ante);
        e->copy_config_from(c);
        e->auto_target = c.auto_target;                  // (a tarot's targets chosen at once, as in the root's game)
        e->unchecked = true;
        return e;
    }
    static uint64_t signature(const Env& e) {
        const ObsPtr& p = e.p;
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
        for (int i = 0; i < F_GLOBAL; i++) mix((uint64_t)(int64_t)std::llround((double)p.g[i] * 1000.0));
        for (int i = 0; i < MAX_JOKERS; i++) mix((uint64_t)p.jid[i]);
        for (int i = 0; i < MAX_JOKERS * F_JOKER; i++) mix((uint64_t)(int64_t)std::llround((double)p.jf[i] * 100.0));
        for (int i = 0; i < MAX_CONSUMABLES; i++) mix((uint64_t)p.cid[i]);
        return h;
    }
    double score(const Env& e) const {
        double tb = 0.0;
        if (target_w > 0.0) for (const Joker& j : e.g.jokers) for (int t : target) if (j.key == t) tb += target_w;
        return tb + score_base(e);
    }
    double score_base(const Env& e) const {
        if (scale_w > 0.0) {
            double phi = std::min(6.0, std::max(0.0, 7.0 * (double)e.p.g[F_GLOBAL_BASE]));        // decades of expected best-play score
            double pot = std::min(2.0, std::max(0.0, 3.0 * (double)e.p.g[F_GLOBAL_BASE + 4]));      // decades the growing jokers add over 3 rounds
            int bi = std::min(2, std::max(0, e.g.blind_idx));
            double cover = (double)e.p.g[F_GLOBAL_BASE + 1 + bi];                                     // hands * score / the next blind's target (<= 5)
            double cover_boss = (double)e.p.g[F_GLOBAL_BASE + 3];
            double c1 = cover * std::pow(10.0, pot / 3.0);
            double c2 = cover_boss * std::pow(10.0, pot);
            double money = (double)e.p.g[17] * 50.0;
            return 4.0 * std::min(c1, 1.0) + 2.0 * std::min(c2, 1.5) + phi + scale_w * pot + money_w * money;
        }
        double phi = std::min(5.5, std::max(0.0, 7.0 * (double)e.p.g[F_GLOBAL_BASE]));
        double money = (double)e.p.g[17] * 50.0;                 // column 17: min(money, 100) / 50
        double pot = pot_w > 0.0 ? 7.0 * (double)e.p.g[F_GLOBAL_BASE + 4] : 0.0;
        return phi + money_w * money + pot_w * pot;
    }
    void ensure(const Env& c) {
        if (tmp) return;                                  // (every state of a search has the same configuration)
        cfg = &c;
        A.clear(); B.clear(); E.clear();
        for (int i = 0; i < sw; i++) { A.push_back(make(c)); B.push_back(make(c)); }
        for (int i = 0; i < ep; i++) E.push_back(make(c));
        for (int i = 0; i < 4; i++) PK.push_back(make(c));
        tmp = make(c);
        rp = make(c);
    }
    void run(const Env& root, LevelOut& out, VecEnv& dest, int dest_base);
};

struct VecEnv {
    int n;
    uint64_t seed0;
    std::vector<Env*> envs;
    std::vector<uint64_t> episode;
    // batch buffers
    std::vector<float> b_g, b_cards, b_jf, b_af;
    std::vector<int64_t> b_jid, b_cid, b_aid, b_subs, b_aid2;
    std::vector<float> b_priv;
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
    float* job_sparse = nullptr;      // (optional) the sparse part of the reward
    bool* job_trunc = nullptr;        // (optional) the episode ended by the step limit
    bool* job_done = nullptr;
    int32_t* job_ante = nullptr;
    bool* job_won = nullptr;
    int32_t* job_blinds = nullptr;
    std::string error;

    VecEnv(int n_, int deck, int stake, uint64_t seed0_, int threads_, double ante_weight, double win_bonus,
           int win_ante, double shape_chips, double shape_gamma, double start_xmult = 0.0,
           double start_any = 0.0, double verdant_bonus = 0.0, const std::vector<int>* stakes = nullptr)
        : n(n_), seed0(seed0_) {
        // stakes: one per environment (a batch that mixes stakes), or none: every environment plays `stake`
        if (stakes && !stakes->empty() && (int)stakes->size() != n) throw std::invalid_argument("stakes: one per environment");
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
        b_priv.assign((size_t)n * F_PRIV, 0.0f);
        episode.assign(n, 0);
        for (int i = 0; i < n; i++) {
            int st = (stakes && !stakes->empty()) ? (*stakes)[i] : stake;
            Env* e = new Env(deck, st, ante_weight, win_bonus, win_ante, shape_chips, shape_gamma);
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
            ptr.priv = b_priv.data() + (size_t)i * F_PRIV;
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
    void set_start_any2(double p) { for (Env* e : envs) e->start_any2 = p; }
    void set_blind_bases(const std::vector<long long>& bases) { for (Env* e : envs) e->blind_bases_cfg = bases; }   // from each environment's next game
    void set_start_eternal(bool on) { for (Env* e : envs) e->start_any_eternal = on; }
    void set_catalogue(double p) { for (Env* e : envs) e->catalogue_p = p; }      // from each environment's next game    // of the games given a random joker, the share given a second one
    // head-start shares for the games that begin from now on (the trainer fades them over a run)
    void set_start(double xmult, double any, double money_p = 0.0, int money = 0, double money2_p = 0.0,
                   int money2 = 0, double grow = 0.0) {
        for (Env* e : envs) {
            e->start_grow = grow;
            e->start_xmult = xmult; e->start_any = any;
            e->start_money_p = money_p; e->start_money = money;
            e->start_money2_p = money2_p; e->start_money2 = money2;
        }
    }
    void copy_many(VecEnv& src, const std::vector<int>& si, const std::vector<int>& di) {
        int m = (int)si.size();
        std::atomic<int> next_i(0);
        auto work = [&]() { int i; while ((i = next_i.fetch_add(1)) < m) { envs[di[i]]->copy_state_from(*src.envs[si[i]], true); } };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
    }
    std::vector<std::vector<std::vector<int>>> solve_rounds(const std::vector<int>& idx, int max_plans, int node_cap) {
        int m = (int)idx.size();
        std::vector<std::vector<std::vector<int>>> out(m);
        std::atomic<int> next_i(0);
        std::string err;
        std::mutex err_mu;
        auto work = [&]() {
            try {
                RoundSolver rs;
                rs.max_plans = max_plans; rs.node_cap = node_cap;
                int i;
                while ((i = next_i.fetch_add(1)) < m) {
                    rs.solve(*envs[idx[i]]);
                    for (const RoundPlan& pl : rs.plans) out[i].push_back(pl.acts);
                }
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (err.empty()) err = ex.what();
            }
        };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
        if (!err.empty()) throw std::runtime_error("solve_rounds: " + err);
        return out;
    }
    void set_exact(bool on) {
        for (Env* e : envs) e->exact = on;
    }
    void set_priv(bool on) {
        for (Env* e : envs) { e->priv_on = on; e->priv_shop_valid = false; e->fut_ante = -1; e->encode(); }
    }
    std::vector<LevelOut> expand_level(const std::vector<int>& idx, VecEnv& dest, int ends_per, int sw, int sl, int node_cap,
                                       double money_w, double pot_w, double scale_w = 0.0,
                                       const std::vector<int>& target = {}, double target_w = 0.0) {
        int m = (int)idx.size();
        std::vector<LevelOut> out(m);
        std::atomic<int> next_i(0);
        std::string err;
        std::mutex err_mu;
        RoundCache cache;
        std::atomic<bool> won{false};
        auto work = [&]() {
            try {
                ShopSearch ss;
                ss.cache = &cache;
                ss.sw = sw; ss.sl = sl; ss.ep = ends_per; ss.money_w = money_w; ss.pot_w = pot_w; ss.scale_w = scale_w; ss.target = target; ss.target_w = target_w;
                ss.rs.node_cap = node_cap; ss.rs.max_plans = 3;
                int i;
                while ((i = next_i.fetch_add(1)) < m) {
                    if (won.load()) continue;                         // a beam state has already won this level: the rest are not needed
                    ss.run(*envs[idx[i]], out[i], dest, i * ends_per);
                    if (out[i].win) won = true;
                }
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (err.empty()) err = ex.what();
            }
        };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
        if (!err.empty()) throw std::runtime_error("expand_level: " + err);
        return out;
    }
    void set_sandbox(int level_cap_mode, int pack_restock, const std::vector<double>& bias) {
        for (Env* e : envs) {
            e->level_cap_mode = level_cap_mode;
            e->pack_restock = pack_restock;
            for (int k = 0; k < 5 && k < (int)bias.size(); k++) e->pack_bias[k] = bias[k];
        }
    }
    std::vector<int> targeting_flags(const std::vector<int>& idx) const {   // 1: the game is waiting for a consumable's target cards
        std::vector<int> out;
        for (int i : idx) out.push_back(envs[i]->g.targeting ? 1 : 0);
        return out;
    }
    // per environment (outside a round): Env::final_boss_clear -> [the share of the samples that clear the ante-8 boss, the mean min(1, best share)]
    std::vector<std::vector<double>> final_boss_clear(const std::vector<int>& idx, int samples, int node_cap) {
        int m = (int)idx.size();
        std::vector<std::vector<double>> out(m);
        std::atomic<int> next_i(0);
        auto work = [&]() {
            int i;
            while ((i = next_i.fetch_add(1)) < m) { auto pr = envs[idx[i]]->final_boss_clear(samples, node_cap); out[i] = {pr.first, pr.second}; }
        };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
        return out;
    }
    // per environment (outside a round): the projected share at the boss of each ante 1..8 (0 for an ante already passed) -> [m * 8]
    std::vector<double> power_curve(const std::vector<int>& idx, int samples) {
        int m = (int)idx.size();
        std::vector<double> out((size_t)m * 8, 0.0);
        std::atomic<int> next_i(0);
        auto work = [&]() {
            int i;
            while ((i = next_i.fetch_add(1)) < m) {
                Env* e = envs[idx[i]];
                if (e->g.state == ST_SELECTING_HAND || e->g.state == ST_PACK) continue;          // (outside a round, not in a pack)
                const int a0 = std::max(1, e->g.ante);
                try { for (int a = a0; a <= 8; a++) if (a <= a0 + 2 || a == 8) out[(size_t)i * 8 + (a - 1)] = e->projected_share(a, samples); }   // (this ante, the next two and
                                                                                                                                                  //  the last; the caller interpolates)
                catch (const std::exception&) {}
            }
        };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
        return out;
    }
    // per environment (outside a round): Env::projected_power
    std::vector<double> projected_power(const std::vector<int>& idx, int samples) {
        int m = (int)idx.size();
        std::vector<double> out(m, 0.0);
        std::atomic<int> next_i(0);
        auto work = [&]() { int i; while ((i = next_i.fetch_add(1)) < m) out[i] = envs[idx[i]]->projected_power(samples); };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
        return out;
    }
    // per environment (at the start of a blind, or outside a round for the coming blind): [hands x mean best play / target, hands x max best play / target, mean best play,
    // max best play, this ante's boss: its hands x mean best play / its target]
    std::vector<std::vector<double>> fast_power(const std::vector<int>& idx, int samples) {
        int m = (int)idx.size();
        std::vector<std::vector<double>> out(m);
        std::atomic<int> next_i(0);
        auto work = [&]() {
            int i;
            while ((i = next_i.fetch_add(1)) < m) {
                Env* e = envs[idx[i]];
                auto pr = e->fast_power(samples);
                const bool inr = e->g.state == ST_SELECTING_HAND;
                // outside a round: the coming blind (blind_idx already points at it in the shop and at a blind select) and its hands (1 against The Needle)
                double tg = std::max(1.0, inr ? e->g.target : (double)e->g.blind_target(e->g.blind_idx));
                int hands_next = (e->g.blind_idx == 2 && e->g.boss == B_NEEDLE) ? 1 : e->g.round_hands();
                double H = std::max(1, inr ? e->g.hands_left : hands_next);
                // [4]: against this ante's boss (its target and its hands: 1 against The Needle)
                double tb = std::max(1.0, (double)e->g.blind_target(2));
                double Hb = e->g.boss == B_NEEDLE ? 1.0 : (double)e->g.round_hands();
                out[i] = {H * pr.first / tg, H * pr.second / tg, pr.first, pr.second, Hb * pr.first / tb};
            }
        };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
        return out;
    }
    void set_shop_only(bool on, int nodes = 1000, double partial_w = 0.5) {             // see Env::shop_only, Env::partial_w
        for (Env* e : envs) { e->shop_only = on; e->shop_only_nodes = nodes; e->partial_w = partial_w; }
    }
    void set_auto_target(bool on) { for (Env* e : envs) e->auto_target = on; }   // see Env::auto_target
    void set_rules(bool no_stickers, bool small_blind_pay) {     // from each environment's next game: no joker stickers; the small blind pays at every stake
        for (Env* e : envs) { e->no_stickers = no_stickers; e->small_blind_pay = small_blind_pay; }
    }
    void set_menu(int n) {                                       // the shop shows n jokers (n > 2; else the usual shop): from the next shop of every running game and in every new one
        for (Env* e : envs) { e->menu_slots = n; e->g.menu_slots = n; }
    }
    void set_part_credit(bool on) {
        for (Env* e : envs) e->part_credit = on;
    }
    void set_log_partial(bool on) {
        for (Env* e : envs) e->log_partial = on;
    }
    void set_start_ante(int ante) {
        for (Env* e : envs) e->start_ante = ante;
    }
    void reseed_rng(const std::vector<int>& idx, const std::vector<uint64_t>& salts) {      // (not exact mode) a new future: the game's own generator restarts from the salt; the state stays
        for (size_t i = 0; i < idx.size() && i < salts.size(); i++) { envs[idx[i]]->g.rng.seed_u64(salts[i]); }
    }
    // a given build in the games `ids`, at a blind select of the given ante and blind: the jokers (engine ids) with their counters (NaN: the joker's own), the first `steel_cards` cards of the
    // deck made Steel, the hand levels (empty: as they are), the money (< 0: as it is); the observation, with the exact expected-score columns, is remade.
    void set_build(const std::vector<int>& ids, int ante, int blind, const std::vector<int>& keys, const std::vector<double>& counters, int steel_cards,
                   const std::vector<int>& levels, long long money, const std::vector<int>& editions = {}, int suit = -1, int suit_n = 0) {
        for (int id : ids) {
            Env* e = envs[id];
            e->g.jokers.clear();
            e->g.joker_slots = std::max<int>(5, (int)keys.size());
            for (size_t k = 0; k < keys.size(); k++) {
                Joker j;
                j.key = (int16_t)keys[k];
                j.base_cost = j.d().cost;
                if (k < counters.size() && !std::isnan(counters[k])) j.st.set(counters[k]);
                if (k < editions.size()) j.edition = (uint8_t)editions[k];
                e->g.add_joker(j);
            }
            int n = 0;
            for (int cid : e->g.full_deck) { if (n >= steel_cards) break; e->g.cards[cid].enh = E_STEEL; n++; }
            if (suit >= 0 && suit < 4) {                                       // a deck tilted toward one suit: the first suit_n cards not of it change to it
                int c2 = 0;
                for (int cid : e->g.full_deck) { if (c2 >= suit_n) break; if (e->g.cards[cid].suit != suit) { e->g.cards[cid].suit = (int8_t)suit; c2++; } }
            }
            for (size_t h = 0; h < levels.size() && h < (size_t)N_HANDS; h++) e->g.hand_levels[h] = levels[h];
            if (money >= 0) e->g.money = money;
            e->g.ante = ante;
            e->g.blind_idx = blind;
            e->g.new_ante();
            e->g.state = ST_BLIND_SELECT;
            e->g.furthest_blind = 3 * (ante - 1) + blind;
            e->marg_valid = false;
            e->encode();
        }
    }
    // counterfactual edits for the gap study (tools/gap_study.py levers), on games outside a round:
    //   op 0: hand a (-1: the most played) gains b levels      op 1: money += a       op 2: joker a added (b < 0: appended if a slot is free; else it replaces slot b)
    //   op 3: a cards not of the deck's most common suit change to it          op 4: the a lowest-ranked plain cards are destroyed          op 5: every joker loses its rental sticker
    void edit(const std::vector<int>& ids, int op, int a, int b) {
        for (int id : ids) {
            Game& g = envs[id]->g;
            if (op == 0) {
                int h = a;
                if (h < 0) { h = 0; for (int k = 1; k < N_HANDS; k++) if (g.hand_played[k] > g.hand_played[h]) h = k; }
                g.hand_levels[h] += b;
            } else if (op == 1) {
                g.money += a;
            } else if (op == 2) {
                Joker j;
                j.key = (int16_t)a;
                j.base_cost = j.d().cost;
                if (b >= 0 && b < (int)g.jokers.size()) g.destroy_joker(g.jokers[b].uid);
                if (b >= 0 || (int)g.jokers.size() < g.joker_slots) g.add_joker(j);
            } else if (op == 3) {
                int cnt[4] = {0, 0, 0, 0};
                for (int cid : g.full_deck) if (!g.cards[cid].is_stone()) cnt[g.cards[cid].suit & 3]++;
                int best = 0;
                for (int k = 1; k < 4; k++) if (cnt[k] > cnt[best]) best = k;
                int c2 = 0;
                for (int cid : g.full_deck) { if (c2 >= a) break; if (!g.cards[cid].is_stone() && g.cards[cid].suit != best) { g.cards[cid].suit = (int8_t)best; c2++; } }
            } else if (op == 4) {
                std::vector<int> plain;
                for (int cid : g.full_deck) if (g.cards[cid].enh == E_NONE && g.cards[cid].seal == S_NONE && g.cards[cid].ed == ED_NONE) plain.push_back(cid);
                std::stable_sort(plain.begin(), plain.end(), [&](int x, int y) { return g.cards[x].rank < g.cards[y].rank; });
                if ((int)plain.size() > a) plain.resize(a);
                g.destroy_cards(plain);
            } else if (op == 5) {
                for (Joker& j : g.jokers) j.rental = false;
            }
            envs[id]->marg_valid = false;
            envs[id]->encode();
        }
    }
    void set_modes(const std::vector<int>& modes) {         // each environment's scripted-player mode for its next round
        for (size_t i = 0; i < envs.size() && i < modes.size(); i++) envs[i]->sp_mode = modes[i];
    }
    void set_auto_round(bool on, double pace, int mode = 0) {              // the select action plays the round with the scripted player (mode: see Env::sp_mode)
        for (Env* e : envs) { e->auto_round = on; e->sp_pace = pace; e->sp_mode = mode; }
    }
    std::vector<double> round_report() const {               // per env: cleared, hands played, discards used, money gained, best share of the target, counters before (8), counters after (8)
        std::vector<double> out;
        for (const Env* e : envs) {
            out.push_back(e->rr_clear); out.push_back(e->rr_hands); out.push_back(e->rr_disc); out.push_back(e->rr_money); out.push_back(e->rr_ratio);
            for (int i = 0; i < MAX_JOKERS; i++) out.push_back(e->rr_c0[i]);
            for (int i = 0; i < MAX_JOKERS; i++) out.push_back(e->rr_c1[i]);
        }
        return out;
    }
    std::vector<double> margin_report() const {                 // per env: the margin and lambda at the first decision of the last scripted round (mode 2)
        std::vector<double> out;
        for (const Env* e : envs) { out.push_back(e->rr_margin); out.push_back(e->rr_lambda); }
        return out;
    }
    void set_margin_params(double m_full, int samples, double force_lambda = -1.0) { for (Env* e : envs) { e->sp_mfull = m_full; e->sp_samples = std::max(4, samples); e->sp_force_lambda = force_lambda; } }
    void set_caps(int reroll_cap, int hand_level_cap, int flash_cap = 0) {          // rerolls per shop visit (0: the default), the highest level a planet takes a hand to (0: none)
        for (Env* e : envs) { e->reroll_cap_base = reroll_cap > 0 ? reroll_cap : MAX_REROLLS_PER_SHOP; e->hand_level_cap_cfg = hand_level_cap; e->flash_cap_cfg = flash_cap; }
    }
    void set_blind_mult(const std::vector<double>& m) {      // the reward of blind b (immortal mode) is multiplied by m[b]
        for (Env* e : envs) for (int b = 0; b < 24 && b < (int)m.size(); b++) e->blind_mult[b] = m[b];
    }
    void set_autoplay(bool on, int nodes = 400, bool force = false) {
        for (Env* e : envs) { e->autoplay = on; e->autoplay_nodes = nodes; e->autoplay_force = on && force; e->encode(); }
    }
    void set_immortal(bool on, bool pay = false) {
        for (Env* e : envs) { e->immortal = on; e->immortal_pay = pay; }
    }
    std::vector<uint64_t> last_bits() const {           // the blinds beaten in the last game that ended in each environment
        std::vector<uint64_t> out;
        for (const Env* e : envs) out.push_back(e->last_bits);
        return out;
    }
    // for each environment at the start of a blind: the best share of the target any line the solver explores reaches, whether a clearing line was found, and the nodes used
    std::vector<std::vector<double>> solve_best(const std::vector<int>& idx, int node_cap) {
        int m = (int)idx.size();
        std::vector<std::vector<double>> out(m);
        std::atomic<int> next_i(0);
        auto work = [&]() {
            RoundSolver rs;
            rs.max_plans = 1; rs.node_cap = node_cap;
            int i;
            while ((i = next_i.fetch_add(1)) < m) {
                rs.solve(*envs[idx[i]]);
                out[i] = {rs.best_share, rs.plans.empty() ? 0.0 : 1.0, (double)rs.nodes};
            }
        };
        int nt = std::max(1, std::min(threads > 0 ? threads : 8, m));
        std::vector<std::thread> ts;
        for (int t = 1; t < nt; t++) ts.emplace_back(work);
        work();
        for (std::thread& t : ts) t.join();
        return out;
    }
    void set_round_resources(const std::vector<int>& idx, int hands, int discards) {      // a blind with fewer hands and discards left (small cases for checking the solver)
        for (int i : idx) { Env* e = envs[i]; e->g.hands_left = hands; e->g.discards_left = discards; e->encode(); }
    }
    void set_level_scripted(int mode) { LEVEL_SCRIPTED = mode; }
    void set_search_prune(int bits) { SEARCH_PRUNE = bits; }
    void set_solver_extra(int n) { SOLVER_EXTRA = n; }
    void set_solver_hand_bound(bool on) { SOLVER_HAND_BOUND = on; }
    void set_order_mode(int m) { ORDER_MODE = m; }
    void set_pot_mode(int m, int rounds) { POT_MODE = m; POT_ROUNDS = rounds; for (Env* e : envs) e->marg_valid = false; }
    void set_search_reroll_cap(int n) { SEARCH_REROLL_CAP = n; }
    void set_search_pack_macro(bool on) { SEARCH_PACK_MACRO = on; }
    void load_nnue(const std::string& path, const std::vector<int>& cols, const std::vector<int>& off, const std::vector<int>& consts) {
        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("load_nnue: cannot open " + path);
        int32_t hd[4];
        if (std::fread(hd, 4, 4, fp) != 4) { std::fclose(fp); throw std::runtime_error("load_nnue: bad header"); }
        NnueNet N;
        N.nf = hd[0]; N.w = hd[1]; N.h1 = hd[2]; N.h2 = hd[3];
        auto rd = [&](std::vector<float>& v, size_t k) { v.resize(k); if (std::fread(v.data(), 4, k, fp) != k) throw std::runtime_error("load_nnue: short file"); };
        rd(N.acc, (size_t)N.nf * N.w); rd(N.acc_b, N.w); rd(N.l1, (size_t)N.h1 * N.w); rd(N.l1_b, N.h1); rd(N.l2, (size_t)N.h2 * N.h1); rd(N.l2_b, N.h2);
        rd(N.o, N.h2); rd(N.o_b, 1);
        std::fclose(fp);
        if ((int)cols.size() != NC_N || (int)off.size() != NO_N || consts.size() < 5) throw std::runtime_error("load_nnue: layout size");
        N.cols = cols; N.off = off;
        N.J0 = consts[0]; N.C0 = consts[1]; N.NJ = consts[2]; N.NC = consts[3]; N.nboss = consts[4];
        N.ok = true;
        NNUE_NET = N;
    }
    void set_round_eval(double w) { ROUND_EVAL_W = w; }
    // the network on environments idx as they stand (their encoded observation; the future list from the privileged observation when on)
    std::vector<double> nnue_eval(const std::vector<int>& idx, bool as_next_select) {
        std::vector<double> out;
        int feats[1024];
        for (int i : idx) {
            Env* e = envs[i];
            const float* fut = e->priv_on ? e->p.priv + 200 : nullptr;
            int nfe = nnue_features(e->p.g, e->p.jid, e->p.jf, F_JOKER, e->p.cid, fut, as_next_select, feats);
            out.push_back(NNUE_NET.ok ? nnue_forward(feats, nfe) : 0.0);
        }
        return out;
    }
    std::vector<std::vector<int>> nnue_feature_lists(const std::vector<int>& idx) {
        std::vector<std::vector<int>> out;
        int feats[1024];
        for (int i : idx) {
            Env* e = envs[i];
            const float* fut = e->priv_on ? e->p.priv + 200 : nullptr;
            int nfe = nnue_features(e->p.g, e->p.jid, e->p.jf, F_JOKER, e->p.cid, fut, false, feats);
            out.emplace_back(feats, feats + nfe);
        }
        return out;
    }
    void set_solver_speed(bool lite, bool cache, double dead_f) { SOLVER_LITE = lite; SOLVER_CACHE = cache; SOLVER_DEAD_F = dead_f; }
    void set_solver_tarots(bool on, int targets) { SOLVER_TAROTS = on; SOLVER_TARGETS = std::max(1, targets); }
    void set_solver_order(bool smart) { SOLVER_SMART_ORDER = smart; }
    void set_curve_crn(uint64_t seed, bool generic_boss) { FP_CRN_SEED = seed; CURVE_GENERIC_BOSS = generic_boss; }
    void set_solver_tt(bool on) { SOLVER_TT = on; }
    void hide_future(const std::vector<int>& idx, const std::vector<uint64_t>& salts) {
        for (size_t i = 0; i < idx.size() && i < salts.size(); i++) envs[idx[i]]->hide_future(salts[i]);
    }
    std::vector<long long> solver_cache_stats(bool reset) {
        std::vector<long long> o{SOLVER_CACHE_HITS.load(), SOLVER_CACHE_MISSES.load(), SOLVER_CACHE_FALLBACKS.load()};
        if (reset) { SOLVER_CACHE_HITS = 0; SOLVER_CACHE_MISSES = 0; SOLVER_CACHE_FALLBACKS = 0; }
        return o;
    }
    // per environment: (the state key without the money, the money): see Env::state_key
    std::vector<std::pair<uint64_t, long long>> state_keys(const std::vector<int>& idx) const {
        std::vector<std::pair<uint64_t, long long>> out;
        for (int i : idx) out.push_back({envs[i]->state_key(), envs[i]->g.money});
        return out;
    }
    std::vector<std::string> diff_states(int i, int j) const { return envs[i]->diff_names(*envs[j]); }
    std::vector<std::string> diff_envs(int i, int j) const { return envs[i]->diff_env_names(*envs[j]); }
    // per environment, the numbers a cell key is made from: ante, blind, state, money, vouchers, joker slots, consumables held, then the jokers (key, edition, flags) after their count,
    // the 12 hand levels, and the deck: size, counts per enhancement (12), per seal (8), per edition (8)
    std::vector<std::vector<long long>> cell_info(const std::vector<int>& idx) const {
        std::vector<std::vector<long long>> out;
        for (int i : idx) {
            const Game& g = envs[i]->g;
            std::vector<long long> v = {(long long)g.ante, (long long)g.blind_idx, (long long)g.state, (long long)g.money, (long long)g.vouchers, (long long)g.joker_slots, (long long)g.consumables.size(),
                                        (long long)g.jokers.size()};
            for (const Joker& j : g.jokers) { v.push_back(j.key); v.push_back((long long)j.edition); v.push_back((long long)j.eternal + 2 * (long long)j.rental + 4 * (long long)j.has_perish); }
            for (int h = 0; h < N_HANDS; h++) v.push_back(g.hand_levels[h]);
            long long enh[12] = {0}, seal[8] = {0}, ed[8] = {0};
            for (int id : g.full_deck) { const Card& c = g.cards[id]; enh[(int)c.enh % 12]++; seal[(int)c.seal % 8]++; ed[(int)c.ed % 8]++; }
            v.push_back((long long)g.full_deck.size());
            for (int k = 0; k < 12; k++) v.push_back(enh[k]);
            for (int k = 0; k < 8; k++) v.push_back(seal[k]);
            for (int k = 0; k < 8; k++) v.push_back(ed[k]);
            out.push_back(v);
        }
        return out;
    }
    std::vector<std::vector<int>> hand_levels(const std::vector<int>& idx) const {
        std::vector<std::vector<int>> out;
        for (int i : idx) { std::vector<int> l; for (int h = 0; h < N_HANDS; h++) l.push_back(envs[i]->g.hand_levels[h]); out.push_back(l); }
        return out;
    }
    void set_round_feed(double w) { ROUND_FEED_W = w; }
    void set_solver(int n_plays, int n_discards, int first_plays = 0, int first_discards = 0) { SOLVER_NPLAYS = std::max(1, n_plays); SOLVER_NDISC = std::max(0, n_discards); SOLVER_FIRST_PLAYS = first_plays; SOLVER_FIRST_DISC = first_discards; }
    void set_verdant_relative(bool on) {
        for (Env* e : envs) e->verdant_rel = on;
    }
    void set_power(double lo, double hi) {
        for (Env* e : envs) { e->scn_power_lo = lo; e->scn_power_hi = hi; }
    }
    void set_build_seeds(const std::vector<uint64_t>& seeds, int force_blind, double level_conc) {      // env i's scenario build comes from seeds[i] (0: from the game seed)
        for (size_t i = 0; i < envs.size(); i++) {
            envs[i]->scn_build_seed = i < seeds.size() ? seeds[i] : 0;
            envs[i]->scn_force_blind = force_blind;
            envs[i]->scn_level_conc = level_conc;
        }
    }
    void set_scaled(int start_n, bool shop, int rounds) {
        for (Env* e : envs) { e->scaled_start = start_n; e->scaled_shop = shop; e->scaled_rounds = rounds; }
    }
    void set_scenario(double p, double w_one, double w_few, double w_full, int min_ante, int max_ante, int reroll_cap, double couple) {
        for (Env* e : envs) {
            e->scn_p = p; e->scn_w[0] = w_one; e->scn_w[1] = w_few; e->scn_w[2] = w_full;
            e->scn_min_ante = min_ante; e->scn_max_ante = max_ante; e->scn_reroll_cap = reroll_cap; e->scn_couple = couple;
        }
    }
    std::vector<double> play_scores() const {           // the score of each environment's last step (a play) or 0
        std::vector<double> out;
        for (const Env* e : envs) out.push_back(e->last_score);
        return out;
    }
    std::vector<int> scn_kinds() const {                // the kind of the game each environment is in now (-1: a real game)
        std::vector<int> out;
        for (const Env* e : envs) out.push_back(e->scn_kind);
        return out;
    }
    void set_level_cap(double p, int n, int ante) {
        for (Env* e : envs) { e->level_cap_p = p; e->level_cap_n = n; e->level_cap_ante = ante; }
    }
    void set_ban(const std::vector<int>& jokers, double p) {
        for (Env* e : envs) { e->banned.assign(jokers.begin(), jokers.end()); e->ban_p = p; }
    }
    void set_joker_cap(const std::vector<int>& jokers, int max_held) {    // one group (replacing any); an empty list: no cap
        for (Env* e : envs) { e->cap_set.assign(J_COUNT, 0); e->cap_maxes.clear(); e->cap_max = -1; }
        if (!jokers.empty()) add_joker_cap(jokers, max_held);
    }
    void add_joker_cap(const std::vector<int>& jokers, int max_held) {    // one more group with its own maximum
        for (Env* e : envs) {
            if ((int)e->cap_set.size() != J_COUNT) e->cap_set.assign(J_COUNT, 0);
            e->cap_maxes.push_back(std::max(0, max_held));
            for (int k : jokers) if (k >= 0 && k < J_COUNT) e->cap_set[k] = (uint8_t)e->cap_maxes.size();
            e->cap_max = 0;
        }
    }
    // scaling multiplier jokers handed out at the start, counter pre-grown (the trainer fades both)
    void set_xscale(double share, double val) {
        for (Env* e : envs) { e->start_xscale = share; e->start_xscale_val = val; }
    }
    // the stake of every environment, from its next game on (a curriculum that moves environments between stakes)
    void set_stakes(const std::vector<int>& stakes) {
        if (stakes.size() != envs.size()) throw std::invalid_argument("stakes: one per environment");
        for (size_t i = 0; i < envs.size(); i++) envs[i]->stake = stakes[i];
    }
    void set_school(double p, const std::vector<int>& students) {
        for (Env* e : envs) { e->school_p = p; e->school_ids.assign(students.begin(), students.end()); }
    }
    void set_death_penalty(double pen, double cap) {
        for (Env* e : envs) { e->death_pen = pen; e->death_cap = cap > 1.0 ? cap : 1.0; }
    }
    std::vector<int> school_student() const {
        std::vector<int> out(envs.size());
        for (size_t i = 0; i < envs.size(); i++) out[i] = envs[i]->school_student;
        return out;
    }
    // finished school games since the last call: per joker id, [games, sum of log(final counter)]
    std::vector<double> school_stats() {
        std::vector<double> out((size_t)J_COUNT * 2, 0.0);
        for (Env* e : envs)
            for (int k = 0; k < J_COUNT; k++) {
                out[2 * k] += e->sc_n[k]; out[2 * k + 1] += e->sc_sumlog[k];
                e->sc_n[k] = 0.0; e->sc_sumlog[k] = 0.0;
            }
        return out;
    }
    // a hypothetical state of a game at a blind select: its ante and blind (ante 0: leave them) and each joker's counter (NaN: leave it), all else as it is; the observation (the
    // exact expected-score columns included) is remade. counters: [len(ids) * MAX_JOKERS], row-major.
    void set_state(const std::vector<int>& ids, int ante, int blind, const std::vector<double>& counters) {
        for (size_t k = 0; k < ids.size(); k++) {
            Env* e = envs[ids[k]];
            if (ante > 0) {
                e->g.ante = ante;
                e->g.blind_idx = blind;
                e->g.new_ante();
                e->g.state = ST_BLIND_SELECT;
                e->g.furthest_blind = 3 * (ante - 1) + blind;
            }
            for (int i = 0; i < MAX_JOKERS && i < (int)e->g.jokers.size(); i++) {
                double c = counters.size() > k * MAX_JOKERS + i ? counters[k * MAX_JOKERS + i] : std::nan("");
                if (!std::isnan(c)) e->g.jokers[i].st.set(c);
            }
            e->marg_valid = false;
            e->encode();
        }
    }
    void set_marginal_blind_only(bool on) { for (Env* e : envs) e->marg_blind_only = on; }
    void set_marginal(int level) {
        for (Env* e : envs) { e->marginal = level; e->encode(); }          // (the observations already made get the inputs)
    }
    void set_weight_shape(double ante_weight, double power) {
        for (Env* e : envs) e->set_weights(ante_weight, power);
    }
    void set_weight_geom(double ratio) {
        for (Env* e : envs) e->set_weights_geom(ratio);
    }
    std::shared_ptr<Archive> archive;
    // p_start of the games from now on start from an archived state of the blinds of antes lo..hi; the start of a blind is offered to the archive with
    // chance p_store (cap per stake and blind, reservoir); a blind with fewer than min_fill snapshots is not used
    void set_archive(double p_start, double p_store, int lo_ante, int hi_ante, int cap, int min_fill, bool winners = false) {
        if (!archive) archive = std::make_shared<Archive>();
        archive->cap = cap;
        bool on = p_start > 0.0 || p_store > 0.0;
        for (Env* e : envs) {
            e->arch = on ? archive : nullptr;
            e->arch_p = p_start; e->arch_store = p_store;
            e->arch_lo = 3 * (lo_ante - 1); e->arch_hi = 3 * hi_ante - 1; e->arch_min = min_fill; e->arch_winners = winners;
        }
    }
    void share_archive(VecEnv& other) {                    // this VecEnv's environments keep to and start from the other's archive
        if (!other.archive) other.archive = std::make_shared<Archive>();
        archive = other.archive;
        for (Env* e : envs) e->arch = archive;
    }
    std::vector<int> archive_outcomes(bool reset) {       // [24 games ended, 24 games won], by the blind they started from
        std::vector<int> out(48, 0);
        if (!archive) return out;
        std::lock_guard<std::mutex> lk(archive->mu);
        for (int b = 0; b < 24; b++) {
            out[b] = (int)archive->ends[b]; out[24 + b] = (int)archive->wins[b];
            if (reset) { archive->ends[b] = 0; archive->wins[b] = 0; }
        }
        return out;
    }
    std::vector<int> archive_starts(bool reset) {          // games started from each blind (since the last reset)
        std::vector<int> out(24, 0);
        if (!archive) return out;
        std::lock_guard<std::mutex> lk(archive->mu);
        for (int b = 0; b < 24; b++) { out[b] = (int)archive->starts[b]; if (reset) archive->starts[b] = 0; }
        return out;
    }
    // plant the games `ids` as they are now (each at a blind select) into the archive, at their own stake and blind; a full blind replaces a random
    // snapshot. -> the number planted. (The reverse curriculum's planted builds: train.py --plant-file.)
    int archive_put(const std::vector<int>& ids) {
        if (!archive) archive = std::make_shared<Archive>();
        int n = 0;
        for (int id : ids) {
            if (id < 0 || id >= (int)envs.size()) continue;
            Env* e = envs[id];
            int b = 3 * (e->g.ante - 1) + e->g.blind_idx;
            if (b < 0 || b > 23 || e->g.state != ST_BLIND_SELECT) continue;
            std::lock_guard<std::mutex> lk(archive->mu);
            std::vector<ArchSnap>& v = archive->slot[e->stake][b];
            ArchSnap snap{e->g, e->restricted, e->level_capped_game};
            if ((int)v.size() < archive->cap) v.push_back(snap);
            else v[archive->next() % v.size()] = snap;
            n++;
        }
        return n;
    }
    std::vector<int> archive_counts(int stake) {          // snapshots held per blind (stake < 0: all stakes)
        std::vector<int> out(24, 0);
        if (!archive) return out;
        std::lock_guard<std::mutex> lk(archive->mu);
        for (int s = 0; s < 9; s++)
            for (int b = 0; b < 24; b++)
                if (stake < 0 || s == stake) out[b] += (int)archive->slot[s][b].size();
        return out;
    }
    void set_sparse(int from_blind) {
        for (Env* e : envs) e->sparse_from_blind = from_blind;
    }
    void set_reward(double blind_scale, double ante8_bonus) {
        for (Env* e : envs) { e->blind_scale = blind_scale; e->ante8_bonus = ante8_bonus; }
    }
    void set_tilt(double p, int levels) {
        for (Env* e : envs) { e->tilt_p = p; e->tilt_levels = levels; }
    }
    void restart() {                                    // every environment starts its first game again (after set_exact ...)
        for (int i = 0; i < n; i++) { episode[i] = 0; new_episode(i); }
    }
    long long fixed_seed = -1;                          // >= 0: every game, in every environment, is this seed (PPO on one seed)
    void set_fixed_seed(long long s) { fixed_seed = s; }
    void new_episode(int i) {
        envs[i]->reset(fixed_seed >= 0 ? (uint64_t)fixed_seed : seed0 + (uint64_t)i + (uint64_t)n * episode[i]);
        episode[i] += 1;
    }
    void step_one(int i) {
        if (job_actions[i] < 0) {                       // a negative action leaves the environment as it is (a search's idle copies)
            job_rew[i] = 0.0f; job_done[i] = false; job_ante[i] = envs[i]->g.ante; job_won[i] = false;
            if (job_sparse) job_sparse[i] = 0.0f;
            if (job_trunc) job_trunc[i] = false;
            job_blinds[i] = envs[i]->g.furthest_blind;
            return;
        }
        Env& e = *envs[i];
        StepResult r = e.step((int)job_actions[i]);
        job_rew[i] = (float)r.reward;
        if (job_sparse) job_sparse[i] = (float)r.sparse;
        if (job_trunc) job_trunc[i] = r.trunc;
        job_done[i] = r.done;
        job_ante[i] = e.g.ante;
        job_won[i] = e.g.state == ST_WON;
        job_blinds[i] = e.immortal ? popcount32(e.last_bits) : e.g.furthest_blind;      // (immortal: the blinds beaten, not the furthest)
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
    void step(const int64_t* actions, float* rew, bool* done, int32_t* ante, bool* won, int32_t* blinds, float* sparse = nullptr, bool* trunc = nullptr) {
        job_actions = actions; job_rew = rew; job_done = done; job_ante = ante; job_won = won; job_blinds = blinds;
        job_sparse = sparse; job_trunc = trunc;
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

inline void ShopSearch::run(const Env& root, LevelOut& out, VecEnv& dest, int dest_base) {
    ensure(root);
    out = LevelOut();
    // the kept end states: a bounded set by score, without duplicates
    struct End { double score; uint64_t sig; std::vector<int> path; int slot; uint64_t key = 0; int64_t money = 0; };
    std::vector<End> ends;
    std::vector<uint64_t> end_sigs;
    const bool dom = (SEARCH_PRUNE & 2) != 0;
    auto offer_end = [&](const Env& c, const std::vector<int>& path) {
        uint64_t sg = signature(c);
        for (uint64_t x : end_sigs) if (x == sg) return;
        double sc = score(c);
        uint64_t key = dom ? c.state_key() : 0;
        if (dom) {
            for (End& e : ends) if (e.key == key) {                       // the same future: keep the richer one
                if (c.g.money <= e.money) return;
                E[e.slot]->copy_state_from(c, false);
                for (uint64_t& x : end_sigs) if (x == e.sig) { x = sg; break; }
                e.score = sc; e.sig = sg; e.path = path; e.money = c.g.money;
                return;
            }
        }
        if ((int)ends.size() < ep) {
            int slot = (int)ends.size();
            E[slot]->copy_state_from(c, false);
            ends.push_back({sc, sg, path, slot, key, c.g.money});
            end_sigs.push_back(sg);
        } else {
            int w = 0;
            for (int k = 1; k < (int)ends.size(); k++) if (ends[k].score < ends[w].score) w = k;
            if (sc <= ends[w].score) return;
            E[ends[w].slot]->copy_state_from(c, false);
            for (uint64_t& x : end_sigs) if (x == ends[w].sig) { x = sg; break; }
            ends[w].score = sc; ends[w].sig = sg; ends[w].path = path; ends[w].key = key; ends[w].money = c.g.money;
        }
    };
    // layer 0: the states after the blind is skipped (if its tag opens a pack) and after each clearing line
    struct Ly { std::vector<int> path; int ms = 0, mpk = 0; };      // ms / mpk: the lowest shop / pack slot the next buy may take
    std::vector<Ly> layer;
    int n = 0;
    if (root.g.state == ST_SHOP || root.g.state == ST_PACK) {          // a root inside a shop: the search starts there (no blind to play)
        A[0]->copy_state_from(root, true);
        layer.push_back({std::vector<int>{}});
        n = 1;
    } else {
        tmp->copy_state_from(root, true);
        const bool* mask = tmp->p.mask;
        bool can_skip = mask[A_SKIP], can_select = mask[A_SELECT];
        if (can_skip) {
            Env& c = *rp;
            c.copy_state_from(root, true);
            c.step(A_SKIP);
            nodes++;
            std::vector<int> path{A_SKIP};
            if (c.g.state == ST_BLIND_SELECT) offer_end(c, path);
            else if (c.g.state != ST_GAME_OVER && n < sw) { A[n]->copy_state_from(c, false); layer.push_back({path}); n++; }
        }
        if (can_select && LEVEL_SCRIPTED >= 0) {                          // shop-only search: the scripted player plays the whole blind; in a path the select is -3
            Env& c = *tmp;
            c.copy_state_from(root, true);
            c.auto_round = true; c.sp_mode = LEVEL_SCRIPTED; c.sp_pace = 1.5;
            c.step(A_SELECT);
            nodes++;
            std::vector<int> path{-3};
            if (c.g.state == ST_WON) { out.win = true; out.win_path = path; return; }
            if (c.g.state != ST_GAME_OVER && n < sw) { A[n]->copy_state_from(c, false); layer.push_back({path}); n++; }
        } else if (can_select) {
            Env& c = *tmp;
            c.copy_state_from(root, true);
            c.step(A_SELECT);
            nodes++;
            std::vector<RoundPlan> plans;
            uint64_t rk = 0;
            bool cached = false;
            if (cache && SOLVER_CACHE && c.g.state == ST_SELECTING_HAND) {
                rk = c.round_key();
                cached = cache->get(rk, plans);
                if (cached) {                                                    // a cached line must clear here too, or the round is solved afresh
                    for (const RoundPlan& pl : plans) {
                        Env& r = *rp;
                        r.copy_state_from(c, false);
                        r.lite = true;
                        bool legal = true;
                        size_t used = 0;
                        for (int a : pl.acts) {
                            if (r.g.state != ST_SELECTING_HAND || r.g.targeting) { legal = false; break; }      // the round ended early: not the same line
                            used++;
                            r.candidates();
                            bool ok = (a >= A_PLAY && a < A_DISC) ? r.play_len[a - A_PLAY] > 0 : (a >= A_DISC && a < A_SELECT && r.disc_len[a - A_DISC] > 0 && r.g.discards_left > 0);
                            if (!ok) { legal = false; break; }
                            r.step_core(a);
                        }
                        r.lite = false;
                        uint8_t st = r.g.state;
                        if (!legal || used != pl.acts.size() || !(st == ST_SHOP || st == ST_WON || st == ST_BLIND_SELECT)) { cached = false; break; }
                    }
                    if (cached) SOLVER_CACHE_HITS++; else SOLVER_CACHE_FALLBACKS++;
                } else {
                    SOLVER_CACHE_MISSES++;
                }
            }
            if (!cached) {
                rs.solve(c);
                plans = rs.plans;
                if (cache && SOLVER_CACHE && rk) cache->put(rk, plans);
            }
            for (const RoundPlan& pl : plans) {
                Env& r = *rp;
                r.copy_state_from(c, true);
                std::vector<int> path{A_SELECT};
                bool ok = true;
                for (int a : pl.acts) { if (r.g.state != ST_SELECTING_HAND) { ok = false; break; } r.step(a); path.push_back(a); nodes++; }
                if (!ok) continue;
                if (r.g.state == ST_WON) { out.win = true; out.win_path = path; return; }
                if (r.g.state != ST_GAME_OVER && n < sw) { A[n]->copy_state_from(r, false); layer.push_back({path}); n++; }
            }
        }
    }
    std::vector<std::unique_ptr<Env>>* cur = &A;
    std::vector<std::unique_ptr<Env>>* nxt = &B;
    for (int depth = 0; depth < sl && n > 0; depth++) {
        struct Ch { double score; uint64_t sig; std::vector<int> path; int slot; int ms, mpk; uint64_t key = 0; int64_t money = 0; };
        std::vector<Ch> kept;
        std::vector<uint64_t> seen;
        for (int i = 0; i < n; i++) {
            Env& par = *(*cur)[i];
            par.encode();
            const double par_score = (SEARCH_PRUNE & 1) ? score(par) : 0.0;
            double sell_sc[MAX_JOKERS];                                   // bit 16: the score after selling each joker (the weakest are the ones a replace may give up)
            bool sell_seen[MAX_JOKERS];
            for (int q = 0; q < MAX_JOKERS; q++) { sell_sc[q] = -1e300; sell_seen[q] = false; }
            int weak[2] = {-1, -1};
            bool weak_ready = false;
            std::vector<int> legal;
            for (int a = 0; a < N_ACTIONS; a++) if (par.p.mask[a]) legal.push_back(a);
            for (int a : legal) {
                int cms = 0, cmpk = 0;
                if (a >= A_BUY && a < A_BUY_PACK) {                 // buys in one canonical order: the order of the slots does not matter
                    int sl_ = a - A_BUY;
                    if (sl_ < layer[i].ms) continue;
                    cms = sl_; cmpk = layer[i].mpk;
                } else if (a >= A_BUY_PACK && a < A_VOUCHER) {
                    int pk_ = a - A_BUY_PACK;
                    if (pk_ < layer[i].mpk) continue;
                    cms = layer[i].ms; cmpk = pk_;
                } else if (a >= A_SELL_J && a < A_SELL_C) {         // selling is only tried when the slots are full (it makes room)
                    if ((int)par.g.jokers.size() < par.g.joker_slots) continue;
                } else if (a >= A_SELL_C && a < A_USE_C) {          // bit 8: a consumable is only sold when the consumable slots are full
                    if ((SEARCH_PRUNE & 8) && (int)par.g.consumables.size() < par.g.consumable_slots) continue;
                } else if (a >= A_VOUCHER && a < A_REROLL) {        // bit 32: a voucher only when $5 is left after it
                    int vi = a - A_VOUCHER;
                    if ((SEARCH_PRUNE & 32) && vi < (int)par.g.shop_vouchers.size() && par.g.money - par.g.shop_vouchers[vi].cost < 5) continue;
                } else if (a == A_REROLL) {
                    if (SEARCH_REROLL_CAP > 0 && par.rerolls >= SEARCH_REROLL_CAP) continue;
                } else if (a >= A_SWAP && a < A_SWAP_END) {         // (ORDER_MODE 1 already makes swaps illegal without a copy joker)
                } else if (a >= A_REPLACE && (SEARCH_PRUNE & 16)) {  // bit 16: a replace only gives up one of the two weakest jokers
                    if (!weak_ready) {
                        weak_ready = true;
                        for (int q = 0; q < MAX_JOKERS; q++) {
                            if (!sell_seen[q]) continue;
                            if (weak[0] < 0 || sell_sc[q] > sell_sc[weak[0]]) { weak[1] = weak[0]; weak[0] = q; }
                            else if (weak[1] < 0 || sell_sc[q] > sell_sc[weak[1]]) weak[1] = q;
                        }
                    }
                    int jq = (a - A_REPLACE) % MAX_JOKERS;
                    if (jq != weak[0] && jq != weak[1]) continue;
                }
                // one child state c reached by `path` (the first action a): offered as an end state, or kept for the next depth (false: a win was found)
                auto consider = [&](Env& c, const std::vector<int>& path, bool done) -> bool {
                    if (c.g.state == ST_WON) { out.win = true; out.win_path = path; return false; }
                    if (done) return true;
                    if (c.g.state == ST_BLIND_SELECT) { offer_end(c, path); return true; }
                    uint64_t sg = signature(c);
                    for (uint64_t x : seen) if (x == sg) return true;
                    double sc = score(c);
                    if (a >= A_SELL_J && a < A_SELL_C) { sell_sc[a - A_SELL_J] = sc; sell_seen[a - A_SELL_J] = true; }
                    if ((SEARCH_PRUNE & 1) && ((a >= A_SWAP && a < A_SWAP_END) || (a >= A_SELL_J && a < A_USE_C)) && sc <= par_score + 1e-9) return true;   // a swap or a sell that does not help
                    uint64_t key = dom ? c.state_key() : 0;
                    if (dom) {
                        for (Ch& k : kept) if (k.key == key) {
                            if (c.g.money > k.money) {
                                (*nxt)[k.slot]->copy_state_from(c, false);
                                for (uint64_t& x : seen) if (x == k.sig) { x = sg; break; }
                                k.score = sc; k.sig = sg; k.path = path; k.ms = cms; k.mpk = cmpk; k.money = c.g.money;
                            }
                            return true;
                        }
                    }
                    if ((int)kept.size() < sw) {
                        int slot = (int)kept.size();
                        (*nxt)[slot]->copy_state_from(c, false);
                        kept.push_back({sc, sg, path, slot, cms, cmpk, key, c.g.money});
                        seen.push_back(sg);
                    } else {
                        int w = 0;
                        for (int k = 1; k < (int)kept.size(); k++) if (kept[k].score < kept[w].score) w = k;
                        if (sc <= kept[w].score) return true;
                        (*nxt)[kept[w].slot]->copy_state_from(c, false);
                        for (uint64_t& x : seen) if (x == kept[w].sig) { x = sg; break; }
                        kept[w].score = sc; kept[w].sig = sg; kept[w].path = path; kept[w].ms = cms; kept[w].mpk = cmpk; kept[w].key = key; kept[w].money = c.g.money;
                    }
                    return true;
                };
                // a pack opened by this buy: each pick (and the skip), and a second pick for a mega pack, is a child of the buy itself
                std::function<bool(Env&, const std::vector<int>&, int)> open_pack = [&](Env& base, const std::vector<int>& path, int d) -> bool {
                    std::vector<int> acts;
                    for (int q = A_PICK; q <= A_PSKIP; q++) if (base.p.mask[q]) acts.push_back(q);
                    for (int q : acts) {
                        Env& e2 = *PK[d];
                        e2.copy_state_from(base, false);
                        StepResult r2 = e2.step(q);
                        nodes++;
                        std::vector<int> p2 = path;
                        p2.push_back(q);
                        if (e2.g.targeting) continue;                     // (needs auto_target: a targeting step is not searched here)
                        if (e2.g.state == ST_PACK && !r2.done && d + 1 < (int)PK.size()) { if (!open_pack(e2, p2, d + 1)) return false; }
                        else if (!consider(e2, p2, r2.done)) return false;
                    }
                    return true;
                };
                Env& c = *tmp;
                c.copy_state_from(par, false);
                StepResult r = c.step(a);
                nodes++;
                std::vector<int> path = layer[i].path;
                path.push_back(a);
                if (SEARCH_PACK_MACRO && a >= A_BUY_PACK && a < A_VOUCHER && c.g.state == ST_PACK && !r.done && !c.g.targeting) {
                    if (!open_pack(c, path, 0)) return;
                    continue;
                }
                if (!consider(c, path, r.done)) return;
            }
        }
        std::vector<Ly> nl;
        std::vector<std::unique_ptr<Env>>* tgt = nxt;
        // renumber so that layer[i] matches the pool slot i
        for (int i = 0; i < (int)kept.size(); i++) nl.push_back({kept[i].path, kept[i].ms, kept[i].mpk});
        // kept[i].slot == i by construction (slots are reused in place, and paths follow them)
        layer = nl;
        n = (int)kept.size();
        std::swap(cur, nxt);
        (void)tgt;
    }
    for (int k = 0; k < (int)ends.size(); k++) {
        int slot = dest_base + k;
        dest.envs[slot]->copy_state_from(*E[ends[k].slot], true);
        LevelCand lc;
        lc.path = ends[k].path; lc.score = ends[k].score; lc.sig = ends[k].sig; lc.slot = slot;
        for (const Joker& j : E[ends[k].slot]->g.jokers) lc.jokers.push_back((int)j.key);
        out.cands.push_back(lc);
    }
}

}  // namespace bal
