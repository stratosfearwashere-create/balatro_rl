// The Balatro game engine in C++: a port of sim/game.py, sim/scoring.py, sim/jokers.py, sim/hands.py and
// sim/cards.py, function by function, drawing from a replica of CPython's random.Random (../pyrandom.hpp)
// in the same order. Given the same seed and the same actions it reaches the same states as the Python
// game (tests/test_engine.py compares them step by step, including the generator's state).
//
// Differences in representation only: keys are enums (tables_gen.hpp), cards live in an arena and piles
// hold their ids (a card's id is its uid: creation order, which is all the Python uid is used for), a
// joker's state dict is a small struct with "present" flags, sets are bitmasks.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "../pyrandom.hpp"
#include "tables_gen.hpp"
#include "exact.hpp"

namespace bal {

using balatro::PyRandom;

enum Enh : uint8_t { E_NONE = 0, E_BONUS, E_MULT, E_WILD, E_GLASS, E_STEEL, E_STONE, E_GOLD, E_LUCKY, E_HIDDEN };
enum Edition : uint8_t { ED_NONE = 0, ED_FOIL, ED_HOLO, ED_POLY, ED_NEG };
enum Seal : uint8_t { S_NONE = 0, S_RED, S_BLUE, S_GOLD, S_PURPLE };
enum State : uint8_t { ST_BLIND_SELECT = 0, ST_SELECTING_HAND, ST_SHOP, ST_PACK, ST_GAME_OVER, ST_WON };
enum Hand { HC = 0, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH, FIVE_KIND,
            FLUSH_HOUSE, FLUSH_FIVE, N_HANDS };

static const int MAX_HAND = 16, MAX_JOKERS = 8, MAX_CONSUMABLES = 3, MAX_SHOP = 24, MAX_PACK = 5, MAX_VOUCHERS = 2;
static const int WIN_ANTE = 8;
static const int MAXJ = 64;                 // jokers a plan can hold (Negative jokers grow the slots)

inline bool is_secret_hand(int h) { return h == FIVE_KIND || h == FLUSH_HOUSE || h == FLUSH_FIVE; }

struct Card {
    int8_t rank = 2, suit = 0;
    uint8_t enh = E_NONE, ed = ED_NONE, seal = S_NONE;
    bool deb = false, hidden = false;
    int extra = 0;
    int uid = 0;

    bool is_stone() const { return enh == E_STONE || enh == E_HIDDEN; }
    int chip_value() const {
        if (enh == E_HIDDEN) return 0;
        if (enh == E_STONE) return 50;
        if (rank == 14) return 11;
        return rank < 10 ? rank : 10;
    }
    bool is_face(bool pareidolia = false, bool from_boss = false) const {
        if (is_stone() || (deb && !from_boss)) return false;
        return pareidolia || (rank >= 11 && rank <= 13);
    }
    bool has_suit(int s, bool smeared = false) const {
        if (is_stone()) return false;
        if (enh == E_WILD) return true;
        if (smeared) return (suit % 2) == (s % 2);
        return suit == s;
    }
    bool flush_suit(int s, bool smeared = false) const {
        if (is_stone()) return false;
        if (enh == E_WILD && !deb) return true;
        if (smeared) return (suit % 2) == (s % 2);
        return suit == s;
    }
};

// ------------------------------------------------------------------ hand evaluation (hands.py)
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

inline HandEval evaluate(const Card* const* cs, int n, bool four_fingers, bool shortcut, bool smeared) {
    if (n > 5) throw std::runtime_error("evaluate: more than 5 cards");
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
            const Card* c = cs[normal[j]];
            if (c->enh == E_WILD && !c->deb) wild++; else ns[c->suit]++;
        }
        if (smeared) { ns[0] = ns[2] = ns[0] + ns[2]; ns[1] = ns[3] = ns[1] + ns[3]; }
        int best_s = -1;
        for (int s = 0; s < 4; s++) if (ns[s] + wild >= need) { best_s = s; break; }
        if (best_s >= 0)
            for (int j = 0; j < nn; j++) {
                const Card* c = cs[normal[j]];
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

inline void hand_base(int hand, int level, int& chips, int& mult) {
    int lv = level > 1 ? level : 1;
    chips = HAND_BASE[hand][0] + HAND_BASE[hand][2] * (lv - 1);
    mult = HAND_BASE[hand][1] + HAND_BASE[hand][3] * (lv - 1);
}

// ------------------------------------------------------------------ jokers, consumables, shop items
struct JState {                 // a joker's state dict: "val", "rank", "suit", "hand", "left"
    bool has_val = false;
    double val = 0.0;
    int rank = -1, suit = -1, hand = -1, left = -1;       // -1: the key is absent
    bool empty() const { return !has_val && rank < 0 && suit < 0 && hand < 0 && left < 0; }
    double get(double dflt) const { return has_val ? val : dflt; }
    void set(double v) { has_val = true; val = v; }
};

struct Joker {
    int16_t key = J_JOKER;
    uint8_t edition = ED_NONE;
    bool eternal = false, rental = false, debuffed = false, hidden = false;
    bool has_perish = false;
    int perishable = 0;
    int base_cost = 0, sell_bonus = 0;
    bool has_cost = false;
    int cost = 0;
    JState st;
    int uid = 0;

    const JokerDef& d() const { return JOKER_DEFS[key]; }
    int sell_value() const {
        int paid = has_cost ? cost : base_cost;
        int half = paid / 2;                       // paid >= 0 always
        return (half > 1 ? half : 1) + sell_bonus;
    }
};

struct Cons {
    int8_t id = C_NONE;
    bool negative = false;
    int sell_bonus = 0;
    int kind() const { return CONS_DEFS[id].kind; }
    int sell_value() const { return (kind() != CK_SPECTRAL ? 1 : 2) + sell_bonus; }
};

enum ShopKind : uint8_t { SK_JOKER = 0, SK_TAROT, SK_PLANET, SK_SPECTRAL, SK_VOUCHER, SK_PACK, SK_CARD };

struct ShopItem {
    uint8_t kind = SK_JOKER;
    int key = 0;                  // joker id / consumable id / voucher id / pack index
    int cost = 0;
    Joker joker;
    int card = -1;                // arena id of a playing card on sale
    bool is_ante = false;         // a voucher: it is the ante's own (ante_voucher)
};

enum PackItemKind : uint8_t { PI_JOKER = 0, PI_CONS, PI_CARD };
struct PackItem {
    uint8_t kind = PI_JOKER;
    Joker joker;
    Cons cons;
    int card = -1;
};

// ------------------------------------------------------------------ scoring structures
struct Plan {
    int boss = B_NONE;
    int n = 0;
    struct PJ { int uid; int16_t key; uint8_t edition; int8_t rarity; uint16_t hooks; };
    PJ j[MAXJ];
    bool four_fingers = false, shortcut = false, smeared = false, pareidolia = false, splash = false,
         blackboard = false;
    int mime = 0, baseball = 0;
    int nbefore = 0, ncard = 0, nretrig = 0, nheld = 0;
    uint8_t before[MAXJ], card[MAXJ], retrig[MAXJ], held[MAXJ];
};

enum EvKind : uint8_t { EV_CREATE = 0, EV_DNA, EV_DESTROY };
struct ScoreEvent { uint8_t type; int arg; };          // create: consumable kind | (stream name << 4); dna / destroy: card id
// the game's names for the random streams of the consumables and jokers that effects create (exact-seed mode)
enum Src { SRC_SHO = 0, SRC_8BA, SRC_SIXTH, SRC_SUP, SRC_SEA, SRC_VAG, SRC_CAR, SRC_PRI, SRC_EMP, SRC_HAL, N_SRC };
static const char* const SRC_KEYS[N_SRC] = {"sho", "8ba", "sixth", "sup", "sea", "vag", "car", "pri", "emp", "hal"};

struct Game;

struct ScoreCtx {
    Game* g;
    const Plan* plan;
    PyRandom* rng;                       // null: expected value
    Card* played[5];
    int np = 0;
    Card* held[MAX_HAND + 8];
    int nh = 0;
    int hand = 0, contains = 0;
    int scoring[5];
    int ns = 0;
    double chips = 0.0, mult = 0.0, money = 0.0;
    int level_up = 0;
    // the plan jokers' states (copies; committed on a real play). Raw storage: only the first plan->n are
    // ever written or read, and a context is made for every candidate play of every decision
    alignas(JState) unsigned char js_raw[MAXJ * sizeof(JState)];
    JState* js = reinterpret_cast<JState*>(js_raw);
    int ct[MAXJ];                        // plan index a copier copies (-1: nothing)
    bool ov_has[5] = {false, false, false, false, false};
    uint8_t ov[5] = {0, 0, 0, 0, 0};     // Vampire / Midas Mask enhancement overrides of the played cards
    int hiked[5] = {0, 0, 0, 0, 0};      // Hiker's chips gained in this hand, in a prediction
    bool any_hiked = false;
    double lucky_now = 0.0;
    bool copy = false;
    bool first_hand = false;
    ScoreEvent ev[256];
    int nev = 0;
    alignas(Card) unsigned char dna_raw[8 * sizeof(Card)];
    Card* dna_tmp = reinterpret_cast<Card*>(dna_raw);      // DNA's copies in a prediction
    int ndna = 0;
    ScoreCtx() {}
    ScoreCtx(const ScoreCtx&) = delete;
    ScoreCtx& operator=(const ScoreCtx&) = delete;

    bool real() const { return rng != nullptr; }
    int played_index(const Card* c) const {
        for (int i = 0; i < np; i++) if (played[i] == c) return i;
        return -1;
    }
    uint8_t enh(const Card* c) const {
        int i = played_index(c);
        return (i >= 0 && ov_has[i]) ? ov[i] : c->enh;
    }
    void add_chips(double x) { chips += x; }
    void add_mult(double x) { mult += x; }
    void x_mult(double x) { mult *= x; }
    // a roll on the game's named stream `key` in exact-seed mode (defined after Game), else the plain generator
    inline double roll(const char* key);
    inline int roll_int(const char* key, int lo, int hi);
    // the game rolls for a created consumable only while there is room (the cards plus those pending from this hand)
    inline bool cons_room_for_more() const;
    void x_chance(double x, double p, const char* key = nullptr) {
        if (!rng) mult *= 1 + (x - 1) * p;
        else if (roll(key) < p) mult *= x;
    }
    void money_now(double x) { money += x; }
    void money_chance(double x, double p, const char* key = nullptr) {
        if (!rng) money += x * p;
        else if (roll(key) < p) money += x;
    }
    void event(uint8_t type, int arg) {
        if (!rng) return;
        if (nev >= 256) throw std::runtime_error("too many scoring events");
        ev[nev].type = type; ev[nev].arg = arg; nev++;
    }
};

struct ScoreResult {
    double score = 0.0;
    int hand = 0;
    double money = 0.0;
    int scoring = 0;                 // bitmask over played positions
    ScoreEvent ev[256];
    int nev = 0;
};

// ------------------------------------------------------------------ the game
// the counter of a joker that has been owned for k rounds: init + per_round * k, kept inside [lo, hi] (init is what hook_init gives it)
struct ScnGrow { int16_t key; double init, per_round, lo, hi; };
static const ScnGrow SCN_GROW[] = {
    {J_RIDE_THE_BUS, 0, 2.5, 0, 40}, {J_GREEN_JOKER, 0, 2, 0, 60}, {J_RUNNER, 0, 10, 0, 400}, {J_SQUARE, 0, 8, 0, 300},
    {J_RED_CARD, 0, 3, 0, 60}, {J_TROUSERS, 0, 2, 0, 60}, {J_FLASH, 0, 6, 0, 120}, {J_WEE, 0, 8, 0, 300},
    {J_CASTLE, 0, 5, 0, 200}, {J_CONSTELLATION, 1, 0.12, 1, 6}, {J_OBELISK, 1, 0.25, 1, 8}, {J_VAMPIRE, 1, 0.15, 1, 6},
    {J_HOLOGRAM, 1, 0.2, 1, 8}, {J_LUCKY_CAT, 1, 0.15, 1, 6}, {J_CAMPFIRE, 1, 0.3, 1, 5}, {J_GLASS, 1, 0.2, 1, 6},
    {J_HIT_THE_ROAD, 1, 0.3, 1, 5}, {J_CAINO, 1, 0.3, 1, 8}, {J_ICE_CREAM, 100, -12, 5, 100}, {J_POPCORN, 20, -4, 4, 20},
    {J_RAMEN, 2.0, -0.1, 1.05, 2.0}, {J_SELZER, 10, -2, 1, 10}, {J_TURTLE_BEAN, 5, -1, 1, 5}};
// scaled jokers (the late-game sandbox, Env::scaled_start / scaled_shop): a growing joker's counter after `rounds` rounds of growth, with no ceiling
inline bool scaled_counter(int key, int rounds, double& val) {
    for (const ScnGrow& sg : SCN_GROW)
        if (sg.key == key && sg.per_round > 0) { val = sg.init + sg.per_round * rounds; return true; }
    return false;
}

inline const ScnGrow* grow_entry(int key) {
    for (const ScnGrow& sg : SCN_GROW) if (sg.key == key && sg.per_round > 0) return &sg;
    return nullptr;
}

// the jokers' order: 0 = as before (swaps when a joker acts on a neighbour: Blueprint, Brainstorm, Ceremonial Dagger; else +mult before xMult);
// 1 = swaps only with Blueprint, Brainstorm or Ceremonial Dagger, every other build kept in chips / neutral, +mult, xMult order
static int ORDER_MODE = 0;

struct Game {
    // configuration
    int win_ante = WIN_ANTE;
    bool has_pool = false;
    bool pool[J_COUNT];
    PyRandom rng;
    int deck_type = D_RED;
    int stake = 7;
    // run state
    int ante = 1, blind_idx = 0;
    uint8_t state = ST_BLIND_SELECT;
    long long money = 4;
    std::deque<Card> cards;                 // the arena: a card's id is its index (and its uid)
    std::vector<int> full_deck, deck, hand, discard_pile, pack_hand;
    int starting_deck_size = 52;
    std::vector<Joker> jokers;
    std::vector<Joker> graveyard;           // jokers destroyed since the current play began (scoring.Plan refs)
    std::vector<Cons> consumables;
    int joker_slots = 5, consumable_slots = 2, hand_size = 8, base_hands = 4, base_discards = 3;
    int hand_levels[N_HANDS], hand_played[N_HANDS], hand_played_round[N_HANDS];
    uint32_t vouchers = 0;
    std::vector<int8_t> pending_tags;
    // flags
    bool f_verdant = false, f_gros_extinct = false, f_seen_soul = false, f_seen_black_hole = false;
    int f_voucher_bought_ante = -1;         // -1: never
    int f_ecto = 0;
    int f_trading_destroy = -1;             // card id
    uint32_t planets_used = 0;
    int planet_uses = 0;                 // planet cards used so far (every use, unlike the bitmask above)
    int planet_level_cap = -1;           // the late-game sandbox: planet level-ups allowed in all (-1: no limit); after it a planet is still used (Constellation grows) but raises no hand level
    int planet_level_gains = 0;
    int flash_cap = 0;                   // > 0: Flash Card grows from only the first flash_cap rerolls of the game
    int rerolls_total = 0;
    int hand_level_cap = 0;              // > 0: a planet (or Black Hole) raises no hand past this level

    int pack_restock = 0;                // the sandbox: buying a pack puts a new one in its place, this many times per shop visit
    int pack_restocks_used = 0;
    double pack_w_mult[5] = {1, 1, 1, 1, 1};   // the sandbox: weights of the pack kinds (arcana, celestial, standard, buffoon, spectral) are multiplied by these
    int tarots_used = 0;
    int last_consumable = C_NONE;
    int blinds_skipped = 0, blinds_beaten = 0, furthest_blind = 0, total_hands_played = 0, last_hand = 0,
        total_unused_discards = 0;
    uint32_t used_bosses = 0;
    // exact-seed mode: the shop, packs, bosses, tags and deck shuffles come from the game's named streams
    bool exact = false, first_shop_buffoon = false;
    Streams sg;
    int boss_counts[B_COUNT] = {};
    // a probability roll on the game's named stream `key` in exact-seed mode, else the plain generator
    double roll(const char* key) { return exact ? sg.pseudorandom(std::string(key)) : rng.random(); }
    // pseudorandom_element: the index into the list of the element drawn on stream `key`; sortkey[i] orders the elements (the creation
    // id for cards and jokers, the position for anything else)
    int xpick(const char* key, const std::vector<int>& sortkey) {
        int n = (int)sortkey.size();
        std::vector<int> ord(n);
        for (int i = 0; i < n; i++) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return sortkey[a] < sortkey[b]; });
        return ord[sg.pseudorandom(std::string(key), 1, n) - 1];
    }
    // a card id out of `ids` (hand, pack hand ...)
    int pick_card(const char* key, const std::vector<int>& ids) { return exact ? ids[xpick(key, ids)] : ids[rng.choice_index(ids.size())]; }
    // an index into `jokers` out of the candidate indices `cand`
    int pick_joker(const char* key, const std::vector<int>& cand) {
        if (!exact) return cand[rng.choice_index(cand.size())];
        std::vector<int> keys;
        for (int i : cand) keys.push_back(jokers[i].uid);
        return cand[xpick(key, keys)];
    }
    uint8_t edition_poll(const char* key) {            // the guaranteed edition of Aura / Wheel of Fortune (no Negative)
        static const uint8_t trio3[3] = {ED_FOIL, ED_HOLO, ED_POLY};
        static const std::vector<double> w3 = {50, 35, 15};
        if (exact) return exact_poll_edition(key, 1.0, true, true);
        return trio3[rng.choices_indices(w3, 1)[0]];
    }
    // the game's pairs() order of G.GAME.hands (the order lists of "visible hands" come in)
    std::vector<int> visible_hands(int exclude = -1) const {
        static const int ORDER[12] = {FLUSH_HOUSE, FULL_HOUSE, FLUSH, PAIR, HC, STRAIGHT_FLUSH, STRAIGHT, TWO_PAIR, FLUSH_FIVE,
                                      FIVE_KIND, TRIPS, QUADS};
        std::vector<int> out;
        for (int h : ORDER) if ((!is_secret_hand(h) || hand_played[h] > 0) && h != exclude) out.push_back(h);
        return out;
    }
    int todo_pick(int exclude) {
        std::vector<int> v = visible_hands(exclude);
        return v[sg.pseudorandom("to_do", 1, (int)v.size()) - 1];
    }
    // the round targets of The Idol, Mail-In Rebate, Ancient Joker and Castle: one value for the run, redrawn every round
    int tgt_idol_rank = 14, tgt_idol_suit = 0, tgt_mail = 14, tgt_ancient = -1, tgt_castle = 0;
    int orbital_choice[3] = {0, 0, 0};
    void sync_target_state(Joker& j) {
        if (j.key == J_IDOL) { j.st.rank = tgt_idol_rank; j.st.suit = tgt_idol_suit; }
        else if (j.key == J_MAIL) j.st.rank = tgt_mail;
        else if (j.key == J_ANCIENT) j.st.suit = tgt_ancient;
        else if (j.key == J_CASTLE) j.st.suit = tgt_castle;
    }
    void reset_round_targets() {
        std::string a = std::to_string(ante);
        std::vector<int> valid;
        for (int id : full_deck) if (!cards[id].is_stone()) valid.push_back(id);
        std::stable_sort(valid.begin(), valid.end());
        tgt_idol_rank = 14; tgt_idol_suit = 0; tgt_mail = 14;
        if (!valid.empty()) {
            const Card& c = cards[valid[xpick(("idol" + a).c_str(), valid)]];
            tgt_idol_rank = c.rank; tgt_idol_suit = c.suit;
            tgt_mail = cards[valid[xpick(("mail" + a).c_str(), valid)]].rank;
        }
        std::vector<int> suits;
        for (int su = 0; su < 4; su++) if (su != tgt_ancient) suits.push_back(su);
        tgt_ancient = suits[xpick(("anc" + a).c_str(), suits)];
        tgt_castle = valid.empty() ? 0 : cards[valid[xpick(("cas" + a).c_str(), valid)]].suit;
        for (Joker& j : jokers) sync_target_state(j);
    }
    void draw_orbital_choices() {
        for (int t = 0; t < 3; t++) {
            std::vector<int> v = visible_hands();
            orbital_choice[t] = v[sg.pseudorandom("orbital", 1, (int)v.size()) - 1];
        }
    }
    int crimson_pick() {
        if (!exact) return jokers[rng.choice_index(jokers.size())].uid;
        std::vector<int> cand;
        for (size_t i = 0; i < jokers.size(); i++)
            if ((!jokers[i].debuffed && jokers[i].uid != crimson_disabled) || jokers.size() < 2) cand.push_back((int)i);
        if (cand.empty()) return -1;
        return jokers[pick_joker("crimson_heart", cand)].uid;
    }
    // round state
    double chips = 0.0, target = 0.0;
    int hands_left = 0, discards_left = 0, discards_used_round = 0;
    int crimson_disabled = -1, forced_uid = -1;
    std::vector<char> ante_played;          // per card id: played this ante (The Pillar)
    uint32_t round_hand_types = 0;
    int mouth_hand = -1;
    bool boss_disabled = false;
    int juggle = 0;
    int hand_size_override = 0;
    bool targeting = false;
    Cons targeting_cons;
    bool targeting_from_pack = false;
    bool first_draw_done = false;
    int boss_rerolled_ante = 0;
    int ox_hand = -1;
    // shop state
    std::vector<ShopItem> shop, shop_packs, shop_vouchers;
    bool has_ante_voucher = false;
    ShopItem ante_voucher;
    int reroll_cost = 5, free_rerolls = 0, shops_seen = 0;
    // pack state
    std::vector<PackItem> pack_cards;
    int pack_kind = PK_NONE, pack_picks = 0;
    uint8_t pack_return = ST_SHOP;
    // blinds
    int boss = B_NONE;
    int tags_offered[2] = {0, 0};
    int next_joker_uid = 1;
    bool scaled_shop = false;                            // a growing joker made for the shop or a pack arrives with its counter grown (scaled_rounds rounds)
    int scaled_rounds = 30;
    bool immortal_pay = false;                           // with immortal: a failed blind still pays its cash-out (reward money, interest, joker income ...), it just does not count as beaten
    bool no_stickers = false;                            // the stake's joker stickers (eternal, perishable, rental) are never applied (the draws are still made: the other streams are unchanged)
    bool small_blind_pay = false;                        // the small blind pays its $3 at every stake (the Red stake rule that removes it is off)
    bool catalogue = false;                              // the catalogue shop (training): every shop holds every eligible non-legendary joker, MAX_SHOP to a page; a reroll turns the page for free
    std::vector<int16_t> cat_order;                      // (this shop's catalogue, shuffled)
    int cat_page = 0;
    std::vector<long long> blind_bases;                  // (training curricula) each ante's base chips, replacing the stake's table: a longer game with slower scaling; empty: the stake's
    int menu_slots = 0;                                  // > 2: the shop shows this many jokers (and only jokers) instead of its usual two cards (a training curriculum: far more to choose from); 0: the usual shop
    bool immortal = false;                               // a blind that is not beaten does not end the run: no pay, the next blind and its shop (only the ante-8 boss ends it)
    uint64_t cleared_bits = 0;                           // the blinds (0..47) that were actually beaten
    double last_play_score = 0.0;                        // the chips the last play scored (the aux head's target)

    Game(uint64_t seed = 0, int deck = D_RED, int stake_ = 7, int win_ante_ = WIN_ANTE,
         const std::vector<int>* joker_pool = nullptr, const char* exact_seed = nullptr)
        : rng(seed) {
        if (exact_seed && *exact_seed) { exact = true; sg.init(exact_seed); }
        win_ante = win_ante_;
        deck_type = deck;
        stake = stake_;
        if (joker_pool && !joker_pool->empty()) {
            has_pool = true;
            for (int i = 0; i < J_COUNT; i++) pool[i] = false;
            for (int k : *joker_pool) pool[k] = true;
        }
        money = 4 + (deck == D_YELLOW ? 10 : 0);
        if (exact) exact_deck(); else standard_deck();
        if (exact) { this->deck = full_deck; sg.shuffle(this->deck, "shuffle"); }     // the game's first shuffle, before any blind
        starting_deck_size = (int)full_deck.size();
        joker_slots = 5 + (deck == D_BLACK ? 1 : 0) - (deck == D_PAINTED ? 1 : 0);
        consumable_slots = 2;
        hand_size = 8 + (deck == D_PAINTED ? 2 : 0);
        base_hands = 4 + (deck == D_BLUE ? 1 : 0) - (deck == D_BLACK ? 1 : 0);
        base_discards = 3 + (deck == D_RED ? 1 : 0) - (stake >= 4 ? 1 : 0);
        for (int i = 0; i < N_HANDS; i++) { hand_levels[i] = 1; hand_played[i] = 0; hand_played_round[i] = 0; }
        new_ante();
        if (exact) reset_round_targets();
        if (deck == D_MAGIC) {
            vouchers |= 1u << V_CRYSTAL_BALL;
            consumable_slots += 1;
            consumables.push_back(make_cons(C_FOOL));
            consumables.push_back(make_cons(C_FOOL));
        } else if (deck == D_NEBULA) {
            vouchers |= 1u << V_TELESCOPE;
            consumable_slots -= 1;
        } else if (deck == D_ZODIAC) {
            vouchers |= (1u << V_TAROT_MERCHANT) | (1u << V_PLANET_MERCHANT) | (1u << V_OVERSTOCK_NORM);
        } else if (deck == D_GHOST) {
            consumables.push_back(make_cons(C_HEX));
        }
    }

    // ---------------------------------------------------------------- cards
    static Cons make_cons(int id, bool negative = false) {
        Cons c; c.id = (int8_t)id; c.negative = negative; c.sell_bonus = 0; return c;
    }
    int new_card(int rank, int suit, uint8_t enh = E_NONE, uint8_t ed = ED_NONE, uint8_t seal = S_NONE,
                 int extra = 0, bool deb = false) {
        Card c;
        c.rank = (int8_t)rank; c.suit = (int8_t)suit; c.enh = enh; c.ed = ed; c.seal = seal; c.extra = extra;
        c.deb = deb; c.hidden = false;
        c.uid = (int)cards.size();
        cards.push_back(c);
        ante_played.push_back(0);
        return c.uid;
    }
    int copy_card(int id) {                              // Card.copy(): a new uid, not hidden
        const Card c = cards[id];
        return new_card(c.rank, c.suit, c.enh, c.ed, c.seal, c.extra, c.deb);
    }
    void standard_deck() {
        if (deck_type == D_ERRATIC) {
            for (int i = 0; i < 52; i++) {
                int r = (int)rng.randrange(2, 15);
                int s = (int)rng.randrange(4);
                full_deck.push_back(new_card(r, s));
            }
            return;
        }
        for (int s = 0; s < 4; s++)
            for (int r = 2; r < 15; r++) {
                if (deck_type == D_ABANDONED && (r == 11 || r == 12 || r == 13)) continue;
                int suit = s;
                if (deck_type == D_CHECKERED) suit = (s == 0 || s == 2) ? 0 : 1;
                full_deck.push_back(new_card(r, suit));
            }
    }
    void exact_deck() {                                  // creation order of the game: C D H S, ranks 2-9 A J K Q T
        static const int R[13] = {2, 3, 4, 5, 6, 7, 8, 9, 14, 11, 13, 12, 10};
        static const int S[4] = {2, 3, 1, 0};
        if (deck_type == D_ERRATIC) { standard_deck(); return; }
        for (int si = 0; si < 4; si++)
            for (int ri = 0; ri < 13; ri++) {
                int r = R[ri], s0 = S[si];
                if (deck_type == D_ABANDONED && (r == 11 || r == 12 || r == 13)) continue;
                int suit = s0;
                if (deck_type == D_CHECKERED) suit = (s0 == 0 || s0 == 2) ? 0 : 1;
                full_deck.push_back(new_card(r, suit));
            }
    }
    static bool in_pile(const std::vector<int>& pile, int id) {
        return std::find(pile.begin(), pile.end(), id) != pile.end();
    }
    static void remove_from(std::vector<int>& pile, int id) {
        auto it = std::find(pile.begin(), pile.end(), id);
        if (it != pile.end()) pile.erase(it);
    }
    void sort_hand(std::vector<int>& h) const {         // rank desc (stones as 0), then suit, then uid
        std::stable_sort(h.begin(), h.end(), [this](int a, int b) {
            const Card& x = cards[a];
            const Card& y = cards[b];
            int rx = x.is_stone() ? 0 : x.rank, ry = y.is_stone() ? 0 : y.rank;
            if (rx != ry) return rx > ry;
            if (x.suit != y.suit) return x.suit < y.suit;
            return x.uid < y.uid;
        });
    }

    // ---------------------------------------------------------------- helpers
    bool has_voucher(int v) const { return (vouchers >> v) & 1u; }
    bool has(int key) const {
        for (const Joker& j : jokers) if (j.key == key && !j.debuffed) return true;
        return false;
    }
    int count(int key) const {
        int n = 0;
        for (const Joker& j : jokers) if (j.key == key && !j.debuffed) n++;
        return n;
    }
    int find_joker(int uid) const {
        for (size_t i = 0; i < jokers.size(); i++) if (jokers[i].uid == uid) return (int)i;
        return -1;
    }
    const Joker* joker_by_uid(int uid) const {
        int i = find_joker(uid);
        if (i >= 0) return &jokers[i];
        for (const Joker& j : graveyard) if (j.uid == uid) return &j;
        return nullptr;
    }
    // The joker a Blueprint / Brainstorm at index i copies: its index, or -1 (jokers._copy_target)
    int copy_target(int i) const {
        int crimson = boss_active() == B_CRIMSON_HEART ? crimson_disabled : -1;
        int seen[MAXJ];
        int nseen = 0;
        int cur = i;
        while (true) {
            for (int k = 0; k < nseen; k++) if (seen[k] == jokers[cur].uid) return -1;
            if (nseen < MAXJ) seen[nseen++] = jokers[cur].uid;
            int t;
            if (jokers[cur].key == J_BLUEPRINT) {
                if (cur + 1 >= (int)jokers.size()) return -1;
                t = cur + 1;
            } else {
                t = 0;
            }
            if (t == cur || jokers[t].debuffed || jokers[t].uid == crimson) return -1;
            if (jokers[t].key != J_BLUEPRINT && jokers[t].key != J_BRAINSTORM)
                return jokers[t].d().copyable ? t : -1;
            cur = t;
        }
    }
    int copies(int i) const {                           // the key copied, or -1
        if (jokers[i].key != J_BLUEPRINT && jokers[i].key != J_BRAINSTORM) return -1;
        int t = copy_target(i);
        return t >= 0 ? jokers[t].key : -1;
    }
    int count_with_copies(int key) const {
        int n = 0;
        for (size_t i = 0; i < jokers.size(); i++)
            if (!jokers[i].debuffed && (jokers[i].key == key || copies((int)i) == key)) n++;
        return n;
    }
    bool showman() const { return has(J_RING_MASTER); }
    double prob(int n, int d) const {
        double v = (double)n * (double)(1 << count(J_OOPS)) / (double)d;
        return v < 1.0 ? v : 1.0;
    }
    int debt_limit() const { return has(J_CREDIT_CARD) ? -20 : 0; }
    bool can_afford(long long cost) const { return money - cost >= debt_limit(); }
    int cons_used() const {
        int n = 0;
        for (const Cons& c : consumables) if (!c.negative) n++;
        return n;
    }
    bool cons_room() const { return cons_used() < consumable_slots; }
    void create_consumable(int kind, bool negative = false, int name = C_NONE, int src = 0) {
        if (negative || cons_room()) {
            Cons c = name == C_NONE ? random_consumable(kind, 0ull, src) : make_cons(name);
            c.negative = negative;
            consumables.push_back(c);
        }
    }
    bool done() const { return state == ST_GAME_OVER || state == ST_WON; }
    int scaling() const { return stake >= 5 ? 3 : (stake >= 2 ? 2 : 1); }
    long long blind_target(int idx) const {
        int a = ante < 1 ? 1 : (ante > 8 ? 8 : ante);
        long long base = BLIND_BASE[scaling() - 1][a - 1];
        if (!blind_bases.empty()) base = blind_bases[(size_t)std::min<int>(std::max(1, ante), (int)blind_bases.size()) - 1];
        double mult = idx == 0 ? 1.0 : (idx == 1 ? 1.5 : (boss != B_NONE ? (double)BOSS_DEFS[boss].mult : 2.0));
        if (deck_type == D_PLASMA) mult *= 2;
        if (stake >= 8) mult *= 1.25;                    // Diamond (not in the game): 25% more chips per blind
        return (long long)(base * mult);
    }
    int boss_active() const {
        return (blind_idx == 2 && state == ST_SELECTING_HAND && !boss_disabled) ? boss : B_NONE;
    }
    int random_deck_rank() {
        std::vector<int> normal;
        for (int id : full_deck) if (!cards[id].is_stone()) normal.push_back(id);
        if (normal.empty()) return 14;
        return cards[normal[rng.choice_index(normal.size())]].rank;
    }
    void random_deck_card(int& rank, int& suit) {
        std::vector<int> normal;
        for (int id : full_deck) if (!cards[id].is_stone()) normal.push_back(id);
        if (normal.empty()) { rank = 14; suit = 0; return; }
        const Card& c = cards[normal[rng.choice_index(normal.size())]];
        rank = c.rank; suit = c.suit;
    }
    int interest_cap() const { return has_voucher(V_MONEY_TREE) ? 20 : (has_voucher(V_SEED_MONEY) ? 10 : 5); }
    double discount() const { return has_voucher(V_LIQUIDATION) ? 0.5 : (has_voucher(V_CLEARANCE_SALE) ? 0.25 : 0.0); }
    int price(int base) const {
        int p = (int)((base + 0.5) * (1 - discount()));
        return p > 1 ? p : 1;
    }
    int effective_hand_size() const {
        if (hand_size_override) return hand_size_override;
        double hs = hand_size + (has_voucher(V_PAINT_BRUSH) ? 1 : 0) + (has_voucher(V_PALETTE) ? 1 : 0)
                    + count(J_JUGGLER) + 2 * count(J_TROUBADOUR) - 2 * count(J_STUNTMAN) - count(J_MERRY_ANDY) + juggle;
        double bean = 0.0;
        for (const Joker& j : jokers) if (j.key == J_TURTLE_BEAN && !j.debuffed) bean += j.st.get(0);
        hs += bean;
        hs -= (boss_active() == B_MANACLE ? 1 : 0);
        int v = (int)hs;
        if (v > MAX_HAND) v = MAX_HAND;
        if (v < 1) v = 1;
        return v;
    }
    int round_hands() const {
        int v = base_hands + (has_voucher(V_GRABBER) ? 1 : 0) + (has_voucher(V_NACHO_TONG) ? 1 : 0)
                - (has_voucher(V_HIEROGLYPH) ? 1 : 0) - count(J_TROUBADOUR);
        return v > 1 ? v : 1;
    }
    int round_discards() const {
        int v = base_discards + count(J_DRUNKARD) + 3 * count(J_MERRY_ANDY) + (has_voucher(V_WASTEFUL) ? 1 : 0)
                + (has_voucher(V_RECYCLOMANCY) ? 1 : 0) - (has_voucher(V_PETROGLYPH) ? 1 : 0);
        return v > 0 ? v : 0;
    }

    // ---------------------------------------------------------------- antes / blinds
    int exact_boss() {                                   // get_new_boss(): least-used bosses, key order
        int a = ante < 1 ? 1 : ante;
        int cand[N_POOL_BOSS], n = 0, low = 1 << 30;
        for (int i = 0; i < N_POOL_BOSS; i++) {
            const BossPool& b = POOL_BOSS[i];
            bool ok = (!b.showdown && b.min_ante <= a && ((a % 8 != 0) || ante < 2)) || (b.showdown && ante % 8 == 0 && ante >= 2);
            if (!ok) continue;
            cand[n++] = i;
            low = std::min(low, boss_counts[b.boss]);
        }
        int m = 0;
        for (int i = 0; i < n; i++) if (boss_counts[POOL_BOSS[cand[i]].boss] <= low) cand[m++] = cand[i];
        int boss_id = POOL_BOSS[cand[sg.pseudorandom("boss", 1, m) - 1]].boss;
        boss_counts[boss_id] += 1;
        return boss_id;
    }
    int exact_tag() {                                    // get_next_tag_key()
        int pool[N_POOL_TAG], n = N_POOL_TAG;
        for (int i = 0; i < n; i++) pool[i] = POOL_TAG[i].min_ante <= ante ? POOL_TAG[i].tag : -1;
        std::string key = "Tag" + std::to_string(ante);
        int v = pool[sg.pseudorandom(key, 1, n) - 1];
        for (int it = 2; v < 0; it++) v = pool[sg.pseudorandom(key + "_resample" + std::to_string(it), 1, n) - 1];
        return v;
    }
    void new_ante() {
        if (exact) {
            boss = exact_boss();
            used_bosses |= 1u << boss;
            tags_offered[0] = random_tag();
            tags_offered[1] = random_tag();
            draw_orbital_choices();
            has_ante_voucher = false;
            shop_vouchers.clear();
            std::fill(ante_played.begin(), ante_played.end(), 0);
            return;
        }
        std::vector<int> pl;
        if (ante >= WIN_ANTE) {
            for (int b = N_NORMAL_BOSSES; b < B_COUNT; b++) pl.push_back(b);
        } else {
            for (int b = 0; b < N_NORMAL_BOSSES; b++)
                if (BOSS_DEFS[b].min_ante <= ante && !((used_bosses >> b) & 1u)) pl.push_back(b);
            if (pl.empty()) {
                for (int b = 0; b < N_NORMAL_BOSSES; b++) used_bosses &= ~(1u << b);
                for (int b = 0; b < N_NORMAL_BOSSES; b++) if (BOSS_DEFS[b].min_ante <= ante) pl.push_back(b);
            }
        }
        boss = pl[rng.choice_index(pl.size())];
        used_bosses |= 1u << boss;
        tags_offered[0] = random_tag();
        tags_offered[1] = random_tag();
        has_ante_voucher = false;
        shop_vouchers.clear();
        std::fill(ante_played.begin(), ante_played.end(), 0);
    }
    int random_tag() {
        if (exact) return exact_tag();
        int pl[T_COUNT], n = 0;
        for (int t = 0; t < T_COUNT; t++) if (TAG_MIN_ANTE[t] <= ante) pl[n++] = t;
        return pl[rng.choice_index(n)];
    }
    bool can_reroll_boss() const {
        if (state != ST_BLIND_SELECT || !can_afford(10)) return false;
        return has_voucher(V_RETCON) || (has_voucher(V_DIRECTORS_CUT) && boss_rerolled_ante != ante);
    }
    void reroll_boss() {
        if (!can_reroll_boss()) throw std::runtime_error("reroll_boss: not allowed");
        money -= 10;
        boss_rerolled_ante = ante;
        if (exact) { boss = exact_boss(); return; }
        std::vector<int> pl;
        if (ante >= WIN_ANTE) { for (int b = N_NORMAL_BOSSES; b < B_COUNT; b++) pl.push_back(b); }
        else { for (int b = 0; b < N_NORMAL_BOSSES; b++) if (BOSS_DEFS[b].min_ante <= ante) pl.push_back(b); }
        std::vector<int> other;
        for (int b : pl) if (b != boss) other.push_back(b);
        if (!other.empty()) pl = other;
        boss = pl[rng.choice_index(pl.size())];
    }
    void disable_boss() {
        if (boss_disabled) return;
        int b = blind_idx == 2 ? boss : B_NONE;
        bool in_round = state == ST_SELECTING_HAND;
        boss_disabled = true;
        f_verdant = false;
        if (in_round && b == B_WATER) discards_left = std::max(discards_left, round_discards() - discards_used_round);
        if (in_round && b == B_NEEDLE) {
            int played = 0;
            for (int i = 0; i < N_HANDS; i++) played += hand_played_round[i];
            hands_left = std::max(hands_left, round_hands() - played);
        }
        if (in_round && (b == B_WALL || b == B_VIOLET_VESSEL)) target = target / (b == B_WALL ? 2 : 3);
        if (in_round && b == B_MANACLE && first_draw_done) draw(1);
        forced_uid = -1;
        for (int id : full_deck) { cards[id].deb = false; cards[id].hidden = false; }
        for (Joker& j : jokers) j.hidden = false;
    }
    void select_blind() {
        if (state != ST_BLIND_SELECT) throw std::runtime_error("select_blind: wrong state");
        state = ST_SELECTING_HAND;
        boss_disabled = false;
        target = (double)blind_target(blind_idx);
        chips = 0;
        int b = boss_active();
        hands_left = round_hands();
        discards_left = round_discards();
        if (b == B_NEEDLE) hands_left = 1;
        if (b == B_WATER) discards_left = 0;
        discards_used_round = 0;
        for (int i = 0; i < N_HANDS; i++) hand_played_round[i] = 0;
        round_hand_types = 0;
        mouth_hand = -1;
        first_draw_done = false;
        if (b == B_AMBER_ACORN) {
            if (exact) {
                if (jokers.size() > 1) {                    // three shuffles, each starting from the creation order
                    std::vector<Joker> base = jokers, shuffled;
                    std::stable_sort(base.begin(), base.end(), [](const Joker& a, const Joker& c) { return a.uid < c.uid; });
                    for (int r = 0; r < 3; r++) { shuffled = base; sg.shuffle(shuffled, "aajk"); }
                    jokers = shuffled;
                }
            } else {
                rng.shuffle(jokers);
            }
            for (Joker& j : jokers) j.hidden = true;
        }
        if (b == B_CRIMSON_HEART && !jokers.empty()) {
            crimson_disabled = -1;
            crimson_disabled = crimson_pick();
        }
        if (b == B_VERDANT_LEAF) f_verdant = true;
        {
            std::vector<int> snap;
            for (const Joker& j : jokers) snap.push_back(j.uid);
            for (int uid : snap) {
                int i = find_joker(uid);
                if (i < 0) continue;
                if ((jokers[i].d().hooks & H_BLIND_SELECT) && !jokers[i].debuffed) hook_blind_select(uid, uid);
            }
        }
        if (b != B_NONE && has(J_CHICOT)) {
            disable_boss();
            b = B_NONE;
        }
        if (b == B_OX) {
            int best = 0;
            for (int h = 1; h < N_HANDS; h++) if (hand_played[h] >= hand_played[best]) best = h;
            ox_hand = best;
        }
        for (int id : full_deck) { cards[id].deb = false; cards[id].hidden = false; }
        apply_debuffs();
        deck = full_deck;
        if (exact) {                                     // the cards in creation order, shuffled with the 'nr' + ante stream
            std::stable_sort(deck.begin(), deck.end());
            sg.shuffle(deck, "nr" + std::to_string(ante));
        } else {
            rng.shuffle(deck);
        }
        hand.clear();
        discard_pile.clear();
        draw();
        first_draw_done = true;
        int certs = count_with_copies(J_CERTIFICATE);
        for (int k = 0; k < certs; k++) {
            if (exact) {
                int fk = sg.pseudorandom("cert_fr", 1, 52) - 1;
                double t = sg.pseudorandom("certsl");
                uint8_t seal = t > 0.75 ? S_RED : t > 0.5 ? S_BLUE : t > 0.25 ? S_GOLD : S_PURPLE;
                add_card(new_card(P_CARD_RANK[fk], P_CARD_SUIT[fk], E_NONE, ED_NONE, seal), true);
                continue;
            }
            int r = (int)rng.randrange(2, 15);
            int s = (int)rng.randrange(4);
            static const uint8_t seals[4] = {S_RED, S_BLUE, S_GOLD, S_PURPLE};
            uint8_t seal = seals[rng.choice_index(4)];
            add_card(new_card(r, s, E_NONE, ED_NONE, seal), true);
        }
    }
    void apply_debuffs() {
        int b = boss_active();
        bool par = has(J_PAREIDOLIA), smeared = has(J_SMEARED);
        int bsuit = b != B_NONE ? BOSS_DEFS[b].suit : -1;
        for (int id : full_deck) {
            Card& c = cards[id];
            bool d = false;
            if (bsuit >= 0 && c.has_suit(bsuit, smeared)) d = true;
            if (b == B_PLANT && c.is_face(par, true)) d = true;
            if (b == B_PILLAR && ante_played[id]) d = true;
            if (b == B_VERDANT_LEAF && f_verdant) d = true;
            c.deb = d;
        }
    }
    void skip_blind() {
        if (!(state == ST_BLIND_SELECT && blind_idx < 2)) throw std::runtime_error("skip_blind: not allowed");
        int tag = tags_offered[blind_idx];
        blinds_skipped += 1;
        blind_idx += 1;
        add_tag(tag);
    }
    void add_tag(int tag) {
        if (tag == T_DOUBLE) { pending_tags.push_back(T_DOUBLE); return; }
        int n = 1;
        std::vector<int8_t> rest;
        for (int8_t t : pending_tags) { if (t == T_DOUBLE) n++; else rest.push_back(t); }
        pending_tags = rest;
        for (int k = 0; k < n; k++) apply_tag(tag);
    }
    void apply_tag(int tag) {
        if (tag == T_ECONOMY) {
            money += std::min<long long>(40, std::max<long long>(0, money));
        } else if (tag == T_SPEED) {
            money += 5 * blinds_skipped;
        } else if (tag == T_HANDY) {
            money += total_hands_played;
        } else if (tag == T_GARBAGE) {
            money += total_unused_discards;
        } else if (tag == T_TOP_UP) {
            for (int k = 0; k < 2; k++)
                if ((int)jokers.size() < joker_slots) {
                    Joker j;
                    random_joker(j, 1, false, nullptr, false, "top");
                    add_joker(j);
                }
        } else if (tag == T_ORBITAL) {
            int opts[N_HANDS], n = 0;
            for (int i = 0; i < N_HANDS; i++) if (!is_secret_hand(i) || hand_played[i]) opts[n++] = i;
            if (exact) hand_levels[orbital_choice[std::max(0, std::min(2, blind_idx - 1))]] += 3;
            else hand_levels[opts[rng.choice_index(n)]] += 3;
        } else if (tag == T_JUGGLE) {
            pending_tags.push_back(T_JUGGLE);
        } else if (tag == T_BOSS) {
            std::vector<int> pl;
            if (ante < WIN_ANTE) {
                for (int b = 0; b < N_NORMAL_BOSSES; b++) if (BOSS_DEFS[b].min_ante <= ante && b != boss) pl.push_back(b);
            } else {
                for (int b = N_NORMAL_BOSSES; b < B_COUNT; b++) if (b != boss) pl.push_back(b);
            }
            boss = exact ? exact_boss() : pl[rng.choice_index(pl.size())];
        } else if (tag == T_BUFFOON || tag == T_CHARM || tag == T_METEOR || tag == T_STANDARD || tag == T_ETHEREAL) {
            int kind = tag == T_BUFFOON ? PK_BUFFOON : tag == T_CHARM ? PK_ARCANA : tag == T_METEOR ? PK_CELESTIAL
                       : tag == T_STANDARD ? PK_STANDARD : PK_SPECTRAL;
            int size = tag == T_ETHEREAL ? PS_NORMAL : PS_MEGA;
            open_pack(kind, size, ST_BLIND_SELECT);
        } else {
            pending_tags.push_back((int8_t)tag);
        }
    }

    // ---------------------------------------------------------------- drawing
    void draw(int n = -1000000, bool after_play = false) {
        int tgt = effective_hand_size();
        if (n == -1000000) n = tgt - (int)hand.size();
        n = std::min(n, MAX_HAND - (int)hand.size());
        int b = boss_active();
        bool par = has(J_PAREIDOLIA);
        for (int k = 0; k < std::max(0, n); k++) {
            if (deck.empty()) break;
            int id = deck.back();
            deck.pop_back();
            Card& c = cards[id];
            if (b == B_HOUSE && !first_draw_done) c.hidden = true;
            else if (b == B_WHEEL && roll("wheel") < prob(1, 7)) c.hidden = true;
            else if (b == B_FISH && after_play) c.hidden = true;
            else if (b == B_MARK && c.is_face(par, true)) c.hidden = true;
            hand.push_back(id);
        }
        sort_hand(hand);
        if (boss_active() == B_CERULEAN_BELL && !hand.empty()) {
            bool found = false;
            for (int id : hand) if (cards[id].uid == forced_uid) found = true;
            if (!found) forced_uid = cards[pick_card("cerulean_bell", hand)].uid;
        }
    }

    // ---------------------------------------------------------------- playing
    int forced_pos() const {
        if (boss_active() != B_CERULEAN_BELL) return -1;
        for (size_t i = 0; i < hand.size(); i++) if (cards[hand[i]].uid == forced_uid) return (int)i;
        return -1;
    }
    bool violates_boss(const Card* const* played, int np) const {
        int b = boss_active();
        if (b == B_PSYCHIC && np < 5) return true;
        if (b == B_EYE || b == B_MOUTH) {
            int h = evaluate(played, np, has(J_FOUR_FINGERS), has(J_SHORTCUT), has(J_SMEARED)).hand;
            if (b == B_EYE && ((round_hand_types >> h) & 1u)) return true;
            if (b == B_MOUTH && mouth_hand >= 0 && h != mouth_hand) return true;
        }
        return false;
    }
    void make_plan(Plan& p) const {
        int b = boss_active();
        p.boss = b;
        p.n = 0;
        p.four_fingers = p.shortcut = p.smeared = p.pareidolia = p.splash = p.blackboard = false;
        p.mime = p.baseball = 0;
        p.nbefore = p.ncard = p.nretrig = p.nheld = 0;
        for (size_t i = 0; i < jokers.size(); i++) {
            const Joker& j = jokers[i];
            if (j.debuffed || (b == B_CRIMSON_HEART && j.uid == crimson_disabled)) continue;
            if (p.n >= MAXJ) throw std::runtime_error("too many jokers");
            int k = p.n++;
            p.j[k].uid = j.uid; p.j[k].key = j.key; p.j[k].edition = j.edition;
            p.j[k].rarity = j.d().rarity; p.j[k].hooks = j.d().hooks;
            switch (j.key) {
                case J_FOUR_FINGERS: p.four_fingers = true; break;
                case J_SHORTCUT: p.shortcut = true; break;
                case J_SMEARED: p.smeared = true; break;
                case J_PAREIDOLIA: p.pareidolia = true; break;
                case J_SPLASH: p.splash = true; break;
                case J_BLACKBOARD: p.blackboard = true; break;
                default: break;
            }
            int cp = copies((int)i);
            if (j.key == J_MIME || cp == J_MIME) p.mime++;
            if (j.key == J_BASEBALL || cp == J_BASEBALL) p.baseball++;
            uint16_t h = j.d().hooks;
            if (h & H_BEFORE) p.before[p.nbefore++] = (uint8_t)k;
            if (h & H_CARD) p.card[p.ncard++] = (uint8_t)k;
            if (h & H_RETRIG) p.retrig[p.nretrig++] = (uint8_t)k;
            if (h & H_HELD) p.held[p.nheld++] = (uint8_t)k;
        }
    }

    // the scoring hooks (jokers.py); pi: index into plan.j, js: that joker's state copy
    int plan_index(const ScoreCtx& c, int uid) const {
        for (int k = 0; k < c.plan->n; k++) if (c.plan->j[k].uid == uid) return k;
        return -1;
    }
    void resolve_copiers(ScoreCtx& c) const {
        for (int k = 0; k < c.plan->n; k++) {
            c.ct[k] = -1;
            int key = c.plan->j[k].key;
            if (key != J_BLUEPRINT && key != J_BRAINSTORM) continue;
            int i = find_joker(c.plan->j[k].uid);
            if (i < 0) continue;
            int t = copy_target(i);
            if (t >= 0) c.ct[k] = plan_index(c, jokers[t].uid);
        }
    }
    static double jget(const JState& s, double d) { return s.has_val ? s.val : d; }

    void hook_before(ScoreCtx& c, int pi) {
        JState& st = c.js[pi];
        switch (c.plan->j[pi].key) {
            case J_RIDE_THE_BUS: {
                bool face = false;
                for (int k = 0; k < c.ns; k++) if (c.played[c.scoring[k]]->is_face(c.plan->pareidolia)) face = true;
                if (face) st.set(0); else st.set(jget(st, 0) + 1);
                break;
            }
            case J_GREEN_JOKER: st.set(jget(st, 0) + 1); break;
            case J_RUNNER: if (c.contains & (1 << STRAIGHT)) st.set(jget(st, 0) + 15); break;
            case J_SQUARE: if (c.np == 4) st.set(jget(st, 0) + 4); break;
            case J_TODO_LIST: if (c.hand == (st.hand >= 0 ? st.hand : 1)) c.money_now(4); break;
            case J_SPACE: if (c.rng && c.roll("space") < prob(1, 4)) c.level_up += 1; break;
            case J_TROUSERS: if (c.contains & (1 << TWO_PAIR)) st.set(jget(st, 0) + 2); break;
            case J_LOYALTY_CARD: { double v = jget(st, 6); st.set(v > 0 ? v - 1 : 5); break; }
            case J_OBELISK: {
                int mx = 0;
                for (int i = 0; i < N_HANDS; i++) mx = std::max(mx, hand_played[i]);
                if (hand_played[c.hand] == mx) st.set(1.0); else st.set(jget(st, 1.0) + 0.2);
                break;
            }
            case J_DNA: {
                if (c.first_hand && c.np == 1) {
                    Card* src = c.played[0];
                    Card* cp;
                    if (c.rng) {
                        int id = copy_card(src->uid);
                        cp = &cards[id];
                        cp->enh = c.enh(src);
                        c.event(EV_DNA, id);
                    } else {
                        if (c.ndna >= 8) throw std::runtime_error("too many DNA copies");
                        cp = new (&c.dna_tmp[c.ndna++]) Card(*src);
                        cp->hidden = false;
                        cp->uid = -1 - c.ndna;
                        cp->enh = c.enh(src);
                    }
                    c.held[c.nh++] = cp;
                    for (int k = 0; k < c.plan->n; k++)
                        if (c.plan->j[k].key == J_HOLOGRAM) c.js[k].set(jget(c.js[k], 1.0) + 0.25);
                }
                break;
            }
            case J_SIXTH_SENSE: {
                Card* p0 = c.played[0];
                if (c.first_hand && c.np == 1 && !p0->is_stone() && p0->rank == 6) {
                    c.event(EV_DESTROY, p0->uid);
                    c.event(EV_CREATE, CK_SPECTRAL | (SRC_SIXTH << 4));
                }
                break;
            }
            case J_SUPERPOSITION: {
                if (c.contains & (1 << STRAIGHT)) {
                    bool ace = false;
                    for (int k = 0; k < c.ns; k++) {
                        const Card* x = c.played[c.scoring[k]];
                        if (!x->is_stone() && x->rank == 14) ace = true;
                    }
                    if (ace) c.event(EV_CREATE, CK_TAROT | (SRC_SUP << 4));
                }
                break;
            }
            case J_SEANCE: if (c.contains & (1 << STRAIGHT_FLUSH)) c.event(EV_CREATE, CK_SPECTRAL | (SRC_SEA << 4)); break;
            case J_VAMPIRE: {
                int n = 0;
                for (int k = 0; k < c.ns; k++) {
                    int idx = c.scoring[k];
                    Card* x = c.played[idx];
                    if (c.enh(x) != E_NONE && !x->deb) { c.ov_has[idx] = true; c.ov[idx] = E_NONE; n++; }
                }
                st.set(jget(st, 1.0) + 0.1 * n);
                break;
            }
            case J_VAGABOND: if (money <= 4) c.event(EV_CREATE, CK_TAROT | (SRC_VAG << 4)); break;
            case J_MIDAS_MASK: {
                for (int k = 0; k < c.ns; k++) {
                    int idx = c.scoring[k];
                    if (c.played[idx]->is_face(c.plan->pareidolia)) { c.ov_has[idx] = true; c.ov[idx] = E_GOLD; }
                }
                break;
            }
            case J_BLUEPRINT:
            case J_BRAINSTORM: {
                int t = c.ct[pi];
                if (t < 0) break;
                int tk = c.plan->j[t].key;
                if (tk == J_SPACE || tk == J_DNA || tk == J_SEANCE || tk == J_SUPERPOSITION || tk == J_VAGABOND
                    || tk == J_TODO_LIST)
                    hook_before(c, t);
                break;
            }
            default: break;
        }
    }
    void hook_card(ScoreCtx& c, int pi, Card* x) {
        JState& st = c.js[pi];
        bool sm = c.plan->smeared, par = c.plan->pareidolia;
        switch (c.plan->j[pi].key) {
            case J_GREEDY_JOKER: if (x->has_suit(3, sm)) c.add_mult(3); break;
            case J_LUSTY_JOKER: if (x->has_suit(1, sm)) c.add_mult(3); break;
            case J_WRATHFUL_JOKER: if (x->has_suit(0, sm)) c.add_mult(3); break;
            case J_GLUTTENOUS_JOKER: if (x->has_suit(2, sm)) c.add_mult(3); break;
            case J_SCARY_FACE: if (x->is_face(par)) c.add_chips(30); break;
            case J_EVEN_STEVEN: if (!x->is_stone() && x->rank <= 10 && x->rank % 2 == 0) c.add_mult(4); break;
            case J_ODD_TODD:
                if (!x->is_stone() && (x->rank == 14 || (x->rank <= 10 && x->rank % 2 == 1))) c.add_chips(31);
                break;
            case J_SCHOLAR: if (!x->is_stone() && x->rank == 14) { c.add_chips(20); c.add_mult(4); } break;
            case J_WALKIE_TALKIE:
                if (!x->is_stone() && (x->rank == 10 || x->rank == 4)) { c.add_chips(10); c.add_mult(4); }
                break;
            case J_SMILEY: if (x->is_face(par)) c.add_mult(5); break;
            case J_BUSINESS: if (x->is_face(par)) c.money_chance(2, prob(1, 2), "business"); break;
            case J_PHOTOGRAPH: {
                Card* first = nullptr;
                for (int k = 0; k < c.ns; k++)
                    if (c.played[c.scoring[k]]->is_face(par)) { first = c.played[c.scoring[k]]; break; }
                if (first == x) c.x_mult(2);
                break;
            }
            case J_FIBONACCI:
                if (!x->is_stone() && (x->rank == 14 || x->rank == 2 || x->rank == 3 || x->rank == 5 || x->rank == 8))
                    c.add_mult(8);
                break;
            case J_HIKER: {
                if (c.rng) x->extra += 5;
                else { int i = c.played_index(x); if (i >= 0) { c.hiked[i] += 5; c.any_hiked = true; } }
                break;
            }
            case J_ROUGH_GEM: if (x->has_suit(3, sm)) c.money_now(1); break;
            case J_ARROWHEAD: if (x->has_suit(0, sm)) c.add_chips(50); break;
            case J_ONYX_AGATE: if (x->has_suit(2, sm)) c.add_mult(7); break;
            case J_BLOODSTONE: if (x->has_suit(1, sm)) c.x_chance(1.5, prob(1, 2), "bloodstone"); break;
            case J_IDOL:
                if (!x->is_stone() && x->rank == (st.rank >= 0 ? st.rank : 14)
                    && x->has_suit(st.suit >= 0 ? st.suit : 0, sm))
                    c.x_mult(2);
                break;
            case J_ANCIENT: if (x->has_suit(st.suit >= 0 ? st.suit : 0, sm)) c.x_mult(1.5); break;
            case J_8_BALL:
                if (!x->is_stone() && x->rank == 8 && c.rng && c.cons_room_for_more() && c.roll("8ball") < prob(1, 4)) c.event(EV_CREATE, CK_TAROT | (SRC_8BA << 4));
                break;
            case J_LUCKY_CAT: if (!c.copy && c.lucky_now != 0.0) st.set(jget(st, 1.0) + 0.25 * c.lucky_now); break;
            case J_TICKET: if (c.enh(x) == E_GOLD) c.money_now(4); break;
            case J_WEE: if (!c.copy && !x->is_stone() && x->rank == 2) st.set(jget(st, 0) + 8); break;
            case J_TRIBOULET: if (!x->is_stone() && (x->rank == 12 || x->rank == 13)) c.x_mult(2); break;
            case J_BLUEPRINT:
            case J_BRAINSTORM: {
                int t = c.ct[pi];
                if (t < 0 || !(c.plan->j[t].hooks & H_CARD)) break;
                c.copy = true;
                hook_card(c, t, x);
                c.copy = false;
                break;
            }
            default: break;
        }
    }
    int hook_retrig(ScoreCtx& c, int pi, Card* x, int pos) {
        switch (c.plan->j[pi].key) {
            case J_HANGING_CHAD: return pos == 0 ? 2 : 0;
            case J_HACK: return (!x->is_stone() && x->rank >= 2 && x->rank <= 5) ? 1 : 0;
            case J_DUSK: return hands_left == 1 ? 1 : 0;
            case J_SOCK_AND_BUSKIN: return x->is_face(c.plan->pareidolia) ? 1 : 0;
            case J_SELZER: return 1;
            case J_BLUEPRINT:
            case J_BRAINSTORM: {
                int t = c.ct[pi];
                if (t < 0 || !(c.plan->j[t].hooks & H_RETRIG)) return 0;
                c.copy = true;
                int r = hook_retrig(c, t, x, pos);
                c.copy = false;
                return r;
            }
            default: return 0;
        }
    }
    void hook_held(ScoreCtx& c, int pi, Card* x) {
        switch (c.plan->j[pi].key) {
            case J_RAISED_FIST: {
                Card* low = nullptr;
                for (int i = 0; i < c.nh; i++) {
                    Card* h = c.held[i];
                    if (h->is_stone()) continue;
                    if (!low || h->rank <= low->rank) low = h;          // the lowest rank, the rightmost if tied
                }
                if (low && low == x) c.add_mult(2 * x->chip_value());
                break;
            }
            case J_SHOOT_THE_MOON: if (!x->is_stone() && x->rank == 12) c.add_mult(13); break;
            case J_RESERVED_PARKING: if (x->is_face(c.plan->pareidolia)) c.money_chance(1, prob(1, 2), "parking"); break;
            case J_BARON: if (!x->is_stone() && x->rank == 13) c.x_mult(1.5); break;
            case J_BLUEPRINT:
            case J_BRAINSTORM: {
                int t = c.ct[pi];
                if (t < 0 || !(c.plan->j[t].hooks & H_HELD)) break;
                c.copy = true;
                hook_held(c, t, x);
                c.copy = false;
                break;
            }
            default: break;
        }
    }
    void contains_fx(ScoreCtx& c, int hand, char kind, double amt) {
        if (!(c.contains & (1 << hand))) return;
        if (kind == 'm') c.add_mult(amt); else if (kind == 'c') c.add_chips(amt); else c.x_mult(amt);
    }
    void hook_main(ScoreCtx& c, int pi) {
        JState& st = c.js[pi];
        switch (c.plan->j[pi].key) {
            case J_JOKER: c.add_mult(4); break;
            case J_JOLLY: contains_fx(c, PAIR, 'm', 8); break;
            case J_ZANY: contains_fx(c, TRIPS, 'm', 12); break;
            case J_MAD: contains_fx(c, TWO_PAIR, 'm', 10); break;
            case J_CRAZY: contains_fx(c, STRAIGHT, 'm', 12); break;
            case J_DROLL: contains_fx(c, FLUSH, 'm', 10); break;
            case J_SLY: contains_fx(c, PAIR, 'c', 50); break;
            case J_WILY: contains_fx(c, TRIPS, 'c', 100); break;
            case J_CLEVER: contains_fx(c, TWO_PAIR, 'c', 80); break;
            case J_DEVIOUS: contains_fx(c, STRAIGHT, 'c', 100); break;
            case J_CRAFTY: contains_fx(c, FLUSH, 'c', 80); break;
            case J_HALF: if (c.np <= 3) c.add_mult(20); break;
            case J_BANNER: c.add_chips(30 * discards_left); break;
            case J_MYSTIC_SUMMIT: if (discards_left == 0) c.add_mult(15); break;
            case J_MISPRINT: c.add_mult(c.rng ? (double)c.roll_int("misprint", 0, 23) : 11.5); break;
            case J_ABSTRACT: c.add_mult(3 * (int)jokers.size()); break;
            case J_SUPERNOVA: c.add_mult(hand_played[c.hand] + 1); break;
            case J_RIDE_THE_BUS: case J_GREEN_JOKER: case J_POPCORN: case J_RED_CARD: case J_TROUSERS:
            case J_FLASH: case J_CEREMONIAL:
                c.add_mult(jget(st, 0)); break;
            case J_ICE_CREAM: case J_RUNNER: case J_SQUARE: case J_CASTLE: case J_WEE:
                c.add_chips(jget(st, 0)); break;
            case J_BLUE_JOKER: c.add_chips(2 * (int)deck.size()); break;
            case J_GROS_MICHEL: c.add_mult(15); break;
            case J_CAVENDISH: c.x_mult(3); break;
            case J_SWASHBUCKLER: {
                int uid = c.plan->j[pi].uid, s = 0;
                for (const Joker& o : jokers) if (o.uid != uid) s += o.sell_value();
                c.add_mult(s);
                break;
            }
            case J_FORTUNE_TELLER: c.add_mult(tarots_used); break;
            case J_CONSTELLATION: case J_RAMEN: case J_OBELISK: case J_MADNESS: case J_VAMPIRE: case J_HOLOGRAM:
            case J_CAMPFIRE: case J_GLASS: case J_HIT_THE_ROAD: case J_CAINO: case J_YORICK: {
                double v = jget(st, 1.0);
                c.x_mult(v > 1.0 ? v : 1.0);
                break;
            }
            case J_CARD_SHARP: if (hand_played_round[c.hand] > 0) c.x_mult(3); break;
            case J_BULL: c.add_chips((double)(2 * std::max<long long>(0, money))); break;
            case J_BOOTSTRAPS: c.add_mult((double)(2 * (std::max<long long>(0, money) / 5))); break;
            case J_ACROBAT: if (hands_left == 1) c.x_mult(3); break;
            case J_BLACKBOARD: {
                bool all = true;
                for (int i = 0; i < c.nh; i++) {
                    const Card* h = c.held[i];
                    if (!(!h->is_stone() && (h->flush_suit(0, c.plan->smeared) || h->flush_suit(2, c.plan->smeared))))
                        all = false;
                }
                if (all) c.x_mult(3);
                break;
            }
            case J_FLOWER_POT: {
                static const int order[4] = {1, 3, 0, 2};
                bool filled[4] = {false, false, false, false};
                for (int pass = 0; pass < 2; pass++)
                    for (int k = 0; k < c.ns; k++) {
                        const Card* x = c.played[c.scoring[k]];
                        if ((x->enh == E_WILD) != (pass == 1)) continue;
                        for (int q = 0; q < 4; q++) {
                            int s = order[q];
                            if (!filled[s] && x->has_suit(s, c.plan->smeared)) { filled[s] = true; break; }
                        }
                    }
                if (filled[0] && filled[1] && filled[2] && filled[3]) c.x_mult(3);
                break;
            }
            case J_SEEING_DOUBLE: {
                static const int worder[4] = {2, 3, 0, 1};
                int n[4] = {0, 0, 0, 0}, wilds = 0;
                for (int k = 0; k < c.ns; k++) {
                    const Card* x = c.played[c.scoring[k]];
                    if (x->deb || x->is_stone()) continue;
                    if (x->enh == E_WILD) { wilds++; continue; }
                    for (int s = 0; s < 4; s++) if (x->has_suit(s, c.plan->smeared)) n[s]++;
                }
                for (int w = 0; w < wilds; w++)
                    for (int q = 0; q < 4; q++) if (n[worder[q]] == 0) { n[worder[q]] = 1; break; }
                if (n[2] && (n[0] || n[1] || n[3])) c.x_mult(2);
                break;
            }
            case J_STENCIL: {
                int sten = 0;
                for (const Joker& o : jokers) if (o.key == J_STENCIL) sten++;
                int v = joker_slots - (int)jokers.size() + sten;
                c.x_mult(v > 1 ? v : 1);
                break;
            }
            case J_LOYALTY_CARD: if (jget(st, 6) == 0) c.x_mult(4); break;
            case J_STEEL_JOKER: {
                int n = 0;
                for (int id : full_deck) if (cards[id].enh == E_STEEL) n++;
                c.x_mult(1 + 0.2 * n);
                break;
            }
            case J_DUO: contains_fx(c, PAIR, 'x', 2); break;
            case J_TRIO: contains_fx(c, TRIPS, 'x', 3); break;
            case J_FAMILY: contains_fx(c, QUADS, 'x', 4); break;
            case J_ORDER: contains_fx(c, STRAIGHT, 'x', 3); break;
            case J_TRIBE: contains_fx(c, FLUSH, 'x', 2); break;
            case J_EROSION: c.add_mult(4 * std::max(0, starting_deck_size - (int)full_deck.size())); break;
            case J_STONE: {
                int n = 0;
                for (int id : full_deck) if (cards[id].enh == E_STONE) n++;
                c.add_chips(25 * n);
                break;
            }
            case J_LUCKY_CAT: c.x_mult(jget(st, 1.0)); break;
            case J_THROWBACK: c.x_mult(1 + 0.25 * blinds_skipped); break;
            case J_STUNTMAN: c.add_chips(250); break;
            case J_DRIVERS_LICENSE: {
                int n = 0;
                for (int id : full_deck) if (cards[id].enh != E_NONE) n++;
                if (n >= 16) c.x_mult(3);
                break;
            }
            case J_BLUEPRINT:
            case J_BRAINSTORM: {
                int t = c.ct[pi];
                if (t < 0 || !(c.plan->j[t].hooks & H_MAIN)) break;
                c.copy = true;
                hook_main(c, t);
                c.copy = false;
                break;
            }
            default: break;
        }
    }

    // scoring.score_hand. played / held point into the arena on a real play (rng != null); a prediction may
    // pass placeholder cards.
    void score_once(ScoreCtx& ctx, Card* const* played, int np, Card* const* held, int nh, PyRandom* r,
                    bool commit, const Plan& plan, bool arm_applied) {
        ctx.g = this;
        ctx.plan = &plan;
        ctx.rng = r;
        ctx.np = np;
        for (int i = 0; i < np; i++) ctx.played[i] = played[i];
        ctx.nh = nh;
        for (int i = 0; i < nh; i++) ctx.held[i] = held[i];
        HandEval res = evaluate((const Card* const*)ctx.played, np, plan.four_fingers, plan.shortcut, plan.smeared);
        ctx.hand = res.hand;
        ctx.contains = res.contains;
        ctx.ns = 0;
        if (plan.splash) { for (int i = 0; i < np; i++) ctx.scoring[ctx.ns++] = i; }
        else { for (int i = 0; i < np; i++) if (res.scoring & (1 << i)) ctx.scoring[ctx.ns++] = i; }
        int hpr = 0;
        for (int i = 0; i < N_HANDS; i++) hpr += hand_played_round[i];
        ctx.first_hand = hpr == 0;
        for (int k = 0; k < plan.n; k++) {
            const Joker* j = joker_by_uid(plan.j[k].uid);
            new (&ctx.js[k]) JState(j ? j->st : JState());
        }
        resolve_copiers(ctx);
        int b = plan.boss;

        for (int k = 0; k < plan.nbefore; k++) hook_before(ctx, plan.before[k]);

        int level = hand_levels[ctx.hand];
        if (b == B_ARM && !arm_applied && level > 1) level -= 1;
        level += ctx.level_up;
        int bc, bm;
        hand_base(ctx.hand, level, bc, bm);
        if (b == B_FLINT) {
            bc = std::max(0, (int)(bc / 2.0 + 0.5));
            bm = std::max(1, (int)(bm / 2.0 + 0.5));
        }
        ctx.chips = (double)bc;
        ctx.mult = (double)bm;

        int glass_broken[5], nglass = 0;

        for (int pos = 0; pos < ctx.ns; pos++) {
            int idx = ctx.scoring[pos];
            Card* c = ctx.played[idx];
            if (c->deb) continue;
            uint8_t e = (ctx.ov_has[idx]) ? ctx.ov[idx] : c->enh;
            int reps = 1 + (c->seal == S_RED ? 1 : 0);
            for (int k = 0; k < plan.nretrig; k++) reps += hook_retrig(ctx, plan.retrig[k], c, pos);
            for (int rep = 0; rep < reps; rep++) {
                ctx.lucky_now = 0.0;
                int base;
                if (e == E_STONE) base = 50;
                else if (c->enh == E_STONE) base = c->rank == 14 ? 11 : std::min((int)c->rank, 10);
                else base = c->chip_value();
                ctx.add_chips(base + c->extra + (ctx.any_hiked ? ctx.hiked[idx] : 0));
                if (e == E_BONUS) ctx.add_chips(30);
                else if (e == E_MULT) ctx.add_mult(4);
                else if (e == E_LUCKY) {
                    double p = prob(1, 5);
                    if (!r) { ctx.add_mult(20 * p); ctx.lucky_now += p; }
                    else if (ctx.roll("lucky_mult") < p) { ctx.add_mult(20); ctx.lucky_now += 1; }
                }
                if (c->seal == S_GOLD) ctx.money_now(3);
                if (e == E_LUCKY) {
                    double p2 = prob(1, 15);
                    if (!r) { ctx.money += 20 * p2; ctx.lucky_now += p2; }
                    else if (ctx.roll("lucky_money") < p2) { ctx.money += 20; ctx.lucky_now += 1; }
                }
                if (e == E_GLASS) ctx.x_mult(2);
                if (c->ed == ED_FOIL) ctx.add_chips(50);
                else if (c->ed == ED_HOLO) ctx.add_mult(10);
                else if (c->ed == ED_POLY) ctx.x_mult(1.5);
                for (int k = 0; k < plan.ncard; k++) hook_card(ctx, plan.card[k], c);
            }
            if (e == E_GLASS && r && ctx.roll("glass") < prob(1, 4)) glass_broken[nglass++] = c->uid;
        }

        bool any_steel = false;
        for (int i = 0; i < ctx.nh; i++) if (ctx.held[i]->enh == E_STEEL) any_steel = true;
        if (plan.nheld > 0 || any_steel) {
            for (int i = 0; i < ctx.nh; i++) {
                Card* c = ctx.held[i];
                if (c->deb) continue;
                double b0 = ctx.chips, b1 = ctx.mult, b2 = ctx.money;
                int b3 = ctx.nev;
                if (c->enh == E_STEEL) ctx.x_mult(1.5);
                for (int k = 0; k < plan.nheld; k++) hook_held(ctx, plan.held[k], c);
                if (c->enh != E_STEEL && ctx.chips == b0 && ctx.mult == b1 && ctx.money == b2 && ctx.nev == b3)
                    continue;
                int extra = (c->seal == S_RED ? 1 : 0) + plan.mime;
                for (int q = 0; q < extra; q++) {
                    if (c->enh == E_STEEL) ctx.x_mult(1.5);
                    for (int k = 0; k < plan.nheld; k++) hook_held(ctx, plan.held[k], c);
                }
            }
        }

        for (int k = 0; k < plan.n; k++) {
            uint8_t ed = plan.j[k].edition;
            if (ed == ED_FOIL) ctx.add_chips(50);
            else if (ed == ED_HOLO) ctx.add_mult(10);
            if (plan.j[k].hooks & H_MAIN) hook_main(ctx, k);
            if (plan.baseball && plan.j[k].rarity == 2)
                for (int q = 0; q < plan.baseball; q++) ctx.x_mult(1.5);
            if (ed == ED_POLY) ctx.x_mult(1.5);
        }

        if (has_voucher(V_OBSERVATORY))
            for (const Cons& cs : consumables)
                if (cs.kind() == CK_PLANET && CONS_DEFS[cs.id].planet_hand == ctx.hand) ctx.x_mult(1.5);

        if (deck_type == D_PLASMA) {
            double avg = std::floor((ctx.chips + ctx.mult) / 2);
            ctx.chips = ctx.mult = avg;
        }

        if (commit) {
            for (Joker& j : jokers)
                for (int k = 0; k < plan.n; k++)
                    if (plan.j[k].uid == j.uid) { j.st = ctx.js[k]; break; }
            if (ctx.level_up) hand_levels[ctx.hand] += ctx.level_up;
            for (int i = 0; i < np; i++) if (ctx.ov_has[i]) ctx.played[i]->enh = ctx.ov[i];
            if (nglass) {
                std::vector<int> v(glass_broken, glass_broken + nglass);
                destroy_cards(v);
            }
        }
    }

    // Returns the score; fills `out` (hand, money, scoring cards, events) when given.
    double score_hand(Card* const* played, int np, Card* const* held, int nh, PyRandom* r, bool commit,
                      const Plan& plan, bool arm_applied, bool hooked, ScoreResult* out = nullptr) {
        if (!r && !hooked && plan.boss == B_HOOK && nh > 0) {
            bool steel = false;
            for (int i = 0; i < nh; i++) if (held[i]->enh == E_STEEL) steel = true;
            if (plan.nheld > 0 || plan.blackboard || steel) {
                // the mean over the Hook's equally likely discards (scoring._hook_expected)
                double total = 0.0;
                int nvar = 0, hand0 = 0;
                bool first = true;
                Card* sub[MAX_HAND + 8];
                if (nh <= 2) {
                    ScoreCtx ctx;
                    score_once(ctx, played, np, sub, 0, nullptr, false, plan, arm_applied);
                    total += ctx.chips * ctx.mult;
                    hand0 = ctx.hand;
                    nvar = 1;
                } else {
                    for (int a = 0; a < nh; a++)
                        for (int b2 = a + 1; b2 < nh; b2++) {
                            int m = 0;
                            for (int i = 0; i < nh; i++) if (i != a && i != b2) sub[m++] = held[i];
                            ScoreCtx ctx;
                            score_once(ctx, played, np, sub, m, nullptr, false, plan, arm_applied);
                            total += ctx.chips * ctx.mult;
                            if (first) { hand0 = ctx.hand; first = false; }
                            nvar++;
                        }
                }
                if (out) { out->hand = hand0; out->score = std::floor(total / nvar); }
                return std::floor(total / nvar);
            }
        }
        ScoreCtx ctx;
        score_once(ctx, played, np, held, nh, r, commit, plan, arm_applied);
        double score = std::floor(ctx.chips * ctx.mult);
        if (out) {
            out->score = score;
            out->hand = ctx.hand;
            out->money = ctx.money;
            out->scoring = 0;
            for (int k = 0; k < ctx.ns; k++) out->scoring |= 1 << ctx.scoring[k];
            out->nev = ctx.nev;
            for (int i = 0; i < ctx.nev; i++) out->ev[i] = ctx.ev[i];
        }
        return score;
    }

    // Expected score and hand type of playing these hand slots (Game.predict). view: the hand as the player
    // sees it (hand_view), made once per decision.
    struct View { Card c[MAX_HAND]; int n = 0; };
    void hand_view(View& v) const {
        v.n = (int)hand.size();
        for (int i = 0; i < v.n; i++) {
            const Card& c = cards[hand[i]];
            if (c.hidden) {
                Card ph;
                ph.rank = 2; ph.suit = 0; ph.enh = E_HIDDEN; ph.uid = c.uid;
                v.c[i] = ph;
            } else {
                v.c[i] = c;
            }
        }
    }
    double predict(const int* positions, int np, const Plan& plan, View& view, int& hand_out) {
        Card* played[5];
        Card* held[MAX_HAND];
        int nh = 0;
        for (int i = 0; i < np; i++) played[i] = &view.c[positions[i]];
        for (int i = 0; i < view.n; i++) {
            bool in = false;
            for (int k = 0; k < np; k++) if (positions[k] == i) in = true;
            if (!in) held[nh++] = &view.c[i];
        }
        ScoreResult res;
        double sc = score_hand(played, np, held, nh, nullptr, false, plan, false, false, &res);
        if (violates_boss((const Card* const*)played, np)) sc = 0.0;
        hand_out = res.hand;
        return sc;
    }

    void play(const std::vector<int>& positions) {
        if (!(state == ST_SELECTING_HAND && positions.size() >= 1 && positions.size() <= 5))
            throw std::runtime_error("play: not allowed");
        graveyard.clear();
        int np = (int)positions.size();
        int played_id[5];
        std::vector<int> held_id;
        for (int i = 0; i < np; i++) played_id[i] = hand[positions[i]];
        for (size_t i = 0; i < hand.size(); i++)
            if (std::find(positions.begin(), positions.end(), (int)i) == positions.end()) held_id.push_back(hand[i]);
        Card* played[5];
        for (int i = 0; i < np; i++) { played[i] = &cards[played_id[i]]; played[i]->hidden = false; }
        int b = boss_active();
        Plan plan;
        make_plan(plan);
        bool violated = violates_boss((const Card* const*)played, np);
        HandEval res = evaluate((const Card* const*)played, np, plan.four_fingers, plan.shortcut, plan.smeared);
        int h = res.hand;
        bool arm_lowered = b == B_ARM && hand_levels[h] > 1;
        if (arm_lowered) hand_levels[h] -= 1;
        if (b == B_HOOK && !held_id.empty()) {
            std::vector<int> hooked;
            if (exact) {
                std::vector<int> pl = held_id;
                int take = std::min<int>(2, (int)held_id.size());
                for (int q = 0; q < take; q++) { int k = xpick("hook", pl); hooked.push_back(pl[k]); pl.erase(pl.begin() + k); }
            } else {
                std::vector<size_t> pick = rng.sample_indices(held_id.size(), std::min<size_t>(2, held_id.size()));
                for (size_t k : pick) hooked.push_back(held_id[k]);
            }
            std::vector<int> nh2;
            for (int id : held_id) if (!in_pile(hooked, id)) nh2.push_back(id);
            held_id = nh2;
            std::vector<int> hand2;
            for (int id : hand) if (!in_pile(hooked, id)) hand2.push_back(id);
            hand = hand2;
            discard_effects(hooked, true);
            for (int id : hooked) discard_pile.push_back(id);
            if (f_trading_destroy >= 0) {
                std::vector<int> v{f_trading_destroy};
                f_trading_destroy = -1;
                destroy_cards(v);
            }
        }
        double sc;
        double mny;
        int scoring;
        ScoreResult sr;
        if (violated) {
            sc = 0; mny = 0; scoring = res.scoring; sr.nev = 0;
        } else {
            Card* held[MAX_HAND + 8];
            int nh = 0;
            for (int id : held_id) held[nh++] = &cards[id];
            sc = score_hand(played, np, held, nh, &rng, true, plan, true, true, &sr);
            h = sr.hand; mny = sr.money; scoring = sr.scoring;
        }
        last_play_score = sc;
        if (b != B_NONE && has(J_MATADOR)) {
            bool deb = false;
            for (int i = 0; i < np; i++) if ((scoring & (1 << i)) && played[i]->deb) deb = true;
            bool triggered = violated || deb || b == B_HOOK || b == B_TOOTH || b == B_FLINT || b == B_CRIMSON_HEART
                             || arm_lowered || (b == B_OX && h == ox_hand);
            if (triggered) money += 8 * count(J_MATADOR);
        }
        if (b == B_OX && h == ox_hand) money = 0;
        if (b == B_TOOTH) money -= np;
        if (b == B_MOUTH && mouth_hand < 0) mouth_hand = h;
        round_hand_types |= 1u << h;
        last_hand = h;
        money += (long long)mny;
        chips += sc;
        hand_played[h] += 1;
        hand_played_round[h] += 1;
        total_hands_played += 1;
        hands_left -= 1;
        for (int i = 0; i < np; i++) ante_played[played_id[i]] = 1;
        {
            std::vector<int> snap;
            for (const Joker& j : jokers) snap.push_back(j.uid);
            for (int uid : snap) {
                int i = find_joker(uid);
                if (i < 0) continue;
                if ((jokers[i].d().hooks & H_AFTER) && !jokers[i].debuffed) hook_after(uid);
            }
        }
        std::vector<int> destroyed;
        for (int i = 0; i < sr.nev; i++) {
            if (sr.ev[i].type == EV_CREATE) create_consumable(sr.ev[i].arg & 15, false, C_NONE, sr.ev[i].arg >> 4);
            else if (sr.ev[i].type == EV_DNA) add_card(sr.ev[i].arg, true, false);
            else destroyed.push_back(sr.ev[i].arg);
        }
        if (!destroyed.empty()) destroy_cards(destroyed);
        {
            std::vector<int> hand2;
            for (int id : hand) {
                bool was = false;
                for (int i = 0; i < np; i++) if (played_id[i] == id) was = true;
                if (!was) hand2.push_back(id);
            }
            hand = hand2;
        }
        for (int i = 0; i < np; i++) if (in_pile(full_deck, played_id[i])) discard_pile.push_back(played_id[i]);
        if (b == B_CRIMSON_HEART && !jokers.empty())
            crimson_disabled = crimson_pick();
        if (chips >= target) {
            win_round(false);
        } else if (hands_left <= 0) {
            int bones = -1;
            for (size_t i = 0; i < jokers.size(); i++)
                if (jokers[i].key == J_MR_BONES && !jokers[i].debuffed) { bones = (int)i; break; }
            if (bones >= 0 && chips >= 0.25 * target) {
                destroy_joker(jokers[bones].uid);
                win_round(true);
            } else if (immortal) {
                fail_round();
            } else {
                state = ST_GAME_OVER;
            }
        } else {
            if (b == B_SERPENT) draw(3, true); else draw(-1000000, true);
            if (hand.empty()) { if (immortal) fail_round(); else state = ST_GAME_OVER; }
        }
    }

    void discard_effects(const std::vector<int>& ids, bool from_hook = false) {
        for (int id : ids) cards[id].hidden = false;
        std::vector<int> snap;
        for (const Joker& j : jokers) snap.push_back(j.uid);
        for (int uid : snap) {
            int i = find_joker(uid);
            if (i < 0) continue;
            if ((jokers[i].d().hooks & H_DISCARD) && !jokers[i].debuffed) {
                if (from_hook && (jokers[i].key == J_BURNT || copies(i) == J_BURNT)) continue;
                hook_discard(uid, uid, ids);
            }
        }
        for (int id : ids)
            if (cards[id].seal == S_PURPLE && !cards[id].deb) create_consumable(CK_TAROT, false, C_NONE, SRC_8BA);
    }

    void discard(const std::vector<int>& positions) {
        if (!(state == ST_SELECTING_HAND && discards_left > 0 && positions.size() >= 1 && positions.size() <= 5))
            throw std::runtime_error("discard: not allowed");
        std::vector<int> ids;
        for (int p : positions) ids.push_back(hand[p]);
        discard_effects(ids);
        std::vector<int> hand2;
        for (int id : hand) if (!in_pile(ids, id)) hand2.push_back(id);
        hand = hand2;
        for (int id : ids) discard_pile.push_back(id);
        if (f_trading_destroy >= 0) {
            std::vector<int> v{f_trading_destroy};
            f_trading_destroy = -1;
            destroy_cards(v);
        }
        discards_left -= 1;
        discards_used_round += 1;
        if (boss_active() == B_SERPENT) draw(3); else draw();
    }

    // game-event hooks; `self`: the joker in the list, `as`: the joker whose hook runs (a copier runs its
    // target's hook as that target)
    void hook_after(int uid) {
        int i = find_joker(uid);
        if (i < 0) return;
        Joker& j = jokers[i];
        if (j.key == J_ICE_CREAM) {
            j.st.set(j.st.get(100) - 5);
            if (j.st.val <= 0) destroy_joker(uid);
        } else if (j.key == J_SELZER) {
            j.st.set(j.st.get(10) - 1);
            if (j.st.val <= 0) destroy_joker(uid);
        }
    }
    void hook_discard(int self_uid, int as_uid, const std::vector<int>& ids) {
        int i = find_joker(as_uid);
        if (i < 0) return;
        Joker& j = jokers[i];
        switch (j.key) {
            case J_GREEN_JOKER: j.st.set(std::max(0.0, j.st.get(0) - 1)); break;
            case J_FACELESS: {
                bool par = has(J_PAREIDOLIA);
                int n = 0;
                for (int id : ids) if (cards[id].is_face(par)) n++;
                if (n >= 3) money += 5;
                break;
            }
            case J_MAIL: {
                int rank = j.st.rank >= 0 ? j.st.rank : 2, n = 0;
                for (int id : ids) if (!cards[id].is_stone() && cards[id].rank == rank) n++;
                money += 5 * n;
                break;
            }
            case J_RAMEN: {
                j.st.set(j.st.get(2.0) - 0.01 * (double)ids.size());
                if (j.st.val <= 1.0) destroy_joker(as_uid);
                break;
            }
            case J_TRADING:
                if (discards_used_round == 0 && ids.size() == 1) { money += 3; f_trading_destroy = ids[0]; }
                break;
            case J_CASTLE: {
                int suit = j.st.suit >= 0 ? j.st.suit : 0, n = 0;
                for (int id : ids) if (cards[id].has_suit(suit)) n++;
                j.st.set(j.st.get(0) + 3 * n);
                break;
            }
            case J_HIT_THE_ROAD: {
                int n = 0;
                for (int id : ids) if (!cards[id].is_stone() && cards[id].rank == 11) n++;
                j.st.set(j.st.get(1.0) + 0.5 * n);
                break;
            }
            case J_BURNT: {
                if (discards_used_round == 0) {
                    const Card* cs[5];
                    int n = 0;
                    for (int id : ids) cs[n++] = &cards[id];
                    int h = evaluate(cs, n, has(J_FOUR_FINGERS), has(J_SHORTCUT), has(J_SMEARED)).hand;
                    hand_levels[h] += 1;
                }
                break;
            }
            case J_YORICK: {
                int left = (j.st.left >= 0 ? j.st.left : 23) - (int)ids.size();
                while (left <= 0) { j.st.set(j.st.get(1.0) + 1); left += 23; }
                j.st.left = left;
                break;
            }
            case J_BLUEPRINT:
            case J_BRAINSTORM: {
                int t = copy_target(i);
                if (t < 0) break;
                int tk = jokers[t].key;
                if (tk == J_FACELESS || tk == J_MAIL || tk == J_BURNT) hook_discard(self_uid, jokers[t].uid, ids);
                break;
            }
            default: break;
        }
    }
    long long hook_end_round(int uid) {
        int i = find_joker(uid);
        if (i < 0) return 0;
        Joker& j = jokers[i];
        switch (j.key) {
            case J_POPCORN: {
                j.st.set(j.st.get(20) - 4);
                if (j.st.val <= 0) destroy_joker(uid);
                return 0;
            }
            case J_GOLDEN: return 4;
            case J_EGG: j.sell_bonus += 3; return 0;
            case J_GROS_MICHEL:
                if (roll("gros_michel") < prob(1, 6)) { destroy_joker(uid); f_gros_extinct = true; }
                return 0;
            case J_CAVENDISH: if (roll("cavendish") < prob(1, 1000)) destroy_joker(uid); return 0;
            case J_DELAYED_GRAT: return discards_used_round == 0 ? 2 * discards_left : 0;
            case J_TODO_LIST: j.st.hand = exact ? todo_pick(j.st.hand) : (int)rng.randrange(0, 9); return 0;
            case J_MAIL: if (exact) return 0; j.st.rank = random_deck_rank(); return 0;
            case J_ROCKET: return (long long)j.st.get(1);
            case J_CLOUD_9: {
                int n = 0;
                for (int id : full_deck) if (!cards[id].is_stone() && cards[id].rank == 9) n++;
                return n;
            }
            case J_SATELLITE: {
                int n = 0;
                for (int p = 0; p < N_PLANETS; p++) if ((planets_used >> p) & 1u) n++;
                return n;
            }
            case J_IDOL: { if (exact) return 0; int r, s; random_deck_card(r, s); j.st.rank = r; j.st.suit = s; return 0; }
            case J_ANCIENT: {
                if (exact) return 0;
                int opts[4], n = 0;
                for (int s = 0; s < 4; s++) if (s != j.st.suit) opts[n++] = s;
                j.st.suit = opts[rng.choice_index(n)];
                return 0;
            }
            case J_GIFT: {
                for (Joker& o : jokers) o.sell_bonus += 1;
                for (Cons& c : consumables) c.sell_bonus += 1;
                return 0;
            }
            case J_TURTLE_BEAN: {
                j.st.set(j.st.get(5) - 1);
                if (j.st.val <= 0) destroy_joker(uid);
                return 0;
            }
            case J_CASTLE: { if (exact) return 0; int r, s; random_deck_card(r, s); j.st.suit = s; return 0; }
            case J_HIT_THE_ROAD: j.st.set(1.0); return 0;
            case J_INVISIBLE: j.st.set(j.st.get(0) + 1); return 0;
            default: return 0;
        }
    }
    void hook_init(Joker& j) {
        switch (j.key) {
            case J_RIDE_THE_BUS: case J_GREEN_JOKER: case J_RUNNER: case J_SQUARE: case J_RED_CARD: case J_TROUSERS:
            case J_FLASH: case J_CEREMONIAL: case J_WEE: case J_INVISIBLE:
                if (!j.st.has_val) j.st.set(0); break;
            case J_ICE_CREAM: if (!j.st.has_val) j.st.set(100); break;
            case J_POPCORN: if (!j.st.has_val) j.st.set(20); break;
            case J_CONSTELLATION: case J_OBELISK: case J_MADNESS: case J_VAMPIRE: case J_HOLOGRAM: case J_LUCKY_CAT:
            case J_CAMPFIRE: case J_GLASS: case J_HIT_THE_ROAD: case J_CAINO:
                if (!j.st.has_val) j.st.set(1.0); break;
            case J_ROCKET: if (!j.st.has_val) j.st.set(1); break;
            case J_RAMEN: if (!j.st.has_val) j.st.set(2.0); break;
            case J_SELZER: if (!j.st.has_val) j.st.set(10); break;
            case J_LOYALTY_CARD: if (!j.st.has_val) j.st.set(6); break;
            case J_TURTLE_BEAN: if (!j.st.has_val) j.st.set(5); break;
            case J_TODO_LIST: { int h = exact ? todo_pick(-1) : (int)rng.randrange(0, 9); if (j.st.hand < 0) j.st.hand = h; break; }
            case J_MAIL: { if (exact) { j.st.rank = tgt_mail; break; } int r = random_deck_rank(); if (j.st.rank < 0) j.st.rank = r; break; }
            case J_IDOL:
                if (exact) { j.st.rank = tgt_idol_rank; j.st.suit = tgt_idol_suit; break; }
                if (j.st.rank < 0) { int r, s; random_deck_card(r, s); j.st.rank = r; j.st.suit = s; }
                break;
            case J_ANCIENT: { if (exact) { j.st.suit = tgt_ancient; break; } int s = (int)rng.randrange(4); if (j.st.suit < 0) j.st.suit = s; break; }
            case J_CASTLE: {
                if (exact) { j.st.set(j.st.get(0)); j.st.suit = tgt_castle; break; }
                int r, s;
                random_deck_card(r, s);
                j.st.set(j.st.get(0));
                if (j.st.suit < 0) j.st.suit = s;
                break;
            }
            case J_YORICK:
                j.st.set(j.st.get(1.0));
                if (j.st.left < 0) j.st.left = 23;
                break;
            default: break;
        }
    }
    void hook_blind_select(int self_uid, int as_uid) {
        int i = find_joker(as_uid);
        if (i < 0) return;
        switch (jokers[i].key) {
            case J_CEREMONIAL: {
                if (i + 1 < (int)jokers.size()) {
                    if (!jokers[i + 1].eternal) {
                        jokers[i].st.set(jokers[i].st.get(0) + 2 * jokers[i + 1].sell_value());
                        destroy_joker(jokers[i + 1].uid);
                    }
                }
                break;
            }
            case J_MARBLE: {
                if (exact) {
                    int fk = sg.pseudorandom("marb_fr", 1, 52) - 1;
                    add_card(new_card(P_CARD_RANK[fk], P_CARD_SUIT[fk], E_STONE));
                    break;
                }
                int s = (int)rng.randrange(4);
                add_card(new_card(2, s, E_STONE));
                break;
            }
            case J_BURGLAR: hands_left += 3; discards_left = 0; break;
            case J_MADNESS: {
                if (blind_idx < 2) {
                    jokers[i].st.set(jokers[i].st.get(1.0) + 0.5);
                    std::vector<int> cand;
                    for (const Joker& o : jokers) if (o.uid != as_uid && !o.eternal) cand.push_back(o.uid);
                    if (!cand.empty()) destroy_joker(exact ? cand[xpick("madness", cand)] : cand[rng.choice_index(cand.size())]);
                }
                break;
            }
            case J_RIFF_RAFF: {
                for (int k = 0; k < 2; k++)
                    if ((int)jokers.size() < joker_slots) {
                        Joker j;
                        random_joker(j, 1, false, nullptr, false, "rif");
                        add_joker(j);
                    }
                break;
            }
            case J_CARTOMANCER: create_consumable(CK_TAROT, false, C_NONE, SRC_CAR); break;
            case J_BLUEPRINT:
            case J_BRAINSTORM: {
                int t = copy_target(i);
                if (t < 0) break;
                int tk = jokers[t].key;
                if (tk == J_CARTOMANCER || tk == J_MARBLE || tk == J_RIFF_RAFF || tk == J_BURGLAR)
                    hook_blind_select(self_uid, jokers[t].uid);
                break;
            }
            default: break;
        }
    }
    void hook_on_sell(const Joker& j) {                 // j: already removed from the list
        if (j.key == J_LUCHADOR) {
            if (boss_active() != B_NONE) disable_boss();
        } else if (j.key == J_DIET_COLA) {
            add_tag(T_DOUBLE);
        } else if (j.key == J_INVISIBLE) {
            if (j.st.get(0) >= 2 && !jokers.empty()) {
                std::vector<int> all;
                for (size_t q = 0; q < jokers.size(); q++) all.push_back((int)q);
                const Joker src = jokers[pick_joker("invisible", all)];
                Joker cp;
                cp.key = src.key;
                cp.edition = src.edition == ED_NEG ? ED_NONE : src.edition;
                cp.base_cost = src.base_cost;
                cp.st = src.st;
                cp.has_cost = src.has_cost;
                cp.cost = src.cost;
                add_joker(cp);
            }
        }
    }

    // ---------------------------------------------------------------- end of round
    // --immortal: the round was not won. It ends like a won one (the jokers' round-end effects, a rental's fee and a perishable's clock still run, the
    // blinds move on, a shop opens) but nothing is paid: no reward, no interest, no hands left, no joker's income. Failing the ante-8 boss ends the game.
    void fail_round() {
        if (immortal_pay) { win_round(false, false); return; }
        bool was_boss = blind_idx == 2;
        {
            std::vector<int> snap;
            for (const Joker& j : jokers) snap.push_back(j.uid);
            for (int uid : snap) {
                int i = find_joker(uid);
                if (i < 0 || jokers[i].debuffed) continue;
                if (was_boss && jokers[i].key == J_CAMPFIRE) jokers[i].st.set(1.0);
                if (jokers[i].d().hooks & H_END_ROUND) (void)hook_end_round(uid);
            }
        }
        for (Joker& j : jokers) {
            if (j.has_perish) {
                j.perishable -= 1;
                if (j.perishable <= 0) j.debuffed = true;
            }
        }
        total_unused_discards += discards_left;
        f_verdant = false;
        for (int id : full_deck) { cards[id].deb = false; cards[id].hidden = false; }
        for (Joker& j : jokers) j.hidden = false;
        juggle = 0;
        if (was_boss) {
            if (ante >= win_ante) { state = ST_GAME_OVER; return; }
            ante += 1;
            blind_idx = 0;
            new_ante();
        } else {
            blind_idx += 1;
        }
        if (exact) reset_round_targets();
        open_shop();
    }
    void win_round(bool saved_by_bones, bool counted = true) {            // counted false: a failed blind that is paid anyway (immortal_pay)
        bool was_boss = blind_idx == 2;
        if (counted) {
            cleared_bits |= 1ull << std::min(47, 3 * (ante - 1) + blind_idx);
            blinds_beaten += 1;
            furthest_blind = std::max(furthest_blind, 3 * (ante - 1) + blind_idx + 1);
        }
        long long earned = 0;
        // at the round's end, before the cash-out: held Gold cards pay and rentals are charged, so interest counts the money after both
        // (the game's end_round eases these dollars; evaluate_round then computes interest from G.GAME.dollars)
        {
            long long pre = 0;
            int mime = count_with_copies(J_MIME);
            for (int id : hand) {
                const Card c = cards[id];
                if (c.deb) continue;
                int triggers = 1 + mime + (c.seal == S_RED ? 1 : 0);
                if (c.enh == E_GOLD) pre += 3 * triggers;
                if (c.seal == S_BLUE)
                    for (int k = 0; k < triggers; k++) create_consumable(CK_PLANET, false, HAND_TO_PLANET[last_hand]);
            }
            for (const Joker& j : jokers) if (j.rental) pre -= 3;
            money += pre;
        }
        if (saved_by_bones) {}
        else if (blind_idx == 0) earned += (stake >= 1 && !small_blind_pay) ? 0 : 3;
        else earned += blind_idx == 1 ? 4 : 5;
        if (deck_type == D_GREEN) {
            earned += 2 * hands_left + discards_left;
        } else {
            earned += hands_left;
            earned += (1 + count(J_TO_THE_MOON)) * std::min<long long>(std::max<long long>(0, money) / 5, interest_cap());
        }
        {
            std::vector<int> snap;
            for (const Joker& j : jokers) snap.push_back(j.uid);
            for (int uid : snap) {
                int i = find_joker(uid);
                if (i < 0) continue;
                if (jokers[i].debuffed) continue;
                if (was_boss && jokers[i].key == J_ROCKET) jokers[i].st.set(jokers[i].st.get(1) + 2);
                if (was_boss && jokers[i].key == J_CAMPFIRE) jokers[i].st.set(1.0);
                if (jokers[i].d().hooks & H_END_ROUND) earned += hook_end_round(uid);
            }
        }
        for (Joker& j : jokers) {
            if (j.has_perish) {
                j.perishable -= 1;
                if (j.perishable <= 0) j.debuffed = true;
            }
        }
        total_unused_discards += discards_left;
        money += earned;
        f_verdant = false;
        for (int id : full_deck) { cards[id].deb = false; cards[id].hidden = false; }
        for (Joker& j : jokers) j.hidden = false;
        juggle = 0;
        if (was_boss) {
            if (deck_type == D_ANAGLYPH) add_tag(T_DOUBLE);
            auto it = std::find(pending_tags.begin(), pending_tags.end(), (int8_t)T_INVESTMENT);
            if (it != pending_tags.end()) { pending_tags.erase(it); money += 25; }
            if (ante >= win_ante) { state = counted ? ST_WON : ST_GAME_OVER; return; }
            ante += 1;
            blind_idx = 0;
            new_ante();
        } else {
            blind_idx += 1;
        }
        if (exact) reset_round_targets();
        open_shop();
    }

    // ---------------------------------------------------------------- shop (exact-seed mode)
    // the cards that exist, for the pools (the game marks every created card as used and the pools skip used keys)
    struct UsedSet { bool j[J_COUNT]; bool c[C_COUNT]; };
    void exact_used(UsedSet& u, const bool* exj, uint64_t exc) const {
        for (int k = 0; k < J_COUNT; k++) u.j[k] = exj ? exj[k] : false;
        for (int c = 0; c < C_COUNT; c++) u.c[c] = (exc >> c) & 1ull;
        for (const Joker& o : jokers) u.j[o.key] = true;
        for (const Cons& c : consumables) u.c[c.id] = true;
        for (const ShopItem& it : shop) {
            if (it.kind == SK_JOKER) u.j[it.key] = true;
            else if (it.kind == SK_TAROT || it.kind == SK_PLANET || it.kind == SK_SPECTRAL) u.c[it.key] = true;
        }
        for (const PackItem& p : pack_cards) {
            if (p.kind == PI_JOKER) u.j[p.joker.key] = true;
            else if (p.kind == PI_CONS) u.c[p.cons.id] = true;
        }
    }
    int exact_pick(const std::vector<int>& pool, const std::string& key) {
        int n = (int)pool.size();
        int v = pool[sg.pseudorandom(key, 1, n) - 1];
        for (int it = 2; v < 0; it++) v = pool[sg.pseudorandom(key + "_resample" + std::to_string(it), 1, n) - 1];
        return v;
    }
    uint8_t exact_poll_edition(const std::string& key, double mod, bool no_neg, bool guaranteed = false) {
        double rate = has_voucher(V_GLOW_UP) ? 4 : (has_voucher(V_HONE) ? 2 : 1);
        double p = sg.pseudorandom(key);
        if (guaranteed) {
            if (p > 1 - 0.003 * 25 && !no_neg) return ED_NEG;
            if (p > 1 - 0.006 * 25) return ED_POLY;
            if (p > 1 - 0.02 * 25) return ED_HOLO;
            if (p > 1 - 0.04 * 25) return ED_FOIL;
            return ED_NONE;
        }
        if (p > 1 - 0.003 * mod && !no_neg) return ED_NEG;
        if (p > 1 - 0.006 * rate * mod) return ED_POLY;
        if (p > 1 - 0.02 * rate * mod) return ED_HOLO;
        if (p > 1 - 0.04 * rate * mod) return ED_FOIL;
        return ED_NONE;
    }
    // create_card('Joker', ...): rarity 0 rolled, 1/2/3 the tags' thresholds (0, 0.9, 1), 4 legendary; sticker: in a shop or pack
    // (stickers by stake); src: the stream name appended ('sho', 'buf', 'uta' ...); pack: a pack's stickers use their own streams
    bool exact_joker(Joker& j, int rarity, bool sticker, const bool* exj, const char* src, bool pack = false) {
        UsedSet u;
        exact_used(u, exj, 0ull);
        std::string a = std::to_string(ante), ap = src;
        double thr = rarity == 1 ? 0.0 : rarity == 2 ? 0.9 : rarity == 3 ? 1.0 : -1.0;
        double r = thr >= 0 ? thr : sg.pseudorandom("rarity" + a + ap);
        int rr = rarity == 4 ? 4 : (r > 0.95 ? 3 : (r > 0.7 ? 2 : 1));
        bool legendary = rarity == 4;
        const int16_t* P = POOL_J[rr - 1];
        int n = N_POOL_J[rr - 1], size = 0;
        std::vector<int> pool;
        for (int i = 0; i < n; i++) {
            int k = P[i];
            bool add = !(u.j[k] && !showman());
            if (add && JOKER_GATE[k] > 0) {                      // Golden Ticket, Glass, Steel, Stone, Lucky Cat: the deck needs the enhancement
                add = false;
                for (int id : full_deck) if (cards[id].enh == JOKER_GATE[k]) { add = true; break; }
            }
            if (k == J_GROS_MICHEL && f_gros_extinct) add = false;
            if (k == J_CAVENDISH && !f_gros_extinct) add = false;
            if (has_pool && !pool_ok(k)) add = false;
            pool.push_back(add ? k : -1);
            size += add ? 1 : 0;
        }
        if (size == 0) { pool.clear(); pool.push_back(J_JOKER); }
        int key = exact_pick(pool, "Joker" + std::to_string(rr) + (legendary ? "" : ap) + (legendary ? "" : a));
        const JokerDef& d = JOKER_DEFS[key];
        bool eternal = false, perish = false, rental = false;
        if (sticker) {
            double poll = sg.pseudorandom((pack ? "packetper" : "etperpoll") + a);
            if (stake >= 3 && poll > 0.7) eternal = true;
            else if (stake >= 6 && 0.4 < poll && poll <= 0.7) perish = true;
            if (stake >= 7 && sg.pseudorandom((pack ? "packssjr" : "ssjr") + a) > 0.7) rental = true;
            if (no_stickers) eternal = perish = rental = false;
        }
        uint8_t edition = exact_poll_edition("edi" + ap + a, 1.0, false);
        j = Joker();
        j.key = (int16_t)key;
        j.edition = edition;
        j.base_cost = d.cost + EDITION_COST[edition];
        j.uid = next_joker_uid++;
        if (sticker) {
            if (eternal && d.eternal_ok) j.eternal = true;
            else if (perish && d.perish_ok && !j.eternal) { j.has_perish = true; j.perishable = 5; }
            if (rental) j.rental = true;
        }
        if (d.hooks & H_INIT) hook_init(j);
        if (scaled_shop) { double v; if (scaled_counter(j.key, scaled_rounds, v)) j.st.set(v); }
        j.has_cost = true;
        j.cost = j.rental ? 1 : price(j.base_cost);
        return true;
    }
    bool pool_ok(int k) const { return pool[k]; }
    // create_card('Tarot' / 'Planet' / 'Spectral', ...): src the stream name; soulable (packs): the Soul / Black Hole chance;
    // forced: a given consumable id (the Telescope's planet)
    Cons exact_consumable(int kind, uint64_t exclude, const char* src, bool soulable = false, int forced = C_NONE) {
        UsedSet u;
        exact_used(u, nullptr, exclude);
        std::string a = std::to_string(ante);
        const char* tn = kind == CK_TAROT ? "Tarot" : kind == CK_PLANET ? "Planet" : "Spectral";
        if (forced == C_NONE && soulable) {
            if ((kind == CK_TAROT || kind == CK_SPECTRAL) && !(u.c[C_SOUL] && !showman()))
                if (sg.pseudorandom(std::string("soul_") + tn + a) > 0.997) forced = C_SOUL;
            if ((kind == CK_PLANET || kind == CK_SPECTRAL) && !(u.c[C_BLACK_HOLE] && !showman()))
                if (sg.pseudorandom(std::string("soul_") + tn + a) > 0.997) forced = C_BLACK_HOLE;
        }
        if (forced != C_NONE) return make_cons(forced);
        std::vector<int> pool;
        int size = 0;
        if (kind == CK_TAROT) {
            for (int i = 0; i < N_POOL_TAROT; i++) {
                int id = POOL_TAROT[i];
                bool add = !(u.c[id] && !showman());
                pool.push_back(add ? id : -1);
                size += add ? 1 : 0;
            }
            if (size == 0) { pool.clear(); pool.push_back(C_STRENGTH); }
        } else if (kind == CK_PLANET) {
            for (int i = 0; i < N_POOL_PLANET; i++) {
                const PlanetPool& p = POOL_PLANET[i];
                bool add = !(u.c[p.cons] && !showman()) && (!p.softlock || hand_played[p.hand] > 0);
                pool.push_back(add ? p.cons : -1);
                size += add ? 1 : 0;
            }
            if (size == 0) { pool.clear(); pool.push_back(C_PLUTO); }
        } else {
            for (int i = 0; i < N_POOL_SPECTRAL; i++) {
                int id = POOL_SPECTRAL[i];
                bool add = !(u.c[id] && !showman()) && !CONS_DEFS[id].hidden;
                pool.push_back(add ? id : -1);
                size += add ? 1 : 0;
            }
            if (size == 0) { pool.clear(); pool.push_back(C_INCANTATION); }
        }
        return make_cons(exact_pick(pool, std::string(tn) + src + a));
    }
    ShopItem exact_shop_card(const bool* exj, uint64_t exc) {
        double tw = has_voucher(V_TAROT_TYCOON) ? 32 : (has_voucher(V_TAROT_MERCHANT) ? 9.6 : 4);
        double pw = has_voucher(V_PLANET_TYCOON) ? 32 : (has_voucher(V_PLANET_MERCHANT) ? 9.6 : 4);
        double sw = deck_type == D_GHOST ? 2 : 0;
        double cw = has_voucher(V_MAGIC_TRICK) ? 4 : 0;
        double total = 20.0;
        total += tw; total += pw; total += cw; total += sw;
        bool ill = has_voucher(V_ILLUSION);                // with Illusion every shop card draws the 'illusion' stream
        double ill_roll = ill ? sg.pseudorandom("illusion") : 0.0;
        double polled = sg.pseudorandom("cdt" + std::to_string(ante)) * total;
        double rates[5] = {20.0, tw, pw, cw, sw};          // Joker, Tarot, Planet, Base, Spectral
        double check = 0;
        int typ = -1;
        for (int t = 0; t < 5; t++) {
            if (polled > check && polled <= check + rates[t]) { typ = t; break; }
            check += rates[t];
        }
        ShopItem it;
        if (typ == 0) {
            exact_joker(it.joker, 0, true, exj, "sho");
            it.kind = SK_JOKER; it.key = it.joker.key; it.cost = it.joker.cost;
        } else if (typ == 1) {
            Cons c = exact_consumable(CK_TAROT, exc, "sho");
            it.kind = SK_TAROT; it.key = c.id; it.cost = price(3);
        } else if (typ == 2) {
            Cons c = exact_consumable(CK_PLANET, exc, "sho");
            it.kind = SK_PLANET; it.key = c.id; it.cost = has(J_ASTRONOMER) ? 0 : price(3);
        } else if (typ == 4) {
            Cons c = exact_consumable(CK_SPECTRAL, exc, "sho");
            it.kind = SK_SPECTRAL; it.key = c.id; it.cost = price(4);
        } else {
            std::string a = std::to_string(ante);
            uint8_t enh = E_NONE;
            if (ill && ill_roll > 0.6) enh = enh8((size_t)(sg.pseudorandom("Enhancedsho" + a, 1, 8) - 1));
            int fk = sg.pseudorandom("frontsho" + a, 1, 52) - 1;
            int id = new_card(P_CARD_RANK[fk], P_CARD_SUIT[fk], enh);
            if (ill && sg.pseudorandom("illusion") > 0.8) {
                double ep = sg.pseudorandom("illusion");
                cards[id].ed = ep > 1 - 0.15 ? ED_POLY : ep > 0.5 ? ED_HOLO : ED_FOIL;
            }
            it.kind = SK_CARD; it.key = 0; it.card = id; it.cost = price(1 + EDITION_COST[cards[id].ed]);
        }
        return it;
    }
    ShopItem exact_pack() {                                // get_pack('shop_pack')
        int idx;
        if (!first_shop_buffoon) {
            first_shop_buffoon = true;
            sg.rng.randint(1, 2);                          // the game's unseeded math.random(1, 2) (it moves the global state)
            idx = PK_BUFFOON * 3 + PS_NORMAL;
        } else {
            double cume = 0;
            for (int i = 0; i < N_POOL_BOOSTER; i++) cume += POOL_BOOSTER[i].weight;
            double poll = sg.pseudorandom("shop_pack" + std::to_string(ante)) * cume;
            double it = 0;
            idx = POOL_BOOSTER[N_POOL_BOOSTER - 1].pack;
            for (int i = 0; i < N_POOL_BOOSTER; i++) {
                double w = POOL_BOOSTER[i].weight;
                it += w;
                if (it >= poll && it - w <= poll) { idx = POOL_BOOSTER[i].pack; break; }
            }
        }
        ShopItem s;
        s.kind = SK_PACK; s.key = idx;
        s.cost = (PACK_DEFS[idx].kind == PK_CELESTIAL && has(J_ASTRONOMER)) ? 0 : price(PACK_DEFS[idx].cost);
        return s;
    }
    int exact_voucher(bool from_tag) {                     // get_next_voucher_key()
        std::vector<int> pool;
        int size = 0;
        for (int i = 0; i < N_POOL_VOUCHER; i++) {
            int v = POOL_VOUCHER[i];
            bool add = !has_voucher(v) && (VOUCHER_DEFS[v].prereq < 0 || has_voucher(VOUCHER_DEFS[v].prereq));
            for (const ShopItem& s : shop_vouchers) if (s.key == v) add = false;
            pool.push_back(add ? v : -1);
            size += add ? 1 : 0;
        }
        if (size == 0) { pool.clear(); pool.push_back(V_BLANK); }
        return exact_pick(pool, from_tag ? std::string("Voucher_fromtag") : "Voucher" + std::to_string(ante));
    }
    PackItem exact_standard_card() {
        std::string a = std::to_string(ante);
        bool enhanced = sg.pseudorandom("stdset" + a) > 0.6;
        uint8_t enh = E_NONE;
        if (enhanced) enh = enh8((size_t)(sg.pseudorandom("Enhanced" + std::string("sta") + a, 1, 8) - 1));
        int fk = sg.pseudorandom("front" + std::string("sta") + a, 1, 52) - 1;
        int id = new_card(P_CARD_RANK[fk], P_CARD_SUIT[fk], enh);
        cards[id].ed = exact_poll_edition("standard_edition" + a, 2.0, true);
        if (sg.pseudorandom("stdseal" + a) > 1 - 0.02 * 10) {
            double t = sg.pseudorandom("stdsealtype" + a);
            cards[id].seal = t > 0.75 ? S_RED : t > 0.5 ? S_BLUE : t > 0.25 ? S_GOLD : S_PURPLE;
        }
        PackItem it;
        it.kind = PI_CARD; it.card = id;
        return it;
    }
    void exact_open_pack(int kind, int shown) {
        for (int i = 0; i < shown; i++) {
            PackItem it;
            if (kind == PK_BUFFOON) {
                it.kind = PI_JOKER;
                exact_joker(it.joker, 0, true, nullptr, "buf", true);
            } else if (kind == PK_ARCANA) {
                it.kind = PI_CONS;
                if (has_voucher(V_OMEN_GLOBE) && sg.pseudorandom("omen_globe") > 0.8) it.cons = exact_consumable(CK_SPECTRAL, 0ull, "ar2", true);
                else it.cons = exact_consumable(CK_TAROT, 0ull, "ar1", true);
            } else if (kind == PK_CELESTIAL) {
                int forced = C_NONE;
                if (has_voucher(V_TELESCOPE) && i == 0) {
                    int most = 0;
                    for (int h = 1; h < N_HANDS; h++) if (hand_played[h] > hand_played[most]) most = h;
                    if (hand_played[most] > 0) forced = HAND_TO_PLANET[most];
                }
                it.kind = PI_CONS;
                it.cons = exact_consumable(CK_PLANET, 0ull, "pl1", true, forced);
            } else if (kind == PK_SPECTRAL) {
                it.kind = PI_CONS;
                it.cons = exact_consumable(CK_SPECTRAL, 0ull, "spe", true);
            } else {
                it = exact_standard_card();
            }
            pack_cards.push_back(it);
        }
    }

    // ---------------------------------------------------------------- shop
    // A random joker (Game.random_joker). rarity 0: rolled. exclude: joker keys (bitmap over J_COUNT) or null.
    // Returns false when allow_empty and no joker of the rarity is left. src: the stream name in exact-seed mode.
    bool random_joker(Joker& j, int rarity, bool sticker, const bool* exclude, bool allow_empty, const char* src = "sho") {
        if (exact) return exact_joker(j, rarity, sticker, exclude, src);
        if (rarity == 0) {
            double r = rng.random();
            rarity = r > 0.95 ? 3 : (r > 0.7 ? 2 : 1);
        }
        bool owned[J_COUNT];
        for (int k = 0; k < J_COUNT; k++) owned[k] = false;
        if (!showman()) {
            for (const Joker& o : jokers) owned[o.key] = true;
            if (exclude) for (int k = 0; k < J_COUNT; k++) if (exclude[k]) owned[k] = true;
        }
        std::vector<int> pl;
        for (int k = 0; k < J_COUNT; k++)
            if (JOKER_DEFS[k].rarity == rarity && !owned[k] && (!JOKER_DEFS[k].not_in_pool || f_gros_extinct))
                pl.push_back(k);
        if (f_gros_extinct) {
            auto it = std::find(pl.begin(), pl.end(), (int)J_GROS_MICHEL);
            if (it != pl.end()) pl.erase(it);
        }
        if (has_pool) {
            std::vector<int> allowed;
            for (int k : pl) if (pool[k]) allowed.push_back(k);
            if (!allowed.empty()) pl = allowed;
            else {
                pl.clear();
                for (int q = 0; q < J_COUNT; q++) {
                    int k = JOKERS_SORTED[q];
                    if (pool[k] && !owned[k] && JOKER_DEFS[k].rarity < 4) pl.push_back(k);
                }
            }
        }
        if (pl.empty()) {
            if (allow_empty) return false;
            pl.push_back(J_JOKER);
        }
        int key = pl[rng.choice_index(pl.size())];
        const JokerDef& d = JOKER_DEFS[key];
        int ed_mult = has_voucher(V_GLOW_UP) ? 4 : (has_voucher(V_HONE) ? 2 : 1);
        double r = rng.random();
        uint8_t edition = ED_NONE;
        if (r > 1 - 0.003) edition = ED_NEG;
        else if (r > 1 - 0.006 * ed_mult) edition = ED_POLY;
        else if (r > 1 - 0.02 * ed_mult) edition = ED_HOLO;
        else if (r > 1 - 0.04 * ed_mult) edition = ED_FOIL;
        j = Joker();
        j.key = (int16_t)key;
        j.edition = edition;
        j.base_cost = d.cost + EDITION_COST[edition];
        j.uid = next_joker_uid++;
        if (sticker) {
            double r1 = rng.random();
            if (stake >= 3 && r1 > 0.7 && d.eternal_ok) j.eternal = true;
            else if (stake >= 6 && 0.4 < r1 && r1 <= 0.7 && d.perish_ok) { j.has_perish = true; j.perishable = 5; }
            if (stake >= 7 && rng.random() > 0.7) j.rental = true;
            if (no_stickers) { j.eternal = false; j.has_perish = false; j.perishable = 0; j.rental = false; }
        }
        if (d.hooks & H_INIT) hook_init(j);
        if (scaled_shop) { double v; if (scaled_counter(j.key, scaled_rounds, v)) j.st.set(v); }
        j.has_cost = true;
        j.cost = j.rental ? 1 : price(j.base_cost);
        return true;
    }
    // exclude: bitmask over consumable ids
    Cons random_consumable(int kind, uint64_t exclude, int src = 0) {
        if (exact) return exact_consumable(kind, exclude, SRC_KEYS[src]);
        int opts[C_COUNT], n = 0;
        if (kind == CK_PLANET) {
            for (int p = 0; p < N_PLANETS; p++) {
                int h = CONS_DEFS[p].planet_hand;
                if (!is_secret_hand(h) || hand_played[h] > 0) opts[n++] = p;
            }
        } else if (kind == CK_TAROT) {
            for (int t = 0; t < N_TAROTS; t++) opts[n++] = N_PLANETS + t;
        } else {
            for (int s = 0; s < N_SPECTRALS; s++) {
                int id = N_PLANETS + N_TAROTS + s;
                if (!CONS_DEFS[id].hidden) opts[n++] = id;
            }
        }
        if (!showman()) {
            int f[C_COUNT], m = 0;
            for (int i = 0; i < n; i++) if (!((exclude >> opts[i]) & 1ull)) f[m++] = opts[i];
            if (m > 0) { n = m; for (int i = 0; i < m; i++) opts[i] = f[i]; }
        }
        return make_cons(opts[rng.choice_index(n)]);
    }
    // one shop card; exj / exc: joker keys and consumable ids already in this shop
    ShopItem shop_card(const bool* exj, uint64_t exc) {
        if (menu_slots > 2) {                                                 // the big menu: jokers only
            ShopItem it;
            random_joker(it.joker, 0, true, exj, false);
            it.kind = SK_JOKER; it.key = it.joker.key; it.cost = it.joker.cost;
            return it;
        }
        if (exact) return exact_shop_card(exj, exc);
        double tw = has_voucher(V_TAROT_TYCOON) ? 32 : (has_voucher(V_TAROT_MERCHANT) ? 9.6 : 4);
        double pw = has_voucher(V_PLANET_TYCOON) ? 32 : (has_voucher(V_PLANET_MERCHANT) ? 9.6 : 4);
        double sw = deck_type == D_GHOST ? 2 : 0;
        double cw = has_voucher(V_MAGIC_TRICK) ? 4 : 0;
        double r = rng.random() * (20 + tw + pw + sw + cw);
        ShopItem it;
        if (r < 20) {
            random_joker(it.joker, 0, true, exj, false);
            it.kind = SK_JOKER; it.key = it.joker.key; it.cost = it.joker.cost;
            return it;
        }
        r -= 20;
        if (r < tw) {
            Cons c = random_consumable(CK_TAROT, exc);
            it.kind = SK_TAROT; it.key = c.id; it.cost = price(3);
            return it;
        }
        r -= tw;
        if (r < pw) {
            Cons c = random_consumable(CK_PLANET, exc);
            it.kind = SK_PLANET; it.key = c.id; it.cost = has(J_ASTRONOMER) ? 0 : price(3);
            return it;
        }
        r -= pw;
        if (r < sw) {
            Cons c = random_consumable(CK_SPECTRAL, exc);
            it.kind = SK_SPECTRAL; it.key = c.id; it.cost = price(4);
            return it;
        }
        int id = shop_playing_card();
        it.kind = SK_CARD; it.key = 0; it.card = id; it.cost = price(1 + EDITION_COST[cards[id].ed]);
        return it;
    }
    uint8_t poll_edition(double mod = 1.0, bool no_neg = false, bool guaranteed = false) {
        int rate = has_voucher(V_GLOW_UP) ? 4 : (has_voucher(V_HONE) ? 2 : 1);
        double r = rng.random();
        double neg, poly, holo, foil;
        if (guaranteed) { neg = 0.003 * 25; poly = 0.006 * 25; holo = 0.02 * 25; foil = 0.04 * 25; }
        else { neg = 0.003 * mod; poly = 0.006 * rate * mod; holo = 0.02 * rate * mod; foil = 0.04 * rate * mod; }
        if (r > 1 - neg && !no_neg) return ED_NEG;
        if (r > 1 - poly) return ED_POLY;
        if (r > 1 - holo) return ED_HOLO;
        if (r > 1 - foil) return ED_FOIL;
        return ED_NONE;
    }
    static uint8_t enh8(size_t i) {                    // BONUS, MULT, WILD, GLASS, STEEL, STONE, GOLD, LUCKY
        static const uint8_t e[8] = {E_BONUS, E_MULT, E_WILD, E_GLASS, E_STEEL, E_STONE, E_GOLD, E_LUCKY};
        return e[i];
    }
    int shop_playing_card() {
        int r = (int)rng.randrange(2, 15);
        int s = (int)rng.randrange(4);
        int id = new_card(r, s);
        if (has_voucher(V_ILLUSION)) {
            if (rng.random() > 0.6) cards[id].enh = enh8(rng.choice_index(8));
            if (rng.random() > 0.8) cards[id].ed = poll_edition(1.0, true, true);
        }
        return id;
    }
    int random_playing_card(bool enhanced) {
        int r = (int)rng.randrange(2, 15);
        int s = (int)rng.randrange(4);
        int id = new_card(r, s);
        if (enhanced) {
            if (rng.random() < 0.4) cards[id].enh = enh8(rng.choice_index(8));
            cards[id].ed = poll_edition(2, true);
            if (rng.random() < 0.2) {
                static const uint8_t seals[4] = {S_RED, S_BLUE, S_GOLD, S_PURPLE};
                cards[id].seal = seals[rng.choice_index(4)];
            }
        }
        return id;
    }
    ShopItem random_pack() {
        if (exact) return exact_pack();
        std::vector<double> w(N_PACKS);
        for (int i = 0; i < N_PACKS; i++) w[i] = PACK_DEFS[i].weight * pack_w_mult[(int)PACK_DEFS[i].kind];
        int k = (int)rng.choices_indices(w, 1)[0];
        ShopItem it;
        it.kind = SK_PACK; it.key = k;
        it.cost = (PACK_DEFS[k].kind == PK_CELESTIAL && has(J_ASTRONOMER)) ? 0 : price(PACK_DEFS[k].cost);
        return it;
    }
    // a shop joker of the given key, with the shop's edition, sticker and price rolls (random_joker's second half)
    ShopItem catalogue_item(int key) {
        const JokerDef& d = JOKER_DEFS[key];
        int ed_mult = has_voucher(V_GLOW_UP) ? 4 : (has_voucher(V_HONE) ? 2 : 1);
        double r = rng.random();
        uint8_t edition = ED_NONE;
        if (r > 1 - 0.003) edition = ED_NEG;
        else if (r > 1 - 0.006 * ed_mult) edition = ED_POLY;
        else if (r > 1 - 0.02 * ed_mult) edition = ED_HOLO;
        else if (r > 1 - 0.04 * ed_mult) edition = ED_FOIL;
        Joker j;
        j.key = (int16_t)key;
        j.edition = edition;
        j.base_cost = d.cost + EDITION_COST[edition];
        j.uid = next_joker_uid++;
        double r1 = rng.random();
        if (stake >= 3 && r1 > 0.7 && d.eternal_ok) j.eternal = true;
        else if (stake >= 6 && 0.4 < r1 && r1 <= 0.7 && d.perish_ok) { j.has_perish = true; j.perishable = 5; }
        if (stake >= 7 && rng.random() > 0.7) j.rental = true;
        if (no_stickers) { j.eternal = false; j.has_perish = false; j.perishable = 0; j.rental = false; }
        if (d.hooks & H_INIT) hook_init(j);
        j.has_cost = true;
        j.cost = j.rental ? 1 : price(j.base_cost);
        ShopItem it;
        it.joker = j; it.kind = SK_JOKER; it.key = j.key; it.cost = j.cost;
        return it;
    }
    // the catalogue's page cat_page after the items `first` (tag jokers): new_order shuffles a fresh catalogue (every eligible non-legendary joker not held)
    std::vector<ShopItem> catalogue_page(const std::vector<ShopItem>& first, bool new_order) {
        if (new_order || cat_order.empty()) {
            cat_order.clear();
            bool owned[J_COUNT] = {false};
            if (!showman()) for (const Joker& o : jokers) owned[o.key] = true;
            for (int k = 0; k < J_COUNT; k++) {
                const JokerDef& d = JOKER_DEFS[k];
                if (d.rarity < 1 || d.rarity > 3 || owned[k]) continue;
                if (d.not_in_pool && !f_gros_extinct) continue;
                if (f_gros_extinct && k == J_GROS_MICHEL) continue;
                if (has_pool && !pool[k]) continue;
                cat_order.push_back((int16_t)k);
            }
            for (int i = (int)cat_order.size() - 1; i > 0; i--) std::swap(cat_order[i], cat_order[rng.choice_index((size_t)(i + 1))]);
            cat_page = 0;
        }
        std::vector<ShopItem> out = first;
        const int per = MAX_SHOP - (int)first.size();
        const int n = (int)cat_order.size();
        if (n == 0 || per <= 0) return out;
        const int pages = (n + per - 1) / per;
        const int pg = ((cat_page % pages) + pages) % pages;
        for (int i = pg * per; i < std::min(n, (pg + 1) * per); i++) out.push_back(catalogue_item(cat_order[i]));
        return out;
    }
    int shop_slots() const {
        if (catalogue) return MAX_SHOP;
        if (menu_slots > 2) return std::min(menu_slots, MAX_SHOP);
        return 2 + (has_voucher(V_OVERSTOCK_NORM) ? 1 : 0) + (has_voucher(V_OVERSTOCK_PLUS) ? 1 : 0);
    }
    static void exclusions(const std::vector<ShopItem>& items, bool* exj, uint64_t& exc) {
        for (int k = 0; k < J_COUNT; k++) exj[k] = false;
        exc = 0;
        for (const ShopItem& it : items) {
            if (it.kind == SK_JOKER) exj[it.key] = true;
            else if (it.kind == SK_TAROT || it.kind == SK_PLANET || it.kind == SK_SPECTRAL) exc |= 1ull << it.key;
        }
    }
    std::vector<ShopItem> shop_cards(int n, const std::vector<ShopItem>* first = nullptr) {
        std::vector<ShopItem> out;
        if (first) out = *first;
        std::vector<ShopItem> saved;
        if (exact) saved.swap(shop);                     // a reroll removes the old cards first: they no longer count as used
        while ((int)out.size() < n) {
            bool exj[J_COUNT];
            uint64_t exc;
            exclusions(out, exj, exc);
            out.push_back(shop_card(exj, exc));
        }
        if (exact) shop.swap(saved);
        return out;
    }
    void apply_edition_tags() {
        std::vector<int8_t> rest;
        for (int8_t t : pending_tags) {
            int ed = t == T_FOIL ? ED_FOIL : t == T_HOLO ? ED_HOLO : t == T_POLYCHROME ? ED_POLY : t == T_NEGATIVE ? ED_NEG : -1;
            ShopItem* it = nullptr;
            if (ed >= 0)
                for (ShopItem& s : shop)
                    if (s.kind == SK_JOKER && s.joker.edition == ED_NONE) { it = &s; break; }
            if (!it) { rest.push_back(t); continue; }
            it->joker.edition = (uint8_t)ed;
            it->joker.base_cost += EDITION_COST[ed];
            it->cost = 0;
            it->joker.has_cost = true;
            it->joker.cost = 0;
        }
        pending_tags = rest;
    }
    std::vector<int> voucher_options() const {
        uint32_t shown = 0;
        for (const ShopItem& it : shop_vouchers) shown |= 1u << it.key;
        std::vector<int> out;
        for (int v = 0; v < V_COUNT; v++) {
            int pre = VOUCHER_DEFS[v].prereq;
            if (!has_voucher(v) && !((shown >> v) & 1u) && (pre < 0 || has_voucher(pre))) out.push_back(v);
        }
        return out;
    }
    void open_shop() {
        state = ST_SHOP;
        shops_seen += 1;
        pack_restocks_used = 0;
        if (exact) {                                     // cashing out shuffles the deck ('cashout' + ante): packs draw from it
            deck = full_deck;
            std::stable_sort(deck.begin(), deck.end());
            sg.shuffle(deck, "cashout" + std::to_string(ante));
        }
        reroll_cost = std::max(0, 5 - 2 * (has_voucher(V_REROLL_SURPLUS) ? 1 : 0) - 2 * (has_voucher(V_REROLL_GLUT) ? 1 : 0));
        free_rerolls = count(J_CHAOS);
        std::vector<ShopItem> tagged;
        std::vector<int8_t> rest;
        for (int8_t t : pending_tags) {
            if ((t == T_UNCOMMON || t == T_RARE) && (int)tagged.size() < shop_slots()) {
                bool exj[J_COUNT];
                uint64_t exc;
                exclusions(tagged, exj, exc);
                ShopItem it;
                if (random_joker(it.joker, t == T_UNCOMMON ? 2 : 3, true, exj, true, t == T_UNCOMMON ? "uta" : "rta")) {
                    it.joker.has_cost = true;
                    it.joker.cost = 0;
                    it.kind = SK_JOKER; it.key = it.joker.key; it.cost = 0;
                    tagged.push_back(it);
                }
            } else {
                rest.push_back(t);
            }
        }
        pending_tags = rest;
        shop = (catalogue && !exact) ? catalogue_page(tagged, true) : shop_cards(shop_slots(), &tagged);
        shop_packs.clear();
        if (shops_seen == 1 && !exact) {
            ShopItem b;
            b.kind = SK_PACK;
            b.key = 9;                                      // p_buffoon_normal
            b.cost = price(4);
            shop_packs.push_back(b);
            shop_packs.push_back(random_pack());
        } else {
            ShopItem a = random_pack();
            ShopItem b2 = random_pack();
            shop_packs.push_back(a);
            shop_packs.push_back(b2);
        }
        if (!has_ante_voucher && !(f_voucher_bought_ante == ante)) {
            shop_vouchers.clear();
            std::vector<int> opts = voucher_options();
            if (!opts.empty()) {
                int v = exact ? exact_voucher(false) : opts[rng.choice_index(opts.size())];
                ante_voucher = ShopItem();
                ante_voucher.kind = SK_VOUCHER; ante_voucher.key = v; ante_voucher.cost = price(VOUCHER_COST);
                ante_voucher.is_ante = true;
                has_ante_voucher = true;
            }
        }
        shop_vouchers.clear();
        if (has_ante_voucher) shop_vouchers.push_back(ante_voucher);
        apply_edition_tags();
        std::vector<int8_t> tags = pending_tags;
        pending_tags.clear();
        for (int8_t t : tags) {
            if (t == T_FOIL || t == T_HOLO || t == T_POLYCHROME || t == T_NEGATIVE) {
                pending_tags.push_back(t);
            } else if (t == T_COUPON) {
                for (ShopItem& it : shop) { it.cost = 0; if (it.kind == SK_JOKER) { it.joker.has_cost = true; it.joker.cost = 0; } }
                for (ShopItem& it : shop_packs) it.cost = 0;
            } else if (t == T_D_SIX) {
                reroll_cost = 0;
            } else if (t == T_VOUCHER) {
                std::vector<int> opts = voucher_options();
                if (!opts.empty() && (int)shop_vouchers.size() < MAX_VOUCHERS) {
                    int v = exact ? exact_voucher(true) : opts[rng.choice_index(opts.size())];
                    ShopItem it;
                    it.kind = SK_VOUCHER; it.key = v; it.cost = price(VOUCHER_COST);
                    shop_vouchers.push_back(it);
                }
            } else {
                pending_tags.push_back(t);
            }
        }
        if ((int)shop.size() > MAX_SHOP) shop.resize(MAX_SHOP);
    }
    bool can_buy(const ShopItem& it) const {
        if (!can_afford(it.cost)) return false;
        if (it.kind == SK_JOKER) return it.joker.edition == ED_NEG || (int)jokers.size() < joker_slots;
        if (it.kind == SK_TAROT || it.kind == SK_PLANET || it.kind == SK_SPECTRAL) return cons_room();
        return true;
    }
    void buy_card(int i) {
        if (i < 0 || i >= (int)shop.size() || !can_buy(shop[i])) throw std::runtime_error("buy_card: not allowed");
        ShopItem it = shop[i];
        money -= it.cost;
        shop.erase(shop.begin() + i);
        if (it.kind == SK_JOKER) {
            it.joker.has_cost = true;
            it.joker.cost = it.cost;
            add_joker(it.joker);
        } else if (it.kind == SK_CARD) {
            add_card(it.card);
        } else {
            consumables.push_back(make_cons(it.key));
        }
    }
    void buy_pack(int i) {
        if (i < 0 || i >= (int)shop_packs.size() || !can_afford(shop_packs[i].cost))
            throw std::runtime_error("buy_pack: not allowed");
        ShopItem it = shop_packs[i];
        money -= it.cost;
        shop_packs.erase(shop_packs.begin() + i);
        if (pack_restock > 0 && pack_restocks_used < pack_restock) {         // the sandbox: another pack takes its place
            shop_packs.push_back(random_pack());
            pack_restocks_used += 1;
        }
        open_pack(PACK_DEFS[it.key].kind, PACK_DEFS[it.key].size, ST_SHOP);
    }
    void buy_voucher(int i = 0) {
        if (i < 0 || i >= (int)shop_vouchers.size() || !can_afford(shop_vouchers[i].cost))
            throw std::runtime_error("buy_voucher: not allowed");
        ShopItem it = shop_vouchers[i];
        money -= it.cost;
        shop_vouchers.erase(shop_vouchers.begin() + i);
        if (it.is_ante) {
            has_ante_voucher = false;
            f_voucher_bought_ante = ante;
        }
        redeem_voucher(it.key);
    }
    void redeem_voucher(int v) {
        vouchers |= 1u << v;
        if (v == V_CRYSTAL_BALL) consumable_slots += 1;
        else if (v == V_ANTIMATTER) joker_slots += 1;
        else if (v == V_HIEROGLYPH || v == V_PETROGLYPH) ante = std::max(1, ante - 1);
        else if (v == V_OVERSTOCK_NORM || v == V_OVERSTOCK_PLUS) {
            shop.push_back(shop_card(nullptr, 0ull));
            if ((int)shop.size() > MAX_SHOP) shop.resize(MAX_SHOP);
        }
    }
    void reroll() {
        if (catalogue && !exact) {                       // the catalogue: the next page, free; not a reroll for Flash Card or anything that counts them
            cat_page += 1;
            shop = catalogue_page({}, false);
            apply_edition_tags();
            return;
        }
        if (free_rerolls > 0) {
            free_rerolls -= 1;
        } else {
            if (!can_afford(reroll_cost)) throw std::runtime_error("reroll: cannot afford");
            money -= reroll_cost;
            reroll_cost += 1;
        }
        rerolls_total += 1;
        if (flash_cap <= 0 || rerolls_total <= flash_cap)
            for (Joker& j : jokers) if (j.key == J_FLASH) j.st.set(j.st.get(0) + 2);
        shop = shop_cards(shop_slots());
        apply_edition_tags();
    }
    void leave_shop() {
        int pk = count(J_PERKEO);
        for (int k = 0; k < pk; k++)
            if (!consumables.empty()) {
                std::vector<int> order;
                for (size_t q = 0; q < consumables.size(); q++) order.push_back((int)q);
                Cons src = consumables[exact ? xpick("perkeo", order) : rng.choice_index(consumables.size())];
                consumables.push_back(make_cons(src.id, true));
            }
        auto it = std::find(pending_tags.begin(), pending_tags.end(), (int8_t)T_JUGGLE);
        if (it != pending_tags.end()) { pending_tags.erase(it); juggle = 3; }
        state = ST_BLIND_SELECT;
    }

    // ---------------------------------------------------------------- jokers / consumables
    void add_joker(Joker j) {
        if (j.uid == 0) j.uid = next_joker_uid++;
        if (j.edition == ED_NEG) joker_slots += 1;
        if ((j.d().hooks & H_INIT) && j.st.empty()) hook_init(j);
        jokers.push_back(j);
    }
    void destroy_joker(int uid) {
        int i = find_joker(uid);
        if (i < 0) return;
        graveyard.push_back(jokers[i]);
        bool neg = jokers[i].edition == ED_NEG;
        jokers.erase(jokers.begin() + i);
        if (neg) joker_slots -= 1;
    }
    void on_sold() {
        for (Joker& o : jokers) if (o.key == J_CAMPFIRE && !o.debuffed) o.st.set(o.st.get(1.0) + 0.25);
    }
    void sell_joker(int i) {
        if (i < 0 || i >= (int)jokers.size() || jokers[i].eternal) throw std::runtime_error("sell_joker: not allowed");
        Joker j = jokers[i];
        money += j.sell_value();
        destroy_joker(j.uid);
        if ((j.d().hooks & H_ON_SELL) && !j.debuffed) hook_on_sell(j);
        on_sold();
        if (f_verdant) {
            f_verdant = false;
            if (state == ST_SELECTING_HAND) apply_debuffs();
        }
    }
    void sell_consumable(int i) {
        if (i < 0 || i >= (int)consumables.size()) throw std::runtime_error("sell_consumable: bad index");
        Cons c = consumables[i];
        consumables.erase(consumables.begin() + i);
        money += c.sell_value();
        on_sold();
    }
    // is the order of the jokers the player's to choose? (env.py swaps_matter)
    bool swaps_matter() const {
        if (ORDER_MODE == 1) {
            for (const Joker& j : jokers) if (j.key == J_BLUEPRINT || j.key == J_BRAINSTORM || j.key == J_CEREMONIAL) return true;
            return false;
        }
        for (const Joker& j : jokers) if (JOKER_SWAP_ROLE[j.key] == 3) return true;
        return false;
    }
    // the jokers in scoring order: additive mult first, multipliers last, the rest as they are (env.py order_jokers)
    void order_jokers() {
        if (jokers.size() < 2 || swaps_matter()) return;
        for (const Joker& j : jokers) if (j.hidden) return;
        std::stable_sort(jokers.begin(), jokers.end(), [](const Joker& a, const Joker& b) {
            return order_rank(a) < order_rank(b);
        });
    }
    static int order_rank(const Joker& j) {
        if (ORDER_MODE == 1) {                               // chips and the rest first, +mult next, xMult last
            int r = JOKER_SWAP_ROLE[j.key] == 3 ? 1 : JOKER_SWAP_ROLE[j.key];
            bool add = r == 1 || j.edition == ED_HOLO;
            bool mul = r == 2 || j.edition == ED_POLY;
            return mul && !add ? 2 : add ? 1 : 0;
        }
        bool add = JOKER_SWAP_ROLE[j.key] == 1 || j.edition == ED_HOLO;
        bool mul = JOKER_SWAP_ROLE[j.key] == 2 || j.edition == ED_POLY;
        return add && !mul ? 0 : mul && !add ? 2 : 1;
    }
    void swap_jokers(int i) {
        if (i < 0 || i + 1 >= (int)jokers.size()) throw std::runtime_error("swap_jokers: bad index");
        std::swap(jokers[i], jokers[i + 1]);
    }
    void destroy_cards(const std::vector<int>& ids) {
        bool par = has(J_PAREIDOLIA);
        int glass = 0, faces = 0;
        for (int id : ids) {
            if (!in_pile(full_deck, id) && !in_pile(pack_hand, id)) continue;
            glass += cards[id].enh == E_GLASS ? 1 : 0;
            faces += cards[id].is_face(par) ? 1 : 0;
            remove_from(full_deck, id);
            remove_from(hand, id);
            remove_from(deck, id);
            remove_from(discard_pile, id);
            remove_from(pack_hand, id);
        }
        for (Joker& j : jokers) {
            if (j.key == J_GLASS && glass) j.st.set(j.st.get(1.0) + 0.75 * glass);
            if (j.key == J_CAINO && faces) j.st.set(j.st.get(1.0) + 1.0 * faces);
        }
    }
    void add_card(int id, bool to_hand = false, bool grow = true) {
        full_deck.push_back(id);
        for (Joker& j : jokers)
            if (grow && j.key == J_HOLOGRAM && !j.debuffed) j.st.set(j.st.get(1.0) + 0.25);
        if (to_hand && state == ST_SELECTING_HAND && (int)hand.size() < MAX_HAND) {
            hand.push_back(id);
            sort_hand(hand);
        } else if (state == ST_SELECTING_HAND) {
            deck.insert(deck.begin(), id);
        }
    }
    bool any_plain_joker() const {
        for (const Joker& j : jokers) if (j.edition == ED_NONE) return true;
        return false;
    }
    bool consumable_usable(const Cons& c, const std::vector<int>& cs, bool from_slot = true) const {
        if (c.kind() == CK_PLANET) return true;
        int need = CONS_DEFS[c.id].need;
        int n = c.id;
        if (need && cs.empty()) return false;
        if (n == C_DEATH && cs.size() < 2) return false;
        if ((n == C_JUDGEMENT || n == C_WRAITH || n == C_SOUL) && (int)jokers.size() >= joker_slots) return false;
        if ((n == C_HIGH_PRIESTESS || n == C_EMPEROR || n == C_FOOL) && !from_slot && !cons_room()) return false;
        if (n == C_FOOL && (last_consumable == C_NONE || last_consumable == C_FOOL)) return false;
        if (n == C_ANKH && jokers.empty()) return false;
        if ((n == C_HEX || n == C_ECTOPLASM || n == C_WHEEL_OF_FORTUNE) && !any_plain_joker()) return false;
        if (n == C_ANKH && (int)jokers.size() >= joker_slots) return false;
        if ((n == C_FAMILIAR || n == C_GRIM || n == C_INCANTATION || n == C_IMMOLATE || n == C_SIGIL || n == C_OUIJA)
            && cs.size() < 2)
            return false;
        if (n == C_AURA) {
            bool any = false;
            for (int id : cs) if (cards[id].ed == ED_NONE) any = true;
            if (!any) return false;
        }
        return true;
    }
    void use_consumable(int i, bool choose_targets = false) {
        if (i < 0 || i >= (int)consumables.size()) throw std::runtime_error("use_consumable: bad index");
        Cons c = consumables[i];
        consumables.erase(consumables.begin() + i);
        bool in_round = state == ST_SELECTING_HAND;
        if (choose_targets && CONS_DEFS[c.id].targeted && in_round && !hand.empty()) {
            targeting = true; targeting_cons = c; targeting_from_pack = false;
            return;
        }
        apply_consumable(c, in_round ? 0 : 2, nullptr);
    }

    // ---------------------------------------------------------------- choosing target cards
    static void target_range(const Cons& c, int& lo, int& hi) {
        if (c.id == C_DEATH) { lo = 2; hi = 2; return; }
        lo = 1; hi = CONS_DEFS[c.id].need;
    }
    const std::vector<int>& target_cards() const { return targeting_from_pack ? pack_hand : hand; }
    void apply_targets(const std::vector<int>& positions) {
        if (!targeting) throw std::runtime_error("no consumable is waiting for targets");
        const std::vector<int>& cs = target_cards();
        int lo, hi;
        target_range(targeting_cons, lo, hi);
        if ((int)positions.size() < lo || (int)positions.size() > hi) throw std::runtime_error("apply_targets: count");
        std::vector<int> pos = positions;
        std::sort(pos.begin(), pos.end());
        for (size_t k = 0; k < pos.size(); k++) {
            if (pos[k] < 0 || pos[k] >= (int)cs.size()) throw std::runtime_error("apply_targets: position");
            if (k > 0 && pos[k] == pos[k - 1]) throw std::runtime_error("apply_targets: duplicate");
        }
        std::vector<int> targets;
        for (int p : pos) targets.push_back(cs[p]);
        bool from_pack = targeting_from_pack;
        Cons c = targeting_cons;
        targeting = false;
        apply_consumable(c, from_pack ? 1 : 0, &targets);
        if (from_pack) finish_pack_pick();
    }
    double card_value(const Card& c) const {
        static const int VE[10] = {0, 8, 10, 6, 20, 15, -5, 5, 10, 0};
        static const int VD[5] = {0, 8, 12, 25, 0};
        static const int VS[5] = {0, 15, 5, 8, 5};
        double v = c.chip_value() + c.rank * 0.1;
        v += VE[c.enh];
        v += VD[c.ed];
        v += VS[c.seal];
        return v;
    }
    // Game.auto_targets: target cards for a tarot / spectral, in hand order
    std::vector<int> auto_targets(int name, const std::vector<int>& cs) const {
        std::vector<int> t;
        if (cs.empty()) return t;
        int n = CONS_DEFS[name].need;
        std::vector<int> plain;
        for (int id : cs) if (cards[id].enh == E_NONE) plain.push_back(id);
        auto by_value = [this](std::vector<int> v, bool desc) {
            std::stable_sort(v.begin(), v.end(), [this, desc](int a, int b) {
                double x = card_value(cards[a]), y = card_value(cards[b]);
                return desc ? (-x < -y) : (x < y);
            });
            return v;
        };
        auto head = [](const std::vector<int>& v, size_t k) {
            return std::vector<int>(v.begin(), v.begin() + std::min(k, v.size()));
        };
        int tsuit = CONS_DEFS[name].suit;
        if (tsuit >= 0) {
            std::vector<int> cand;
            for (int id : cs) if (!cards[id].is_stone() && cards[id].suit != tsuit) cand.push_back(id);
            std::stable_sort(cand.begin(), cand.end(), [this](int a, int b) { return cards[a].rank > cards[b].rank; });
            t = head(cand, n);
            if (t.empty()) t = head(cs, 1);
        } else if (name == C_HANGED_MAN) {
            t = head(by_value(cs, false), 2);
        } else if (name == C_DEATH) {
            double best = -1e9;
            int pa = -1, pb = -1;
            for (size_t a = 0; a < cs.size(); a++)
                for (size_t b = a + 1; b < cs.size(); b++) {
                    double gain = card_value(cards[cs[b]]) - card_value(cards[cs[a]]);
                    if (gain > best) { best = gain; pa = (int)a; pb = (int)b; }
                }
            if (pa >= 0) { t.push_back(cs[pa]); t.push_back(cs[pb]); }
            else t = head(cs, 2);
        } else if (name == C_STRENGTH) {
            std::vector<int> cand;
            for (int id : cs) if (!cards[id].is_stone() && cards[id].rank < 14) cand.push_back(id);
            std::stable_sort(cand.begin(), cand.end(), [this](int a, int b) { return cards[a].rank > cards[b].rank; });
            t = head(cand, 2);
            if (t.empty()) t = head(cs, 1);
        } else if (name == C_TOWER) {
            t = head(by_value(plain.empty() ? cs : plain, false), 1);
        } else if (CONS_DEFS[name].enh >= 0) {
            t = head(by_value(plain.empty() ? cs : plain, true), n);
        } else if (name == C_CRYPTID) {
            t = head(by_value(cs, true), 1);
        } else if (name == C_AURA) {
            std::vector<int> cand;
            for (int id : cs) if (cards[id].ed == ED_NONE) cand.push_back(id);
            t = head(by_value(cand, true), 1);
        } else {
            std::vector<int> cand;
            for (int id : cs) if (cards[id].seal == S_NONE && cards[id].ed == ED_NONE) cand.push_back(id);
            t = head(by_value(cand.empty() ? cs : cand, true), (size_t)std::max(1, n));
        }
        std::vector<int> out;                              // back in hand order
        for (int id : cs) if (in_pile(t, id)) out.push_back(id);
        return out;
    }
    // where: 0 the hand, 1 the pack hand, 2 no cards (used in the shop)
    void apply_consumable(const Cons& c, int where, const std::vector<int>* targets) {
        int n = c.id;
        if (c.kind() == CK_PLANET) {
            if ((planet_level_cap < 0 || planet_level_gains < planet_level_cap)
                && (hand_level_cap <= 0 || hand_levels[CONS_DEFS[n].planet_hand] < hand_level_cap)) {
                hand_levels[CONS_DEFS[n].planet_hand] += 1;
                planet_level_gains += 1;
            }
            planets_used |= 1u << n;
            planet_uses += 1;
            for (Joker& j : jokers) if (j.key == J_CONSTELLATION) j.st.set(j.st.get(1.0) + 0.1);
            last_consumable = n;
            return;
        }
        if (c.kind() == CK_TAROT) {
            tarots_used += 1;
            if (n != C_FOOL) last_consumable = n;
        }
        static const std::vector<int> none;
        // `cards` in the Python code aliases the pile itself: read it fresh wherever it is used after a change
        auto pile = [this, where]() -> const std::vector<int>& { return where == 0 ? hand : (where == 1 ? pack_hand : none); };
        std::vector<int> t = targets ? *targets : auto_targets(n, pile());
        static const uint8_t trio[3] = {ED_FOIL, ED_HOLO, ED_POLY};
        static const std::vector<double> trio_w = {50, 35, 15};
        if (CONS_DEFS[n].enh >= 0) {
            for (int id : t) cards[id].enh = (uint8_t)CONS_DEFS[n].enh;
        } else if (CONS_DEFS[n].suit >= 0) {
            for (int id : t) cards[id].suit = CONS_DEFS[n].suit;
        } else if (n == C_FOOL) {
            if (last_consumable != C_NONE) create_consumable(CONS_DEFS[last_consumable].kind, false, last_consumable);
        } else if (n == C_HIGH_PRIESTESS) {
            for (int k = 0; k < 2; k++) create_consumable(CK_PLANET, false, C_NONE, SRC_PRI);
        } else if (n == C_EMPEROR) {
            for (int k = 0; k < 2; k++) create_consumable(CK_TAROT, false, C_NONE, SRC_EMP);
        } else if (n == C_HERMIT) {
            money += std::min<long long>(20, std::max<long long>(0, money));
        } else if (n == C_TEMPERANCE) {
            long long s = 0;
            for (const Joker& j : jokers) s += j.sell_value();
            money += std::min<long long>(50, s);
        } else if (n == C_JUDGEMENT) {
            if ((int)jokers.size() < joker_slots) {
                Joker j;
                random_joker(j, 0, false, nullptr, false, "jud");
                add_joker(j);
            }
        } else if (n == C_WHEEL_OF_FORTUNE) {
            std::vector<int> cand;
            for (size_t i = 0; i < jokers.size(); i++) if (jokers[i].edition == ED_NONE) cand.push_back((int)i);
            if (!cand.empty() && roll("wheel_of_fortune") < prob(1, 4)) {
                int i = pick_joker("wheel_of_fortune", cand);
                jokers[i].edition = exact ? edition_poll("wheel_of_fortune") : trio[rng.choices_indices(trio_w, 1)[0]];
            }
        } else if (n == C_STRENGTH) {
            for (int id : t) cards[id].rank = cards[id].rank == 14 ? 2 : cards[id].rank + 1;
        } else if (n == C_HANGED_MAN) {
            destroy_cards(t);
        } else if (n == C_DEATH) {
            if (t.size() == 2) {
                Card& a = cards[t[0]];
                const Card& b = cards[t[1]];
                a.rank = b.rank; a.suit = b.suit; a.enh = b.enh; a.ed = b.ed; a.seal = b.seal;
            }
        } else if (n == C_BLACK_HOLE) {
            for (int h = 0; h < N_HANDS; h++) if (hand_level_cap <= 0 || hand_levels[h] < hand_level_cap) hand_levels[h] += 1;
        } else if (n == C_TALISMAN || n == C_DEJA_VU || n == C_TRANCE || n == C_MEDIUM) {
            uint8_t seal = n == C_TALISMAN ? S_GOLD : n == C_DEJA_VU ? S_RED : n == C_TRANCE ? S_BLUE : S_PURPLE;
            for (int id : t) cards[id].seal = seal;
        } else if (n == C_AURA) {
            for (int id : t)
                if (cards[id].ed == ED_NONE) cards[id].ed = exact ? edition_poll("aura") : trio[rng.choices_indices(trio_w, 1)[0]];
        } else if (n == C_CRYPTID) {
            for (int id : t)
                for (int k = 0; k < 2; k++) add_card(copy_card(id), true);
        } else if (n == C_SOUL) {
            if ((int)jokers.size() < joker_slots) {
                Joker j;
                random_joker(j, 4, false, nullptr, false, "sou");
                add_joker(j);
            }
        } else if (n == C_FAMILIAR || n == C_GRIM || n == C_INCANTATION) {
            if (!pile().empty()) {
                std::vector<int> v{pick_card("random_destroy", pile())};
                destroy_cards(v);
            }
            std::vector<int> ranks;
            if (n == C_FAMILIAR) ranks = {11, 12, 13};
            else if (n == C_GRIM) ranks = {14};
            else for (int r = 2; r < 11; r++) ranks.push_back(r);
            int k = n == C_FAMILIAR ? 3 : (n == C_GRIM ? 2 : 4);
            static const uint8_t enh7[7] = {E_BONUS, E_MULT, E_WILD, E_GLASS, E_STEEL, E_GOLD, E_LUCKY};
            for (int q = 0; q < k; q++) {
                if (exact) {
                    static const int SU[4] = {0, 1, 3, 2};       // the game's order S H D C as suit indices (S H C D)
                    std::string key = std::string(n == C_FAMILIAR ? "familiar" : n == C_GRIM ? "grim" : "incantation") + "_create";
                    int rank = n == C_GRIM ? 14 : ranks[sg.pseudorandom(key, 1, (int)ranks.size()) - 1];
                    int suit = SU[sg.pseudorandom(key, 1, 4) - 1];
                    uint8_t e = enh7[sg.pseudorandom("spe_card", 1, 7) - 1];
                    add_card(new_card(rank, suit, e), true);
                    continue;
                }
                int rank = ranks[rng.choice_index(ranks.size())];
                int suit = (int)rng.randrange(4);
                uint8_t e = enh7[rng.choice_index(7)];
                add_card(new_card(rank, suit, e), true);
            }
        } else if (n == C_IMMOLATE) {
            const std::vector<int>& cs = pile();
            std::vector<int> v;
            if (exact) {
                std::vector<int> lst = cs;
                std::stable_sort(lst.begin(), lst.end());
                sg.shuffle(lst, "immolate");
                for (size_t k = 0; k < std::min<size_t>(5, lst.size()); k++) v.push_back(lst[k]);
            } else {
                std::vector<size_t> pick = rng.sample_indices(cs.size(), std::min<size_t>(5, cs.size()));
                for (size_t k : pick) v.push_back(cs[k]);
            }
            destroy_cards(v);
            money += 20;
        } else if (n == C_SIGIL) {
            static const int SU[4] = {0, 1, 3, 2};
            int s = exact ? SU[sg.pseudorandom("sigil", 1, 4) - 1] : (int)rng.randrange(4);
            for (int id : pile()) cards[id].suit = (int8_t)s;
        } else if (n == C_OUIJA) {
            int r = exact ? 1 + sg.pseudorandom("ouija", 1, 13) : (int)rng.randrange(2, 15);
            for (int id : pile()) cards[id].rank = (int8_t)r;
            hand_size -= 1;
        } else if (n == C_ECTOPLASM) {
            std::vector<int> cand;
            for (size_t i = 0; i < jokers.size(); i++) if (jokers[i].edition == ED_NONE) cand.push_back((int)i);
            if (!cand.empty()) {
                int i = pick_joker("ectoplasm", cand);
                jokers[i].edition = ED_NEG;
                joker_slots += 1;
            }
            hand_size -= 1 + f_ecto;
            f_ecto += 1;
        } else if (n == C_WRAITH) {
            if ((int)jokers.size() < joker_slots) {
                Joker j;
                random_joker(j, 3, false, nullptr, false, "wra");
                add_joker(j);
            }
            money = 0;
        } else if (n == C_HEX) {
            if (!jokers.empty()) {
                std::vector<int> cand;
                for (size_t i = 0; i < jokers.size(); i++) if (jokers[i].edition == ED_NONE) cand.push_back((int)i);
                if (cand.empty()) for (size_t i = 0; i < jokers.size(); i++) cand.push_back((int)i);
                int keep = jokers[pick_joker("hex", cand)].uid;
                std::vector<int> snap;
                for (const Joker& j : jokers) snap.push_back(j.uid);
                for (int uid : snap) {
                    int i = find_joker(uid);
                    if (i >= 0 && uid != keep && !jokers[i].eternal) destroy_joker(uid);
                }
                jokers[find_joker(keep)].edition = ED_POLY;
            }
        } else if (n == C_ANKH) {
            if (!jokers.empty()) {
                std::vector<int> all;
                for (size_t q = 0; q < jokers.size(); q++) all.push_back((int)q);
                const Joker src = jokers[pick_joker("ankh_choice", all)];
                std::vector<int> snap;
                for (const Joker& j : jokers) snap.push_back(j.uid);
                for (int uid : snap) {
                    int i = find_joker(uid);
                    if (i >= 0 && uid != src.uid && !jokers[i].eternal) destroy_joker(uid);
                }
                Joker cp;
                cp.key = src.key;
                cp.edition = src.edition == ED_NEG ? ED_NONE : src.edition;
                cp.base_cost = src.base_cost;
                cp.st = src.st;
                cp.has_cost = src.has_cost;
                cp.cost = src.cost;
                add_joker(cp);
            }
        }
        std::vector<int> h2;
        for (int id : hand) if (in_pile(full_deck, id)) h2.push_back(id);
        hand = h2;
        sort_hand(hand);
    }

    // ---------------------------------------------------------------- packs
    void open_pack(int kind, int size, uint8_t return_to) {
        int pidx = kind * 3 + size;
        int shown = PACK_DEFS[pidx].shown;
        pack_kind = kind;
        pack_picks = PACK_DEFS[pidx].picks;
        pack_return = return_to;
        pack_cards.clear();
        pack_hand.clear();
        if (exact) {
            exact_open_pack(kind, shown);
            int hall = count_with_copies(J_HALLUCINATION);
            for (int k = 0; k < hall; k++)
                if (sg.pseudorandom("halu" + std::to_string(ante)) < prob(1, 2)) create_consumable(CK_TAROT, false, C_NONE, SRC_HAL);
            if (kind == PK_ARCANA || kind == PK_SPECTRAL) {        // the hand is drawn from the top of the deck
                size_t take = (size_t)std::min<int>(std::min(MAX_HAND, effective_hand_size()), (int)deck.size());
                for (size_t k = 0; k < take; k++) { pack_hand.push_back(deck.back()); deck.pop_back(); }
                sort_hand(pack_hand);
            }
            state = ST_PACK;
            return;
        }
        if (kind == PK_BUFFOON) {
            for (int k = 0; k < shown; k++) {
                bool exj[J_COUNT];
                for (int q = 0; q < J_COUNT; q++) exj[q] = false;
                for (const PackItem& p : pack_cards) exj[p.joker.key] = true;
                PackItem it;
                it.kind = PI_JOKER;
                random_joker(it.joker, 0, true, exj, false);
                pack_cards.push_back(it);
            }
        } else if (kind == PK_CELESTIAL) {
            std::vector<int> opts;
            for (int p = 0; p < N_PLANETS; p++) {
                int h = CONS_DEFS[p].planet_hand;
                if (!is_secret_hand(h) || hand_played[h] > 0) opts.push_back(p);
            }
            std::vector<size_t> pick = rng.sample_indices(opts.size(), std::min<size_t>((size_t)shown, opts.size()));
            std::vector<int> chosen;
            for (size_t k : pick) chosen.push_back(opts[k]);
            if (has_voucher(V_TELESCOPE)) {
                int most = 0;
                for (int i = 1; i < N_HANDS; i++) if (hand_played[i] > hand_played[most]) most = i;
                int p = HAND_TO_PLANET[most];
                if (std::find(chosen.begin(), chosen.end(), p) == chosen.end()) chosen[0] = p;
            }
            for (int p : chosen) { PackItem it; it.kind = PI_CONS; it.cons = make_cons(p); pack_cards.push_back(it); }
        } else if (kind == PK_ARCANA) {
            std::vector<size_t> pick = rng.sample_indices((size_t)N_TAROTS, (size_t)shown);
            for (size_t k : pick) {
                PackItem it; it.kind = PI_CONS; it.cons = make_cons(N_PLANETS + (int)k); pack_cards.push_back(it);
            }
            if (has_voucher(V_OMEN_GLOBE))
                for (size_t i = 0; i < pack_cards.size(); i++)
                    if (rng.random() < 0.2) {
                        uint64_t ex = 0;
                        for (const PackItem& p : pack_cards) ex |= 1ull << p.cons.id;
                        pack_cards[i].cons = random_consumable(CK_SPECTRAL, ex);
                    }
        } else if (kind == PK_SPECTRAL) {
            std::vector<int> spool;
            for (int s = 0; s < N_SPECTRALS; s++) {
                int id = N_PLANETS + N_TAROTS + s;
                if (!CONS_DEFS[id].hidden) spool.push_back(id);
            }
            std::vector<size_t> pick = rng.sample_indices(spool.size(), (size_t)shown);
            for (size_t k : pick) { PackItem it; it.kind = PI_CONS; it.cons = make_cons(spool[k]); pack_cards.push_back(it); }
        } else if (kind == PK_STANDARD) {
            for (int k = 0; k < shown; k++) {
                PackItem it; it.kind = PI_CARD; it.card = random_playing_card(true); pack_cards.push_back(it);
            }
        }
        for (size_t i = 0; i < pack_cards.size(); i++) {
            if (pack_cards[i].kind != PI_CONS) continue;
            // The Soul (arcana / spectral), then Black Hole (celestial / spectral): 0.3% each, once per run
            if ((kind == PK_ARCANA || kind == PK_SPECTRAL) && rng.random() < 0.003) {
                if (showman() || !f_seen_soul) { pack_cards[i].cons = make_cons(C_SOUL); f_seen_soul = true; }
                continue;
            }
            if ((kind == PK_CELESTIAL || kind == PK_SPECTRAL) && rng.random() < 0.003) {
                if (showman() || !f_seen_black_hole) { pack_cards[i].cons = make_cons(C_BLACK_HOLE); f_seen_black_hole = true; }
                continue;
            }
        }
        int hall = count_with_copies(J_HALLUCINATION);
        for (int k = 0; k < hall; k++)
            if (rng.random() < prob(1, 2)) create_consumable(CK_TAROT);
        if (kind == PK_ARCANA || kind == PK_SPECTRAL) {
            std::vector<int> pl = full_deck;
            rng.shuffle(pl);
            size_t take = (size_t)std::min(MAX_HAND, effective_hand_size());
            if (pl.size() > take) pl.resize(take);
            pack_hand = pl;
            sort_hand(pack_hand);
        }
        state = ST_PACK;
    }
    bool pack_pick_ok(int i) const {
        const PackItem& x = pack_cards[i];
        if (x.kind == PI_JOKER) return x.joker.edition == ED_NEG || (int)jokers.size() < joker_slots;
        if (x.kind == PI_CONS) return consumable_usable(x.cons, pack_hand, false);
        return true;
    }
    void pack_pick(int i, bool choose_targets = false) {
        if (i < 0 || i >= (int)pack_cards.size()) throw std::runtime_error("pack_pick: bad index");
        PackItem x = pack_cards[i];
        pack_cards.erase(pack_cards.begin() + i);
        if (x.kind == PI_JOKER) {
            add_joker(x.joker);
        } else if (x.kind == PI_CONS) {
            if (choose_targets && CONS_DEFS[x.cons.id].targeted && !pack_hand.empty()) {
                targeting = true; targeting_cons = x.cons; targeting_from_pack = true;
                return;
            }
            apply_consumable(x.cons, 1, nullptr);
        } else {
            add_card(x.card);
        }
        finish_pack_pick();
    }
    void finish_pack_pick() {
        std::vector<int> p2;
        for (int id : pack_hand) if (in_pile(full_deck, id)) p2.push_back(id);
        pack_hand = p2;
        sort_hand(pack_hand);
        pack_picks -= 1;
        if (pack_picks <= 0 || pack_cards.empty()) close_pack();
    }
    void pack_skip() {
        for (Joker& j : jokers) if (j.key == J_RED_CARD) j.st.set(j.st.get(0) + 3);
        close_pack();
    }
    void close_pack() {
        if (exact && !pack_hand.empty()) {                // the hand goes back under the deck, in order
            std::vector<int> back;
            for (int id : pack_hand) if (in_pile(full_deck, id)) back.push_back(id);
            deck.insert(deck.begin(), back.begin(), back.end());
        }
        pack_cards.clear();
        pack_hand.clear();
        state = pack_return;
    }
};

inline double ScoreCtx::roll(const char* key) { return (g->exact && key) ? g->sg.pseudorandom(std::string(key)) : rng->random(); }
inline int ScoreCtx::roll_int(const char* key, int lo, int hi) {
    return g->exact ? g->sg.pseudorandom(std::string(key), lo, hi) : (int)rng->randint(lo, hi);
}
inline bool ScoreCtx::cons_room_for_more() const {
    if (!g->exact) return true;
    int pending = 0;
    for (int i = 0; i < nev; i++) if (ev[i].type == EV_CREATE) pending++;
    return g->cons_used() + pending < g->consumable_slots;
}

}  // namespace bal
