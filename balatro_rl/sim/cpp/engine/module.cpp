// balatro_rl.sim._engine: Python bindings of the C++ game engine (game.hpp) and its environment (env.hpp).
//   Game     the game itself, method for method as sim/game.py, plus dump() (the whole state, for the
//            differential tests) and a few debug setters the tests use to build states
//   Env      one environment: observation encoding, legal actions, rewards (env.py)
//   VecEnv   many environments stepped together on a thread pool, observations written into numpy batches
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "game.hpp"
#include "env.hpp"

namespace py = pybind11;
using namespace bal;

static const char* ENH_NAMES[10] = {"", "BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY", "HIDDEN"};
static const char* ED_NAMES[5] = {"", "FOIL", "HOLO", "POLYCHROME", "NEGATIVE"};
static const char* SEAL_NAMES[5] = {"", "RED", "BLUE", "GOLD", "PURPLE"};
static const char* STATE_NAMES[6] = {"BLIND_SELECT", "SELECTING_HAND", "SHOP", "PACK", "GAME_OVER", "WON"};

static int find_name(const char* const* names, int n, const std::string& s, const char* what) {
    for (int i = 0; i < n; i++) if (s == names[i]) return i;
    throw std::invalid_argument(std::string("unknown ") + what + ": " + s);
}
static int joker_id(const std::string& s) {
    for (int i = 0; i < J_COUNT; i++) if (s == JOKER_DEFS[i].key) return i;
    throw std::invalid_argument("unknown joker: " + s);
}
static int cons_id(const std::string& s) {
    for (int i = 0; i < C_COUNT; i++) if (s == CONS_DEFS[i].name) return i;
    throw std::invalid_argument("unknown consumable: " + s);
}
static int voucher_id(const std::string& s) {
    for (int i = 0; i < V_COUNT; i++) if (s == VOUCHER_DEFS[i].key) return i;
    throw std::invalid_argument("unknown voucher: " + s);
}
static int boss_id(const std::string& s) {
    if (s.empty()) return B_NONE;
    for (int i = 0; i < B_COUNT; i++) if (s == BOSS_DEFS[i].key) return i;
    throw std::invalid_argument("unknown boss: " + s);
}
static int deck_id(const std::string& s) { return find_name(DECK_KEYS, D_COUNT, s, "deck"); }
static int stake_id(const std::string& s) { return find_name(STAKE_KEYS, N_STAKES, s, "stake"); }

static py::dict joker_dict(const Joker& j) {
    py::dict d;
    d["key"] = JOKER_DEFS[j.key].key;
    d["edition"] = ED_NAMES[j.edition];
    d["eternal"] = j.eternal;
    if (j.has_perish) d["perishable"] = j.perishable; else d["perishable"] = py::none();
    d["rental"] = j.rental;
    d["base_cost"] = j.base_cost;
    d["sell_bonus"] = j.sell_bonus;
    if (j.has_cost) d["cost"] = j.cost; else d["cost"] = py::none();
    d["debuffed"] = j.debuffed;
    d["hidden"] = j.hidden;
    py::dict st;
    if (j.st.has_val) st["val"] = j.st.val;
    if (j.st.rank >= 0) st["rank"] = j.st.rank;
    if (j.st.suit >= 0) st["suit"] = j.st.suit;
    if (j.st.hand >= 0) st["hand"] = j.st.hand;
    if (j.st.left >= 0) st["left"] = j.st.left;
    d["state"] = st;
    return d;
}

static py::tuple card_tuple(const Game& g, int id) {
    const Card& c = g.cards[id];
    return py::make_tuple((int)c.rank, (int)c.suit, ENH_NAMES[c.enh], ED_NAMES[c.ed], SEAL_NAMES[c.seal], c.extra,
                          c.deb, c.hidden, c.uid);
}
static py::list pile_list(const Game& g, const std::vector<int>& pile) {
    py::list l;
    for (int id : pile) l.append(card_tuple(g, id));
    return l;
}
static py::dict shop_item_dict(const Game& g, const ShopItem& it) {
    py::dict d;
    static const char* kinds[7] = {"joker", "tarot", "planet", "spectral", "voucher", "pack", "card"};
    d["kind"] = kinds[it.kind];
    d["cost"] = it.cost;
    if (it.kind == SK_JOKER) { d["key"] = std::string("j_") + JOKER_DEFS[it.key].key; d["joker"] = joker_dict(it.joker); }
    else if (it.kind == SK_TAROT || it.kind == SK_PLANET || it.kind == SK_SPECTRAL) d["key"] = std::string("c_") + CONS_DEFS[it.key].name;
    else if (it.kind == SK_VOUCHER) d["key"] = std::string("v_") + VOUCHER_DEFS[it.key].key;
    else if (it.kind == SK_PACK) d["key"] = PACK_DEFS[it.key].key;
    else { d["key"] = "<playing_card>"; d["card"] = card_tuple(g, it.card); }
    return d;
}

static py::dict dump_game(const Game& g) {
    py::dict d;
    d["planet_uses"] = g.planet_uses; d["planet_level_gains"] = g.planet_level_gains; d["tarots_used"] = g.tarots_used; d["ante"] = g.ante; d["blind_idx"] = g.blind_idx; d["state"] = STATE_NAMES[g.state]; d["money"] = g.money;
    d["chips"] = g.chips; d["target"] = g.target; d["hands_left"] = g.hands_left; d["discards_left"] = g.discards_left;
    d["discards_used_round"] = g.discards_used_round; d["joker_slots"] = g.joker_slots;
    d["consumable_slots"] = g.consumable_slots; d["hand_size"] = g.hand_size; d["base_hands"] = g.base_hands;
    d["base_discards"] = g.base_discards;
    py::list lv, hp, hpr;
    for (int i = 0; i < N_HANDS; i++) { lv.append(g.hand_levels[i]); hp.append(g.hand_played[i]); hpr.append(g.hand_played_round[i]); }
    d["hand_levels"] = lv; d["hand_played"] = hp; d["hand_played_round"] = hpr;
    py::list vs;
    for (int v = 0; v < V_COUNT; v++) if (g.has_voucher(v)) vs.append(VOUCHER_DEFS[v].key);
    d["vouchers"] = vs;
    py::list tags;
    for (int8_t t : g.pending_tags) tags.append(TAG_KEYS[t]);
    d["pending_tags"] = tags;
    py::dict fl;
    fl["verdant"] = g.f_verdant; fl["gros_michel_extinct"] = g.f_gros_extinct; fl["seen_soul"] = g.f_seen_soul;
    fl["seen_black_hole"] = g.f_seen_black_hole; fl["voucher_bought_ante"] = g.f_voucher_bought_ante;
    fl["ecto"] = g.f_ecto; fl["trading_destroy"] = g.f_trading_destroy >= 0;
    d["flags"] = fl;
    py::list pu;
    for (int p = 0; p < N_PLANETS; p++) if ((g.planets_used >> p) & 1u) pu.append(CONS_DEFS[p].name);
    d["planets_used"] = pu;
    d["tarots_used"] = g.tarots_used;
    d["last_consumable"] = g.last_consumable == C_NONE ? "" : CONS_DEFS[g.last_consumable].name;
    d["blinds_skipped"] = g.blinds_skipped; d["blinds_beaten"] = g.blinds_beaten; d["furthest_blind"] = g.furthest_blind;
    d["total_hands_played"] = g.total_hands_played; d["last_hand"] = g.last_hand;
    d["total_unused_discards"] = g.total_unused_discards;
    py::list ub;
    for (int b = 0; b < B_COUNT; b++) if ((g.used_bosses >> b) & 1u) ub.append(BOSS_DEFS[b].key);
    d["used_bosses"] = ub;
    d["crimson"] = g.find_joker(g.crimson_disabled);
    int forced = -1;
    for (size_t i = 0; i < g.hand.size(); i++) if (g.cards[g.hand[i]].uid == g.forced_uid) forced = (int)i;
    d["forced"] = forced;
    py::list rht;
    for (int h = 0; h < N_HANDS; h++) if ((g.round_hand_types >> h) & 1u) rht.append(h);
    d["round_hand_types"] = rht;
    d["mouth_hand"] = g.mouth_hand; d["boss_disabled"] = g.boss_disabled; d["juggle"] = g.juggle;
    d["first_draw_done"] = g.first_draw_done; d["boss_rerolled_ante"] = g.boss_rerolled_ante; d["ox_hand"] = g.ox_hand;
    if (g.targeting) d["targeting"] = py::make_tuple(CONS_DEFS[g.targeting_cons.id].name, g.targeting_from_pack);
    else d["targeting"] = py::none();
    py::list shop, packs, vouchers;
    for (const ShopItem& it : g.shop) shop.append(shop_item_dict(g, it));
    for (const ShopItem& it : g.shop_packs) packs.append(shop_item_dict(g, it));
    for (const ShopItem& it : g.shop_vouchers) vouchers.append(shop_item_dict(g, it));
    d["shop"] = shop; d["shop_packs"] = packs; d["shop_vouchers"] = vouchers;
    d["ante_voucher"] = g.has_ante_voucher ? std::string("v_") + VOUCHER_DEFS[g.ante_voucher.key].key : std::string("");
    d["reroll_cost"] = g.reroll_cost; d["free_rerolls"] = g.free_rerolls; d["shops_seen"] = g.shops_seen;
    py::list pc;
    for (const PackItem& x : g.pack_cards) {
        if (x.kind == PI_JOKER) pc.append(joker_dict(x.joker));
        else if (x.kind == PI_CONS) pc.append(std::string("c_") + CONS_DEFS[x.cons.id].name);
        else pc.append(card_tuple(g, x.card));
    }
    d["pack_cards"] = pc;
    d["pack_kind"] = g.pack_kind == PK_NONE ? "" : PACK_KIND_KEYS[g.pack_kind];
    d["pack_picks"] = g.pack_picks;
    d["pack_return"] = STATE_NAMES[g.pack_return];
    d["boss"] = g.boss == B_NONE ? "" : BOSS_DEFS[g.boss].key;
    py::list to;
    to.append(TAG_KEYS[g.tags_offered[0]]); to.append(TAG_KEYS[g.tags_offered[1]]);
    d["tags_offered"] = to;
    py::list js;
    for (const Joker& j : g.jokers) js.append(joker_dict(j));
    d["jokers"] = js;
    py::list cs;
    for (const Cons& c : g.consumables) cs.append(py::make_tuple(CONS_DEFS[c.id].name, c.negative, c.sell_bonus));
    d["consumables"] = cs;
    d["full_deck"] = pile_list(g, g.full_deck); d["deck"] = pile_list(g, g.deck); d["hand"] = pile_list(g, g.hand);
    d["discard_pile"] = pile_list(g, g.discard_pile); d["pack_hand"] = pile_list(g, g.pack_hand);
    py::list ap;
    for (int id : g.full_deck) ap.append((bool)g.ante_played[id]);
    d["ante_played"] = ap;
    d["starting_deck_size"] = g.starting_deck_size;
    return d;
}

static py::tuple rng_state(const Game& g) {
    py::list l;
    for (int i = 0; i < PyRandom::N; i++) l.append(g.rng.mt[i]);
    l.append(g.rng.index);
    return py::tuple(l);
}

static Game* make_game(uint64_t seed, const std::string& deck, const std::string& stake, int win_ante,
                       py::object pool, const std::string& exact_seed) {
    std::vector<int> pl;
    if (!pool.is_none())
        for (auto h : pool) pl.push_back(joker_id(h.cast<std::string>()));
    return new Game(seed, deck_id(deck), stake_id(stake), win_ante, pl.empty() ? nullptr : &pl,
                    exact_seed.empty() ? nullptr : exact_seed.c_str());
}

typedef std::vector<py::ssize_t> Shape;

static py::dict obs_dict_single(Obs& o) {
    py::dict d;
    d["g"] = py::array_t<float>(Shape{F_GLOBAL}, o.g);
    d["cards"] = py::array_t<float>(Shape{MAX_HAND, F_CARD}, o.cards);
    d["jid"] = py::array_t<int64_t>(Shape{MAX_JOKERS}, o.jid);
    d["jf"] = py::array_t<float>(Shape{MAX_JOKERS, F_JOKER}, o.jf);
    d["cid"] = py::array_t<int64_t>(Shape{MAX_CONSUMABLES}, o.cid);
    d["aid"] = py::array_t<int64_t>(Shape{N_ACTIONS}, o.aid);
    d["af"] = py::array_t<float>(Shape{N_ACTIONS, F_ACT}, o.af);
    d["mask"] = py::array_t<bool>(Shape{N_ACTIONS}, o.mask);
    d["subs"] = py::array_t<int64_t>(Shape{2 * N_SUB, 5}, o.subs);
    d["aid2"] = py::array_t<int64_t>(Shape{N_ACTIONS}, o.aid2);
    d["priv"] = py::array_t<float>(Shape{F_PRIV}, o.priv);
    return d;
}

PYBIND11_MODULE(_engine, m) {
    m.doc() = "The C++ Balatro engine";
    m.attr("N_ACTIONS") = N_ACTIONS;
    m.attr("F_ACT") = F_ACT;
    m.attr("F_CARD") = F_CARD;
    m.attr("F_JOKER") = F_JOKER;
    m.attr("F_GLOBAL") = F_GLOBAL;
    m.attr("N_SUB") = N_SUB;

    py::class_<Game>(m, "Game")
        .def(py::init(&make_game), py::arg("seed") = 0, py::arg("deck_type") = "RED", py::arg("stake") = "GOLD",
             py::arg("win_ante") = 8, py::arg("joker_pool") = py::none(), py::arg("exact_seed") = "")
        .def("exact_streams", [](const Game& g) {         // the named streams' states and the global math.random state
            py::dict d;
            for (const auto& kv : g.sg.st) d[py::str(kv.first)] = kv.second;
            return d;
        })
        .def("exact_rng", [](const Game& g) { return std::vector<uint64_t>(g.sg.rng.u, g.sg.rng.u + 4); })
        .def("clone", [](const Game& g) { return new Game(g); })
        .def("dump", &dump_game)
        .def("rng_state", &rng_state)
        .def_property_readonly("state", [](const Game& g) { return std::string(STATE_NAMES[g.state]); })
        .def_property_readonly("done", &Game::done)
        .def_readwrite("money", &Game::money)
        .def_readonly("ante", &Game::ante)
        .def_readonly("furthest_blind", &Game::furthest_blind)
        .def("select_blind", &Game::select_blind)
        .def("skip_blind", &Game::skip_blind)
        .def("reroll_boss", &Game::reroll_boss)
        .def("can_reroll_boss", &Game::can_reroll_boss)
        .def("play", &Game::play)
        .def("discard", &Game::discard)
        .def("buy_card", &Game::buy_card)
        .def("buy_pack", &Game::buy_pack)
        .def("buy_voucher", &Game::buy_voucher, py::arg("i") = 0)
        .def("reroll", &Game::reroll)
        .def("leave_shop", &Game::leave_shop)
        .def("sell_joker", &Game::sell_joker)
        .def("sell_consumable", &Game::sell_consumable)
        .def("swap_jokers", &Game::swap_jokers)
        .def("order_jokers", &Game::order_jokers)
        .def("use_consumable", &Game::use_consumable, py::arg("i"), py::arg("choose_targets") = false)
        .def("apply_targets", &Game::apply_targets)
        .def("pack_pick", &Game::pack_pick, py::arg("i"), py::arg("choose_targets") = false)
        .def("pack_skip", &Game::pack_skip)
        .def("predict", [](Game& g, const std::vector<int>& pos) {
            Plan plan;
            g.make_plan(plan);
            Game::View v;
            g.hand_view(v);
            int h = 0;
            double sc = g.predict(pos.data(), (int)pos.size(), plan, v, h);
            return py::make_tuple(sc, h);
        })
        // ---- debug setters (the tests build states with them, mirroring direct edits of the Python game)
        .def("debug_add_joker", [](Game& g, const std::string& key, const std::string& edition, bool eternal,
                                   py::object perishable, bool rental, int base_cost) {
            Joker j;
            j.key = (int16_t)joker_id(key);
            j.edition = (uint8_t)find_name(ED_NAMES, 5, edition, "edition");
            j.eternal = eternal;
            if (!perishable.is_none()) { j.has_perish = true; j.perishable = perishable.cast<int>(); }
            j.rental = rental;
            j.base_cost = base_cost;
            g.add_joker(j);
        }, py::arg("key"), py::arg("edition") = "", py::arg("eternal") = false, py::arg("perishable") = py::none(),
           py::arg("rental") = false, py::arg("base_cost") = 0)
        .def("debug_add_consumable", [](Game& g, const std::string& name, bool negative) {
            g.consumables.push_back(Game::make_cons(cons_id(name), negative));
        }, py::arg("name"), py::arg("negative") = false)
        .def("debug_redeem_voucher", [](Game& g, const std::string& v) { g.redeem_voucher(voucher_id(v)); })
        .def("debug_set_boss", [](Game& g, const std::string& b) { g.boss = boss_id(b); })
        .def("debug_set_card", [](Game& g, int deck_index, const std::string& enh, const std::string& ed,
                                  const std::string& seal) {
            Card& c = g.cards[g.full_deck.at(deck_index)];
            c.enh = (uint8_t)find_name(ENH_NAMES, 10, enh, "enhancement");
            c.ed = (uint8_t)find_name(ED_NAMES, 5, ed, "edition");
            c.seal = (uint8_t)find_name(SEAL_NAMES, 5, seal, "seal");
        })
        .def("debug_add_tag", [](Game& g, const std::string& t) { g.pending_tags.push_back((int8_t)find_name(TAG_KEYS, T_COUNT, t, "tag")); })
        .def("debug_set_tag_offered", [](Game& g, int i, const std::string& t) { g.tags_offered[i] = find_name(TAG_KEYS, T_COUNT, t, "tag"); })
        .def("debug_set_hand_level", [](Game& g, int h, int lv) { g.hand_levels[h] = lv; })
        .def("debug_set_ante", [](Game& g, int a) { g.ante = a; })
        .def("debug_set_joker_state", [](Game& g, int i, const std::string& key, double v) {
            JState& st = g.jokers.at(i).st;
            if (key == "val") st.set(v);
            else if (key == "rank") st.rank = (int)v;
            else if (key == "suit") st.suit = (int)v;
            else if (key == "hand") st.hand = (int)v;
            else if (key == "left") st.left = (int)v;
            else throw std::invalid_argument("unknown joker state key: " + key);
        });

    py::class_<Env>(m, "Env")
        .def(py::init([](const std::string& deck, const std::string& stake, double ante_weight, double win_bonus,
                         int win_ante, double shape_chips, double shape_gamma) {
            return new Env(deck_id(deck), stake_id(stake), ante_weight, win_bonus, win_ante, shape_chips, shape_gamma);
        }), py::arg("deck") = "RED", py::arg("stake") = "GOLD", py::arg("ante_weight") = 1.0,
            py::arg("win_bonus") = 10.0, py::arg("win_ante") = 8, py::arg("shape_chips") = 0.0,
            py::arg("shape_gamma") = 0.995)
        .def("reset", [](Env& e, uint64_t seed) {
            e.reset(seed);
            return obs_dict_single(e.obs_ref());
        })
        .def("observe", [](Env& e) { e.encode(); return obs_dict_single(e.obs_ref()); })
        .def("step", [](Env& e, int a) {
            StepResult r = e.step(a);
            py::dict info;
            info["ante"] = e.g.ante;
            info["won"] = e.g.state == ST_WON;
            info["blinds"] = e.g.furthest_blind;
            py::object ob = r.done ? py::object(py::none()) : py::object(obs_dict_single(e.obs_ref()));
            return py::make_tuple(ob, r.reward, r.done, info);
        })
        // timing helpers: seconds for n repetitions (the state is not changed)
        .def("bench_encode", [](Env& e, int n) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; i++) e.encode();
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        })
        .def("bench_candidates", [](Env& e, int n) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; i++) e.candidates();
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        })
        .def("bench_features", [](Env& e, int n) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; i++) { e.legal_mask(); e.action_features(); e.encode_global(); }
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        })
        .def("bench_reset", [](Env& e, int n) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; i++) e.reset((uint64_t)i);
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        })
        .def_property_readonly("game", [](Env& e) -> Game& { return e.g; }, py::return_value_policy::reference_internal)
        .def_readwrite("start_xmult", &Env::start_xmult)
        .def_readwrite("start_any", &Env::start_any)
        .def_readwrite("start_money_p", &Env::start_money_p)
        .def_readwrite("start_money", &Env::start_money)
        .def_readwrite("start_money2_p", &Env::start_money2_p)
        .def_readwrite("start_money2", &Env::start_money2)
        .def_readwrite("start_grow", &Env::start_grow)
        .def_readwrite("marginal", &Env::marginal)
        .def_readwrite("start_xscale", &Env::start_xscale)
        .def_readwrite("start_xscale_val", &Env::start_xscale_val)
        .def_readwrite("school_p", &Env::school_p)
        .def_readwrite("school_ids", &Env::school_ids)
        .def_readonly("school_student", &Env::school_student)
        .def_readwrite("death_pen", &Env::death_pen)
        .def_readwrite("death_cap", &Env::death_cap)
        .def_readwrite("marg_verify", &Env::marg_verify)
        .def_readonly("marg_dbg", &Env::marg_dbg)
        .def_readonly("marg_dbg_base", &Env::marg_dbg_base)
        .def_readonly("marg_hits", &Env::marg_hits)
        .def_readonly("marg_misses", &Env::marg_misses)
        .def_readwrite("ban_p", &Env::ban_p)
        .def_readwrite("exact", &Env::exact)
        .def_readwrite("tilt_p", &Env::tilt_p)
        .def_readwrite("tilt_levels", &Env::tilt_levels)
        .def_readwrite("banned", &Env::banned)
        .def_readonly("restricted", &Env::restricted)
        .def_readwrite("start_ante", &Env::start_ante)
        .def_readwrite("scn_p", &Env::scn_p)
        .def_readwrite("auto_round", &Env::auto_round)
        .def_readwrite("sp_pace", &Env::sp_pace)
        .def_readwrite("scn_build_seed", &Env::scn_build_seed)
        .def_readwrite("scn_power_lo", &Env::scn_power_lo)
        .def_readwrite("scn_power_hi", &Env::scn_power_hi)
        .def_readwrite("scn_min_ante", &Env::scn_min_ante)
        .def_readwrite("scn_max_ante", &Env::scn_max_ante)
        .def_readwrite("scn_couple", &Env::scn_couple)
        .def_readwrite("scn_force_blind", &Env::scn_force_blind)
        .def_readwrite("scn_level_conc", &Env::scn_level_conc)
        .def_readwrite("immortal", &Env::immortal)
        .def_readwrite("autoplay", &Env::autoplay)
        .def_readwrite("reroll_cap_base", &Env::reroll_cap_base)
        .def_readwrite("hand_level_cap_cfg", &Env::hand_level_cap_cfg)
        .def_readwrite("flash_cap_cfg", &Env::flash_cap_cfg)
        .def_readwrite("scaled_start", &Env::scaled_start)
        .def_readwrite("scaled_shop", &Env::scaled_shop)
        .def_readwrite("scaled_rounds", &Env::scaled_rounds)
        .def_readonly("scn_kind", &Env::scn_kind)
        .def_readonly("scn_limit", &Env::scn_limit)
        .def_readonly("last_score", &Env::last_score)
        .def("set_scn_weights", [](Env& e, double a, double b, double c) { e.scn_w[0] = a; e.scn_w[1] = b; e.scn_w[2] = c; })
        .def_readwrite("log_partial", &Env::log_partial)
        .def_readwrite("level_cap_mode", &Env::level_cap_mode)
        .def_readwrite("pack_restock", &Env::pack_restock)
        .def_property("pack_bias", [](const Env& e) { return std::vector<double>(e.pack_bias, e.pack_bias + 5); },
                      [](Env& e, const std::vector<double>& b) { for (int k = 0; k < 5 && k < (int)b.size(); k++) e.pack_bias[k] = b[k]; })
        .def_readwrite("level_cap_p", &Env::level_cap_p)
        .def_readwrite("level_cap_n", &Env::level_cap_n)
        .def_readwrite("level_cap_ante", &Env::level_cap_ante)
        .def_readonly("level_capped_game", &Env::level_capped_game)
        .def_readwrite("verdant_bonus", &Env::verdant_bonus)
        .def_readonly("steps", &Env::steps)
        .def_readonly("rerolls", &Env::rerolls)
        .def_readonly("swaps", &Env::swaps);

    py::class_<VecEnv>(m, "VecEnv")
        .def(py::init([](int n, const std::string& deck, const std::string& stake, uint64_t seed0, int threads,
                         double ante_weight, double win_bonus, int win_ante, double shape_chips, double shape_gamma,
                         double start_xmult, double start_any, double verdant_bonus,
                         const std::vector<std::string>& stakes) {
            std::vector<int> st;
            for (const std::string& x : stakes) st.push_back(stake_id(x));
            return new VecEnv(n, deck_id(deck), stake_id(stake), seed0, threads, ante_weight, win_bonus, win_ante,
                              shape_chips, shape_gamma, start_xmult, start_any, verdant_bonus, &st);
        }), py::arg("n"), py::arg("deck") = "RED", py::arg("stake") = "GOLD", py::arg("seed0") = 0,
            py::arg("threads") = 0, py::arg("ante_weight") = 1.0, py::arg("win_bonus") = 10.0,
            py::arg("win_ante") = 8, py::arg("shape_chips") = 0.0, py::arg("shape_gamma") = 0.995,
            py::arg("start_xmult") = 0.0, py::arg("start_any") = 0.0, py::arg("verdant_bonus") = 0.0,
            py::arg("stakes") = std::vector<std::string>())
        .def("set_start", &VecEnv::set_start, py::arg("xmult"), py::arg("any"), py::arg("money_p") = 0.0,
             py::arg("money") = 0, py::arg("money2_p") = 0.0, py::arg("money2") = 0, py::arg("grow") = 0.0)
        .def("set_ban", &VecEnv::set_ban, py::arg("jokers"), py::arg("p"))
        .def("set_joker_cap", &VecEnv::set_joker_cap, py::arg("jokers"), py::arg("max_held"))
        .def("add_joker_cap", &VecEnv::add_joker_cap, py::arg("jokers"), py::arg("max_held"))
        .def("set_start_ante", &VecEnv::set_start_ante, py::arg("ante"))
        .def("set_scenario", &VecEnv::set_scenario, py::arg("p"), py::arg("w_one"), py::arg("w_few"), py::arg("w_full"), py::arg("min_ante"),
             py::arg("max_ante"), py::arg("reroll_cap"), py::arg("couple"))
        .def("scn_kinds", &VecEnv::scn_kinds)
        .def("set_immortal", &VecEnv::set_immortal, py::arg("on"), py::arg("pay") = false)
        .def("set_autoplay", &VecEnv::set_autoplay, py::arg("on"), py::arg("nodes") = 400, py::arg("force") = false)
        .def("set_blind_mult", &VecEnv::set_blind_mult, py::arg("mult"))
        .def("last_bits", &VecEnv::last_bits)
        .def("set_weight_geom", &VecEnv::set_weight_geom, py::arg("ratio"))
        .def("set_archive", &VecEnv::set_archive, py::arg("p_start"), py::arg("p_store"), py::arg("lo_ante"), py::arg("hi_ante"), py::arg("cap"), py::arg("min_fill"),
             py::arg("winners") = false)
        .def("share_archive", &VecEnv::share_archive, py::arg("other"))
        .def("archive_outcomes", &VecEnv::archive_outcomes, py::arg("reset") = true)
        .def("archive_counts", &VecEnv::archive_counts, py::arg("stake") = -1)
        .def("archive_put", &VecEnv::archive_put, py::arg("ids"))
        .def("set_solver_tarots", &VecEnv::set_solver_tarots, py::arg("on"), py::arg("targets") = 3)
        .def("set_start_any2", &VecEnv::set_start_any2, py::arg("p"))
        .def("set_blind_bases", &VecEnv::set_blind_bases, py::arg("bases"))
        .def("set_start_eternal", &VecEnv::set_start_eternal, py::arg("on"))
        .def("power_curve", [](VecEnv& v, const std::vector<int>& idx, int samples) { py::gil_scoped_release release; return v.power_curve(idx, samples); }, py::arg("idx"), py::arg("samples") = 2)
        .def("set_catalogue", &VecEnv::set_catalogue, py::arg("p"))
        .def("set_solver_order", &VecEnv::set_solver_order, py::arg("smart"))
        .def("set_curve_crn", &VecEnv::set_curve_crn, py::arg("seed"), py::arg("generic_boss") = true)
        .def("set_solver_tt", &VecEnv::set_solver_tt, py::arg("on"))
        .def("hide_future", &VecEnv::hide_future, py::arg("idx"), py::arg("salts"))
        .def("archive_starts", &VecEnv::archive_starts, py::arg("reset") = true)
        .def("set_verdant_relative", &VecEnv::set_verdant_relative, py::arg("on"))
        .def("set_round_feed", &VecEnv::set_round_feed, py::arg("w"))
        .def("state_keys", &VecEnv::state_keys, py::arg("idx"))
        .def("set_rules", &VecEnv::set_rules, py::arg("no_stickers"), py::arg("small_blind_pay"))
        .def("set_auto_target", &VecEnv::set_auto_target, py::arg("on"))
        .def("set_fixed_seed", &VecEnv::set_fixed_seed, py::arg("seed"))
        .def("targeting_flags", &VecEnv::targeting_flags, py::arg("idx"))
        .def("projected_power", [](VecEnv& v, const std::vector<int>& idx, int samples) { py::gil_scoped_release release; return v.projected_power(idx, samples); }, py::arg("idx"), py::arg("samples") = 4)
        .def("final_boss_clear", [](VecEnv& v, const std::vector<int>& idx, int samples, int node_cap) { py::gil_scoped_release release; return v.final_boss_clear(idx, samples, node_cap); },
             py::arg("idx"), py::arg("samples") = 2, py::arg("node_cap") = 200)
        .def("fast_power", [](VecEnv& v, const std::vector<int>& idx, int samples) { py::gil_scoped_release release; return v.fast_power(idx, samples); }, py::arg("idx"), py::arg("samples") = 8)
        .def("set_shop_only", &VecEnv::set_shop_only, py::arg("on"), py::arg("nodes") = 1000, py::arg("partial_w") = 0.5)
        .def("hand_levels", &VecEnv::hand_levels, py::arg("idx"))
        .def("edit", &VecEnv::edit, py::arg("ids"), py::arg("op"), py::arg("a"), py::arg("b") = -1)
        .def("dump", [](VecEnv& v, int i) { return dump_game(v.envs[i]->g); }, py::arg("i"))      // one environment's whole game (as Game.dump)
        .def("cell_info", &VecEnv::cell_info, py::arg("idx"))
        .def("diff_states", &VecEnv::diff_states, py::arg("i"), py::arg("j"))
        .def("diff_envs", &VecEnv::diff_envs, py::arg("i"), py::arg("j"))
        .def("set_level_scripted", &VecEnv::set_level_scripted, py::arg("mode"))
        .def("set_search_prune", &VecEnv::set_search_prune, py::arg("bits"))
        .def("set_solver_extra", &VecEnv::set_solver_extra, py::arg("n"))
        .def("set_solver_hand_bound", &VecEnv::set_solver_hand_bound, py::arg("on"))
        .def("set_order_mode", &VecEnv::set_order_mode, py::arg("mode"))
        .def("set_pot_mode", &VecEnv::set_pot_mode, py::arg("mode"), py::arg("rounds"))
        .def("set_search_reroll_cap", &VecEnv::set_search_reroll_cap, py::arg("n"))
        .def("set_search_pack_macro", &VecEnv::set_search_pack_macro, py::arg("on"))
        .def("load_nnue", &VecEnv::load_nnue, py::arg("path"), py::arg("cols"), py::arg("off"), py::arg("consts"))
        .def("set_round_eval", &VecEnv::set_round_eval, py::arg("w"))
        .def("nnue_eval", &VecEnv::nnue_eval, py::arg("idx"), py::arg("as_next_select") = false)
        .def("nnue_feature_lists", &VecEnv::nnue_feature_lists, py::arg("idx"))
        .def("set_solver_speed", &VecEnv::set_solver_speed, py::arg("lite"), py::arg("cache"), py::arg("dead_f"))
        .def("solver_cache_stats", &VecEnv::solver_cache_stats, py::arg("reset"))
        .def("solve_best", [](VecEnv& v, const std::vector<int>& idx, int node_cap) { py::gil_scoped_release release; return v.solve_best(idx, node_cap); }, py::arg("idx"), py::arg("node_cap") = 1000)
        .def("set_round_resources", &VecEnv::set_round_resources, py::arg("idx"), py::arg("hands"), py::arg("discards"))
        .def("set_solver", &VecEnv::set_solver, py::arg("n_plays"), py::arg("n_discards"), py::arg("first_plays") = 0, py::arg("first_discards") = 0)
        .def("set_power", &VecEnv::set_power, py::arg("lo"), py::arg("hi"))
        .def("set_build_seeds", &VecEnv::set_build_seeds, py::arg("seeds"), py::arg("force_blind") = -1, py::arg("level_conc") = 0.0)
        .def("set_state", &VecEnv::set_state, py::arg("ids"), py::arg("ante"), py::arg("blind"), py::arg("counters"))
        .def("set_marginal_blind_only", &VecEnv::set_marginal_blind_only, py::arg("on"))
        .def("reseed_rng", &VecEnv::reseed_rng, py::arg("idx"), py::arg("salts"))
        .def("set_build", &VecEnv::set_build, py::arg("ids"), py::arg("ante"), py::arg("blind"), py::arg("keys"), py::arg("counters"), py::arg("steel_cards"), py::arg("levels"), py::arg("money"), py::arg("editions") = std::vector<int>{}, py::arg("suit") = -1, py::arg("suit_n") = 0)
        .def("set_modes", &VecEnv::set_modes, py::arg("modes"))
        .def("set_auto_round", &VecEnv::set_auto_round, py::arg("on"), py::arg("pace"), py::arg("mode") = 0)
        .def("round_report", &VecEnv::round_report)
        .def("margin_report", &VecEnv::margin_report)
        .def("set_margin_params", &VecEnv::set_margin_params, py::arg("m_full"), py::arg("samples"), py::arg("force_lambda") = -1.0)
        .def("set_caps", &VecEnv::set_caps, py::arg("reroll_cap"), py::arg("hand_level_cap"), py::arg("flash_cap") = 0)
        .def("set_scaled", &VecEnv::set_scaled, py::arg("start_n"), py::arg("shop"), py::arg("rounds"))
        .def("play_scores", &VecEnv::play_scores)
        .def("set_log_partial", &VecEnv::set_log_partial, py::arg("on"))
        .def("set_part_credit", &VecEnv::set_part_credit, py::arg("on"))
        .def("set_menu", &VecEnv::set_menu, py::arg("n"))
        .def("set_sandbox", &VecEnv::set_sandbox, py::arg("level_cap_mode"), py::arg("pack_restock"), py::arg("bias"))
        .def("set_level_cap", &VecEnv::set_level_cap, py::arg("p"), py::arg("n"), py::arg("ante"))
        .def("set_tilt", &VecEnv::set_tilt, py::arg("p"), py::arg("levels") = 2)
        .def("set_exact", &VecEnv::set_exact, py::arg("on"))
        .def("restart", &VecEnv::restart)
        .def("set_priv", &VecEnv::set_priv, py::arg("on"))
        .def("reseed", [](VecEnv& v, const std::vector<int>& idx, const std::vector<uint64_t>& salts) {
            for (size_t i = 0; i < idx.size(); i++) { v.envs[idx[i]]->reseed_future(salts[i]); v.envs[idx[i]]->encode(); }
        }, py::arg("idx"), py::arg("salts"))
        .def("expand_level", [](VecEnv& v, const std::vector<int>& idx, VecEnv& dest, int ends_per, int sw, int sl, int node_cap,
                                double money_w, double pot_w, double scale_w, const std::vector<int>& target, double target_w) {
            std::vector<LevelOut> out;
            {
                py::gil_scoped_release release;
                out = v.expand_level(idx, dest, ends_per, sw, sl, node_cap, money_w, pot_w, scale_w, target, target_w);
            }
            py::list res;
            for (const LevelOut& o : out) {
                py::list cands;
                for (const LevelCand& c : o.cands) cands.append(py::make_tuple(c.path, c.score, (py::int_)c.sig, c.slot, c.jokers));
                res.append(py::make_tuple(o.win, o.win_path, cands));
            }
            return res;
        }, py::arg("idx"), py::arg("dest"), py::arg("ends_per") = 8, py::arg("sw") = 32, py::arg("sl") = 7, py::arg("node_cap") = 1000,
           py::arg("money_w") = 0.01, py::arg("pot_w") = 0.0, py::arg("scale_w") = 0.0, py::arg("target") = std::vector<int>(), py::arg("target_w") = 0.0)
        .def("copy_many", [](VecEnv& dst, VecEnv& src, const std::vector<int>& si, const std::vector<int>& di) {
            py::gil_scoped_release release;
            dst.copy_many(src, si, di);
        }, py::arg("src"), py::arg("si"), py::arg("di"))
        .def("solve_rounds", [](VecEnv& v, const std::vector<int>& idx, int max_plans, int node_cap) {
            py::gil_scoped_release release;
            return v.solve_rounds(idx, max_plans, node_cap);
        }, py::arg("idx"), py::arg("max_plans") = 3, py::arg("node_cap") = 400)
        .def("copy_env_from", [](VecEnv& dst, VecEnv& src, int i, int j) {      // environment j of this one := environment i of src
            if (i < 0 || i >= src.n || j < 0 || j >= dst.n) throw std::invalid_argument("copy_env_from: bad index");
            dst.envs[j]->copy_state_from(*src.envs[i]);
        }, py::arg("src"), py::arg("i"), py::arg("j"))
        .def("set_reward", &VecEnv::set_reward, py::arg("blind_scale"), py::arg("ante8_bonus"))
        .def("set_weight_shape", &VecEnv::set_weight_shape, py::arg("ante_weight"), py::arg("power"))
        .def("set_marginal", &VecEnv::set_marginal, py::arg("on"))
        .def("set_school", &VecEnv::set_school, py::arg("p"), py::arg("students"))
        .def("set_xscale", &VecEnv::set_xscale, py::arg("share"), py::arg("val"))
        .def("set_stakes", [](VecEnv& v, const std::vector<std::string>& names) {
            std::vector<int> st;
            for (const std::string& x : names) st.push_back(stake_id(x));
            v.set_stakes(st);
        }, py::arg("stakes"))
        .def("set_death_penalty", &VecEnv::set_death_penalty, py::arg("pen"), py::arg("cap"))
        .def("school_student", &VecEnv::school_student)
        .def("school_stats", [](VecEnv& v) {
            std::vector<double> o = v.school_stats();
            py::array_t<double> out(Shape{(py::ssize_t)J_COUNT, (py::ssize_t)2}, o.data());
            return out;
        })
        .def("restricted", [](VecEnv& v) {          // per environment: the game being played has the ban
            py::array_t<bool> out(v.n);
            bool* o = out.mutable_data();
            for (int i = 0; i < v.n; i++) o[i] = v.envs[i]->restricted;
            return out;
        })
        .def_readonly("n", &VecEnv::n)
        .def_readonly("threads", &VecEnv::threads)
        // the batched observation: numpy views of the environments' buffers (no copy; step() overwrites them)
        .def("obs", [](py::object self) {
            VecEnv& v = self.cast<VecEnv&>();
            py::ssize_t n = v.n;
            py::dict d;
            d["g"] = py::array_t<float>(Shape{n, F_GLOBAL}, v.b_g.data(), self);
            d["cards"] = py::array_t<float>(Shape{n, MAX_HAND, F_CARD}, v.b_cards.data(), self);
            d["jid"] = py::array_t<int64_t>(Shape{n, MAX_JOKERS}, v.b_jid.data(), self);
            d["jf"] = py::array_t<float>(Shape{n, MAX_JOKERS, F_JOKER}, v.b_jf.data(), self);
            d["cid"] = py::array_t<int64_t>(Shape{n, MAX_CONSUMABLES}, v.b_cid.data(), self);
            d["aid"] = py::array_t<int64_t>(Shape{n, N_ACTIONS}, v.b_aid.data(), self);
            d["af"] = py::array_t<float>(Shape{n, N_ACTIONS, F_ACT}, v.b_af.data(), self);
            d["mask"] = py::array_t<bool>(Shape{n, N_ACTIONS}, (bool*)v.b_mask.data(), self);
            d["subs"] = py::array_t<int64_t>(Shape{n, 2 * N_SUB, 5}, v.b_subs.data(), self);
            d["aid2"] = py::array_t<int64_t>(Shape{n, N_ACTIONS}, v.b_aid2.data(), self);
            d["priv"] = py::array_t<float>(Shape{n, F_PRIV}, v.b_priv.data(), self);
            return d;
        })
        .def("step", [](VecEnv& v, py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions) {
            if (actions.size() != v.n) throw std::invalid_argument("actions: wrong size");
            py::array_t<float> rew(v.n);
            py::array_t<bool> done(v.n), won(v.n);
            py::array_t<int32_t> ante(v.n), blinds(v.n);
            const int64_t* a = actions.data();
            float* r = rew.mutable_data();
            bool* d = done.mutable_data();
            int32_t* an = ante.mutable_data();
            bool* w = won.mutable_data();
            int32_t* bl = blinds.mutable_data();
            {
                py::gil_scoped_release release;
                v.step(a, r, d, an, w, bl);
            }
            return py::make_tuple(rew, done, ante, won, blinds);
        })
        .def("set_sparse", &VecEnv::set_sparse, py::arg("from_blind"))
        // step with the reward split: (reward, done, ante, won, blinds, sparse part of the reward, ended by the step limit)
        .def("step_split", [](VecEnv& v, py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions) {
            if (actions.size() != v.n) throw std::invalid_argument("actions: wrong size");
            py::array_t<float> rew(v.n), sparse(v.n);
            py::array_t<bool> done(v.n), won(v.n), trunc(v.n);
            py::array_t<int32_t> ante(v.n), blinds(v.n);
            const int64_t* a = actions.data();
            {
                py::gil_scoped_release release;
                v.step(a, rew.mutable_data(), done.mutable_data(), ante.mutable_data(), won.mutable_data(), blinds.mutable_data(),
                       sparse.mutable_data(), trunc.mutable_data());
            }
            return py::make_tuple(rew, done, ante, won, blinds, sparse, trunc);
        });
}
