# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled score prediction: a port of scoring.score_hand + hands.evaluate + the jokers' scoring
hooks for the expected-value case (rng=None, commit=False), plus Game.violates_boss.

The Python code is the reference. fastscore.py turns a game state into the flat tables used here,
and tests/test_fastscore.py checks the two agree. Effects that only move money are left out
(they never change the score)."""
cimport cython
from libc.math cimport pow, floor, isinf, isnan
from libc.stdlib cimport malloc
from cpython.long cimport PyLong_FromDouble

cdef enum:
    MAXC = 64          # cards in hand (view)
    MAXJ = 64          # jokers
    MAXPOOL = 512      # cards a pooled scorer can draw hands from
    MAXHAND = 16       # hand size for the pooled calls (subset patterns are precomputed up to this)

# hand types, same numbering as hands.py
cdef enum:
    HC = 0
    PAIR = 1
    TWO_PAIR = 2
    TRIPS = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    QUADS = 7
    STRAIGHT_FLUSH = 8
    FIVE_KIND = 9
    FLUSH_HOUSE = 10
    FLUSH_FIVE = 11

# card enhancements / editions / seals
cdef enum:
    E_NONE = 0
    E_BONUS = 1
    E_MULT = 2
    E_WILD = 3
    E_GLASS = 4
    E_STEEL = 5
    E_STONE = 6
    E_GOLD = 7
    E_LUCKY = 8
    E_HIDDEN = 9
    E_OTHER = 10
cdef enum:
    ED_NONE = 0
    ED_FOIL = 1
    ED_HOLO = 2
    ED_POLY = 3
    ED_NEG = 4
    ED_OTHER = 5
cdef enum:
    S_NONE = 0
    S_RED = 1
    S_OTHER = 2

# bosses that matter for prediction
cdef enum:
    B_NONE = 0
    B_FLINT = 1
    B_ARM = 2
    B_PSYCHIC = 3
    B_EYE = 4
    B_MOUTH = 5

# add kind for "m"/"c"/"x" style effects
cdef enum:
    K_MULT = 0
    K_CHIPS = 1
    K_X = 2

# hook kinds (per phase)
cdef enum:
    BK_NONE = 0
    BK_BUS = 1
    BK_GREEN = 2
    BK_RUNNER = 3
    BK_SQUARE = 4
    BK_TROUSERS = 5
    BK_LOYALTY = 6
    BK_OBELISK = 7
    BK_VAMPIRE = 8
    BK_MIDAS = 9
cdef enum:
    CK_NONE = 0
    CK_SUIT = 1        # has suit p_suit -> add p_amt (p_kind)
    CK_SCARY = 2
    CK_EVEN = 3
    CK_ODD = 4
    CK_SCHOLAR = 5
    CK_WALKIE = 6
    CK_SMILEY = 7
    CK_PHOTO = 8
    CK_FIB = 9
    CK_BLOOD = 10
    CK_IDOL = 11
    CK_ANCIENT = 12
    CK_WEE = 13
    CK_TRIB = 14
    CK_COPY = 15
cdef enum:
    RK_NONE = 0
    RK_CHAD = 1
    RK_HACK = 2
    RK_DUSK = 3
    RK_SOCK = 4
    RK_SELZER = 5
    RK_COPY = 6
cdef enum:
    HK_NONE = 0
    HK_SHOOT = 1
    HK_BARON = 2
    HK_COPY = 3
cdef enum:
    MK_NONE = 0
    MK_CONST = 1       # add p_amt (p_kind)
    MK_CONTAINS = 2    # p_hand in contains -> add p_amt (p_kind)
    MK_HALF = 3
    MK_BANNER = 4
    MK_MYSTIC = 5
    MK_RAISED_FIST = 6
    MK_ABSTRACT = 7
    MK_SUPERNOVA = 8
    MK_VAL_MULT = 9
    MK_VAL_CHIPS = 10
    MK_VAL_X = 11
    MK_BLUE = 12
    MK_SWASH = 13
    MK_FORTUNE = 14
    MK_CARD_SHARP = 15
    MK_BULL = 16
    MK_BOOTSTRAPS = 17
    MK_ACROBAT = 18
    MK_BLACKBOARD = 19
    MK_FLOWER = 20
    MK_SEEING = 21
    MK_STENCIL = 22
    MK_LOYALTY = 23
    MK_STEEL = 24
    MK_EROSION = 25
    MK_STONE = 26
    MK_LUCKY_CAT = 27
    MK_BASEBALL = 28
    MK_THROWBACK = 29
    MK_DRIVERS = 30
    MK_COPY = 31

CODES = dict(
    E_NONE=E_NONE, E_BONUS=E_BONUS, E_MULT=E_MULT, E_WILD=E_WILD, E_GLASS=E_GLASS, E_STEEL=E_STEEL,
    E_STONE=E_STONE, E_GOLD=E_GOLD, E_LUCKY=E_LUCKY, E_HIDDEN=E_HIDDEN, E_OTHER=E_OTHER,
    ED_NONE=ED_NONE, ED_FOIL=ED_FOIL, ED_HOLO=ED_HOLO, ED_POLY=ED_POLY, ED_NEG=ED_NEG, ED_OTHER=ED_OTHER,
    S_NONE=S_NONE, S_RED=S_RED, S_OTHER=S_OTHER,
    B_NONE=B_NONE, B_FLINT=B_FLINT, B_ARM=B_ARM, B_PSYCHIC=B_PSYCHIC, B_EYE=B_EYE, B_MOUTH=B_MOUTH,
    K_MULT=K_MULT, K_CHIPS=K_CHIPS, K_X=K_X,
    BK_NONE=BK_NONE, BK_BUS=BK_BUS, BK_GREEN=BK_GREEN, BK_RUNNER=BK_RUNNER, BK_SQUARE=BK_SQUARE,
    BK_TROUSERS=BK_TROUSERS, BK_LOYALTY=BK_LOYALTY, BK_OBELISK=BK_OBELISK, BK_VAMPIRE=BK_VAMPIRE,
    BK_MIDAS=BK_MIDAS,
    CK_NONE=CK_NONE, CK_SUIT=CK_SUIT, CK_SCARY=CK_SCARY, CK_EVEN=CK_EVEN, CK_ODD=CK_ODD,
    CK_SCHOLAR=CK_SCHOLAR, CK_WALKIE=CK_WALKIE, CK_SMILEY=CK_SMILEY, CK_PHOTO=CK_PHOTO, CK_FIB=CK_FIB,
    CK_BLOOD=CK_BLOOD, CK_IDOL=CK_IDOL, CK_ANCIENT=CK_ANCIENT, CK_WEE=CK_WEE, CK_TRIB=CK_TRIB,
    CK_COPY=CK_COPY,
    RK_NONE=RK_NONE, RK_CHAD=RK_CHAD, RK_HACK=RK_HACK, RK_DUSK=RK_DUSK, RK_SOCK=RK_SOCK,
    RK_SELZER=RK_SELZER, RK_COPY=RK_COPY,
    HK_NONE=HK_NONE, HK_SHOOT=HK_SHOOT, HK_BARON=HK_BARON, HK_COPY=HK_COPY,
    MK_NONE=MK_NONE, MK_CONST=MK_CONST, MK_CONTAINS=MK_CONTAINS, MK_HALF=MK_HALF, MK_BANNER=MK_BANNER,
    MK_MYSTIC=MK_MYSTIC, MK_RAISED_FIST=MK_RAISED_FIST, MK_ABSTRACT=MK_ABSTRACT,
    MK_SUPERNOVA=MK_SUPERNOVA, MK_VAL_MULT=MK_VAL_MULT, MK_VAL_CHIPS=MK_VAL_CHIPS, MK_VAL_X=MK_VAL_X,
    MK_BLUE=MK_BLUE, MK_SWASH=MK_SWASH, MK_FORTUNE=MK_FORTUNE, MK_CARD_SHARP=MK_CARD_SHARP,
    MK_BULL=MK_BULL, MK_BOOTSTRAPS=MK_BOOTSTRAPS, MK_ACROBAT=MK_ACROBAT, MK_BLACKBOARD=MK_BLACKBOARD,
    MK_FLOWER=MK_FLOWER, MK_SEEING=MK_SEEING, MK_STENCIL=MK_STENCIL, MK_LOYALTY=MK_LOYALTY,
    MK_STEEL=MK_STEEL, MK_EROSION=MK_EROSION, MK_STONE=MK_STONE, MK_LUCKY_CAT=MK_LUCKY_CAT,
    MK_BASEBALL=MK_BASEBALL, MK_THROWBACK=MK_THROWBACK, MK_DRIVERS=MK_DRIVERS, MK_COPY=MK_COPY,
)

# (base chips, base mult, chips per level, mult per level), as hands.HAND_BASE
HAND_BASE = [(5, 1, 10, 1), (10, 2, 15, 1), (20, 2, 20, 1), (30, 3, 20, 2), (30, 4, 30, 3), (35, 4, 15, 2),
             (40, 4, 25, 2), (60, 7, 30, 3), (100, 8, 40, 4), (120, 12, 35, 3), (140, 14, 40, 4),
             (160, 16, 50, 3)]
cdef int HB[12][4]
for _h in range(12):
    for _i in range(4):
        HB[_h][_i] = HAND_BASE[_h][_i]


cdef struct Crd:
    int rank
    int suit
    int enh
    int ed
    int seal
    int deb
    double extra

cdef struct Jkr:
    int bk, ck, rk, hk, mk
    int p_hand, p_kind, p_suit
    double p_amt
    int ed
    int target          # blueprint/brainstorm: resolved copy target (table index) or -1
    int st_rank, st_suit
    int has_val
    double val
    long long sell

cdef struct Ctx:
    double chips, mult, lucky
    int trig, hand, contains, k, nsc, nh
    int pos[5]          # view index of each played card
    int scoring[5]      # indices into the played list
    int held[MAXC]      # view indices of held cards
    int ov[5]           # enhancement override per played card (-1 = none)
    double val[MAXJ]
    int hv[MAXJ]
    int photo[MAXJ]


cdef inline bint is_stone(int e) noexcept nogil:
    return e == E_STONE or e == E_HIDDEN

cdef inline int chip_value(const Crd* c) noexcept nogil:
    if c.enh == E_HIDDEN:
        return 0
    if c.enh == E_STONE:
        return 50
    if c.rank == 14:
        return 11
    return c.rank if c.rank < 10 else 10

cdef inline bint is_face(const Crd* c, bint par) noexcept nogil:
    if is_stone(c.enh):
        return False
    return par or (c.rank >= 11 and c.rank <= 13)

cdef inline bint has_suit(const Crd* c, int s, bint sm) noexcept nogil:
    if is_stone(c.enh):
        return False
    if c.enh == E_WILD:
        return True
    if sm:
        return (c.suit % 2) == (s % 2)
    return c.suit == s


cdef bint _straight_ok(int* r, int n, int maxgap) noexcept nogil:
    cdef int i, j, t
    for i in range(1, n):                 # insertion sort
        t = r[i]
        j = i - 1
        while j >= 0 and r[j] > t:
            r[j + 1] = r[j]
            j -= 1
        r[j + 1] = t
    for i in range(n - 1):
        if not (0 < r[i + 1] - r[i] <= maxgap):
            return False
    return True


cdef bint _is_straight_ranks(const int* ranks, int n, bint shortcut) noexcept nogil:
    cdef int r[5]
    cdef int i, j
    cdef int maxgap = 2 if shortcut else 1
    cdef bint ace = False
    for i in range(n):
        r[i] = ranks[i]
    for i in range(n):                    # duplicate ranks -> no straight
        for j in range(i + 1, n):
            if r[i] == r[j]:
                return False
    for i in range(n):
        if r[i] == 14:
            ace = True
    if _straight_ok(r, n, maxgap):
        return True
    if ace:
        for i in range(n):
            r[i] = 1 if ranks[i] == 14 else ranks[i]
        if _straight_ok(r, n, maxgap):
            return True
    return False


cdef void evaluate(const Crd** cs, int n, bint ff, bint shortcut, bint sm,
                   int* out_hand, int* out_scoring, int* out_contains) noexcept nogil:
    """hands.evaluate for n <= 5 cards. Scoring cards and contained hands come back as bitmasks."""
    cdef int need = 4 if ff else 5
    cdef int normal[5]
    cdef int nn = 0, stones = 0, i, j, m, k, s, r, hand, sc, contains
    cdef int cnt[16]
    cdef int ns[4]
    cdef int ranks[5]
    cdef int wild = 0, best_s, best_n, distinct = 0
    cdef int g0r = -1, g0c = 0, g1r = -1, g1c = 0, top, second
    cdef int fmask = 0, smask = 0
    cdef bint full, is_flush, is_straight
    for i in range(16):
        cnt[i] = 0
    for i in range(n):
        if is_stone(cs[i].enh):
            stones |= 1 << i
        else:
            normal[nn] = i
            nn += 1
            cnt[cs[i].rank] += 1
    for r in range(14, -1, -1):           # rank groups ordered by (-count, -rank)
        if cnt[r] > 0:
            distinct += 1
        if cnt[r] > g0c:
            g0c = cnt[r]
            g0r = r
    for r in range(14, -1, -1):
        if r != g0r and cnt[r] > g1c:
            g1c = cnt[r]
            g1r = r
    top = g0c
    second = g1c

    # flush: first suit with the most cards (wilds count for every suit)
    if nn >= need:
        ns[0] = ns[1] = ns[2] = ns[3] = 0
        for j in range(nn):
            i = normal[j]
            if cs[i].enh == E_WILD:
                wild += 1
            else:
                ns[cs[i].suit] += 1
        if sm:
            ns[0] = ns[2] = ns[0] + ns[2]
            ns[1] = ns[3] = ns[1] + ns[3]
        best_s = -1
        best_n = need - 1
        for s in range(4):
            if ns[s] + wild > best_n:
                best_s = s
                best_n = ns[s] + wild
        if best_s >= 0:
            for j in range(nn):
                i = normal[j]
                if cs[i].enh == E_WILD or (cs[i].suit % 2 == best_s % 2 if sm else cs[i].suit == best_s):
                    fmask |= 1 << i
    is_flush = fmask != 0

    # straight: union of every straight subset of size need..min(5, nn)
    if nn >= need and distinct >= need:
        for m in range(1, 1 << nn):
            k = 0
            for j in range(nn):
                if m & (1 << j):
                    ranks[k] = cs[normal[j]].rank
                    k += 1
            if k < need or k > 5:
                continue
            if _is_straight_ranks(ranks, k, shortcut):
                for j in range(nn):
                    if m & (1 << j):
                        smask |= 1 << normal[j]
    is_straight = smask != 0

    contains = 1 << HC
    if top >= 2:
        contains |= 1 << PAIR
    if top >= 3:
        contains |= 1 << TRIPS
    if top >= 4:
        contains |= 1 << QUADS
    if top >= 5:
        contains |= 1 << FIVE_KIND
    if (top >= 2 and second >= 2) or top >= 4:
        contains |= 1 << TWO_PAIR
    if is_straight:
        contains |= 1 << STRAIGHT
    if is_flush:
        contains |= 1 << FLUSH
    if is_straight and is_flush:
        contains |= 1 << STRAIGHT_FLUSH
    full = top >= 3 and second >= 2
    if full:
        contains |= 1 << FULL_HOUSE

    sc = 0
    if top >= 5 and is_flush:
        hand = FLUSH_FIVE
        sc = _rank_mask(cs, normal, nn, g0r, -1)
    elif full and is_flush:
        hand = FLUSH_HOUSE
        sc = _rank_mask(cs, normal, nn, g0r, g1r)
    elif top >= 5:
        hand = FIVE_KIND
        sc = _rank_mask(cs, normal, nn, g0r, -1)
    elif is_straight and is_flush:
        hand = STRAIGHT_FLUSH
        sc = fmask | smask
    elif top >= 4:
        hand = QUADS
        sc = _rank_mask(cs, normal, nn, g0r, -1)
    elif full:
        hand = FULL_HOUSE
        sc = _rank_mask(cs, normal, nn, g0r, g1r)
    elif is_flush:
        hand = FLUSH
        sc = fmask
    elif is_straight:
        hand = STRAIGHT
        sc = smask
    elif top >= 3:
        hand = TRIPS
        sc = _rank_mask(cs, normal, nn, g0r, -1)
    elif top >= 2 and second >= 2:
        hand = TWO_PAIR
        sc = _rank_mask(cs, normal, nn, g0r, g1r)
    elif top >= 2:
        hand = PAIR
        sc = _rank_mask(cs, normal, nn, g0r, -1)
    else:
        hand = HC
        if nn > 0:
            k = normal[0]                 # highest rank, first position among ties
            for j in range(1, nn):
                if cs[normal[j]].rank > cs[k].rank:
                    k = normal[j]
            sc = 1 << k
    if hand == FLUSH_FIVE:
        contains |= (1 << FLUSH) | (1 << FIVE_KIND)
    elif hand == FLUSH_HOUSE:
        contains |= (1 << FLUSH) | (1 << FULL_HOUSE)
    out_hand[0] = hand
    out_scoring[0] = sc | stones
    out_contains[0] = contains


cdef inline int _rank_mask(const Crd** cs, const int* normal, int nn, int r1, int r2) noexcept nogil:
    cdef int j, m = 0
    for j in range(nn):
        if cs[normal[j]].rank == r1 or cs[normal[j]].rank == r2:
            m |= 1 << normal[j]
    return m


def kept_feats_many(cards, subsets):
    """env._kept_feats for many discards from the same hand. cards: (rank, suit, enh, hidden) tuples."""
    cdef int n = len(cards), i, k, s, r, nk, mx_suit, mx_rank, pairs, present, best, w, cnt_w, chips
    cdef int rank[MAXC]
    cdef int suit[MAXC]
    cdef int enh[MAXC]
    cdef int hid[MAXC]
    cdef int ns[4]
    cdef int nr[15]
    cdef long long insub
    cdef Crd tmp
    if n > MAXC:
        raise ValueError("too many cards")
    for i, c in enumerate(cards):
        rank[i], suit[i], enh[i], hid[i] = c
    out = []
    for sub in subsets:
        insub = 0
        k = len(sub)
        for x in sub:
            insub |= (<long long>1) << <int>x
        ns[0] = ns[1] = ns[2] = ns[3] = 0
        for r in range(15):
            nr[r] = 0
        nk = 0
        chips = 0
        for i in range(n):
            if insub & ((<long long>1) << i):
                tmp.rank = rank[i]
                tmp.enh = enh[i]
                chips += chip_value(&tmp)
                continue
            if hid[i] or is_stone(enh[i]):
                continue
            if enh[i] == E_WILD:
                ns[0] += 1; ns[1] += 1; ns[2] += 1; ns[3] += 1
            else:
                ns[suit[i]] += 1
            nr[rank[i]] += 1
            nk += 1
        mx_suit = ns[0]
        for s in range(1, 4):
            if ns[s] > mx_suit:
                mx_suit = ns[s]
        mx_rank = 0
        pairs = 0
        present = 0
        for r in range(15):
            if nr[r] > mx_rank:
                mx_rank = nr[r]
            if nr[r] >= 2:
                pairs += 1
            if r >= 2 and nr[r]:
                present |= 1 << r
        if present & (1 << 14):
            present |= 2                  # ace also counts low
        best = 0
        for w in range(1, 11):
            cnt_w = 0
            for r in range(w, w + 5):
                if present & (1 << r):
                    cnt_w += 1
            if cnt_w > best:
                best = cnt_w
        out.append([mx_suit / 5.0, mx_rank / 4.0, best / 5.0, pairs / 3.0,
                    chips / (11.0 * (k if k > 1 else 1)), nk / 8.0])
    return out


# ------------------------------------------------------------------ subset patterns
# every 1-5 card subset of an n-card hand, in the order of
#   [c for k in range(1, min(5, n) + 1) for c in itertools.combinations(range(n), k)]
# (the order the Python callers use, so ties between equal scores resolve the same way)
cdef int PAT_OFF[MAXHAND + 1]
cdef int PAT_N[MAXHAND + 1]
cdef int* PAT_K
cdef int* PAT_POS


def _init_patterns():
    global PAT_K, PAT_POS
    from itertools import combinations
    pats = []
    for n in range(MAXHAND + 1):
        PAT_OFF[n] = len(pats)
        subs = [c for k in range(1, min(5, n) + 1) for c in combinations(range(n), k)]
        PAT_N[n] = len(subs)
        pats += subs
    PAT_K = <int*> malloc(len(pats) * sizeof(int))
    PAT_POS = <int*> malloc(5 * len(pats) * sizeof(int))
    if PAT_K == NULL or PAT_POS == NULL:
        raise MemoryError()
    for t, c in enumerate(pats):
        PAT_K[t] = len(c)
        for i in range(5):
            PAT_POS[5 * t + i] = c[i] if i < len(c) else -1


_init_patterns()


def subset_patterns(int n):
    """The subsets the pooled calls score, as tuples, in their order (for mapping indices back)."""
    from itertools import combinations
    return [c for k in range(1, min(5, n) + 1) for c in combinations(range(n), k)]


cdef inline object _score_obj(double f, bint viol):
    # what predict_many returns for one play: 0.0 when the boss forbids it, else the int floor
    return 0.0 if viol else PyLong_FromDouble(f)


cdef inline void _check_finite(double f) except *:
    # predict_many converts every floor(score) with PyLong_FromDouble, which raises on inf / nan
    if isinf(f):
        raise OverflowError("cannot convert float infinity to integer")
    if isnan(f):
        raise ValueError("cannot convert float NaN to integer")


@cython.final
cdef class Scorer:
    """Game state for one decision, flattened; predict_many scores candidate plays."""
    cdef Crd cards[MAXC]
    cdef int ncards
    cdef Jkr jk[MAXJ]
    cdef int nj
    cdef int l_before[MAXJ]
    cdef int l_card[MAXJ]
    cdef int l_retrig[MAXJ]
    cdef int l_held[MAXJ]
    cdef int l_main[MAXJ]
    cdef int n_before, n_card, n_retrig, n_held, n_main
    cdef bint ff, sc, sm, par, splash
    cdef int mime
    cdef bint vff, vsc, vsm
    cdef int boss, vboss, round_types, mouth_hand
    cdef int levels[12]
    cdef int hplayed[12]
    cdef int hplayed_round[12]
    cdef int obs[12]
    cdef double p5, p15, p2
    cdef bint plasma
    cdef long long discards_left, hands_left, money, njokers, deck_len, tarots, skipped, slots
    cdef long long stencils, steel, stone, enhanced, full_len, start_len, sell_total, rare2
    cdef Crd pool[MAXPOOL]
    cdef int npool
    cdef double last_chips, last_mult    # of the latest _score call (after Plasma Deck averaging)

    def __init__(self, cards, jokers, lists, flags, boss, arrays, probs, scalars):
        cdef int i
        if len(cards) > MAXC or len(jokers) > MAXJ:
            raise ValueError("too many cards or jokers for the compiled scorer")
        self.ncards = len(cards)
        for i, c in enumerate(cards):
            (self.cards[i].rank, self.cards[i].suit, self.cards[i].enh, self.cards[i].ed,
             self.cards[i].seal, self.cards[i].extra, self.cards[i].deb) = c
        self.nj = len(jokers)
        for i, j in enumerate(jokers):
            (self.jk[i].bk, self.jk[i].ck, self.jk[i].rk, self.jk[i].hk, self.jk[i].mk,
             self.jk[i].p_hand, self.jk[i].p_kind, self.jk[i].p_suit, self.jk[i].p_amt,
             self.jk[i].ed, self.jk[i].target, self.jk[i].st_rank, self.jk[i].st_suit,
             self.jk[i].has_val, self.jk[i].val, self.jk[i].sell) = j
        before, card, retrig, held, main = lists
        self.n_before = len(before)
        for i, x in enumerate(before):
            self.l_before[i] = x
        self.n_card = len(card)
        for i, x in enumerate(card):
            self.l_card[i] = x
        self.n_retrig = len(retrig)
        for i, x in enumerate(retrig):
            self.l_retrig[i] = x
        self.n_held = len(held)
        for i, x in enumerate(held):
            self.l_held[i] = x
        self.n_main = len(main)
        for i, x in enumerate(main):
            self.l_main[i] = x
        (self.ff, self.sc, self.sm, self.par, self.splash, self.mime, self.vff, self.vsc, self.vsm) = flags
        self.boss, self.vboss, self.round_types, self.mouth_hand = boss
        levels, hplayed, hplayed_round, obs = arrays
        for i in range(12):
            self.levels[i] = levels[i]
            self.hplayed[i] = hplayed[i]
            self.hplayed_round[i] = hplayed_round[i]
            self.obs[i] = obs[i]
        self.p5, self.p15, self.p2 = probs
        (self.plasma, self.discards_left, self.hands_left, self.money, self.njokers, self.deck_len,
         self.tarots, self.skipped, self.slots, self.stencils, self.steel, self.stone, self.enhanced,
         self.full_len, self.start_len, self.sell_total, self.rare2) = scalars

    def predict_many(self, subsets):
        """[(score, hand type)] for each subset of hand positions, like Game.predict."""
        cdef int pos[5]
        cdef int k, i, hand
        cdef bint viol
        cdef double v
        out = []
        for s in subsets:
            k = len(s)
            if k < 1 or k > 5:
                raise ValueError("a play has 1 to 5 cards")
            for i in range(k):
                pos[i] = s[i]
                if pos[i] < 0 or pos[i] >= self.ncards:
                    raise IndexError("hand position out of range")
            v = self._score(pos, k, &hand, &viol)
            sc = PyLong_FromDouble(floor(v))     # math.floor(chips * mult), overflow errors included
            out.append((0.0 if viol else sc, hand))
        return out

    # ------------------------------------------------------------------ pooled hands (batched calls)
    def set_pool(self, cards):
        """Cards (same row format as the constructor's) that hands are drawn from by index."""
        cdef int i
        if len(cards) > MAXPOOL:
            raise ValueError("too many cards for the pooled scorer")
        for i, c in enumerate(cards):
            (self.pool[i].rank, self.pool[i].suit, self.pool[i].enh, self.pool[i].ed,
             self.pool[i].seal, self.pool[i].extra, self.pool[i].deb) = c
        self.npool = len(cards)

    cdef int _load(self, hand, long long hands_left, long long discards_left, long long deck_len,
                   int round_types, int mouth_hand) except -1:
        cdef int n = len(hand), i, k
        if n > MAXHAND:
            raise ValueError("hand too large for the pooled scorer")
        for i in range(n):
            k = hand[i]
            if k < 0 or k >= self.npool:
                raise IndexError("pool index out of range")
            self.cards[i] = self.pool[k]
        self.ncards = n
        self.hands_left, self.discards_left, self.deck_len = hands_left, discards_left, deck_len
        self.round_types, self.mouth_hand = round_types, mouth_hand
        return n

    cdef list _best_two(self, int n):
        """[(score, hand type, subset index)] for the two best plays, best first; ties keep the earlier
        subset (what a stable sort by score gives)."""
        cdef int off = PAT_OFF[n], m = PAT_N[n], t, i, k, hand, h1 = -1, h2 = -1, b1 = -1, b2 = -1
        cdef int pos[5]
        cdef double v, f, s1 = -1e308, s2 = -1e308
        cdef bint viol, v1 = False, v2 = False
        for t in range(m):
            k = PAT_K[off + t]
            for i in range(k):
                pos[i] = PAT_POS[5 * (off + t) + i]
            v = self._score(pos, k, &hand, &viol)
            f = floor(v)
            _check_finite(f)
            if viol:
                f = 0.0
            if b1 < 0 or f > s1:
                b2, s2, h2, v2 = b1, s1, h1, v1
                b1, s1, h1, v1 = t, f, hand, viol
            elif b2 < 0 or f > s2:
                b2, s2, h2, v2 = t, f, hand, viol
        out = []
        if b1 >= 0:
            out.append((_score_obj(s1, v1), h1, b1))
        if b2 >= 0:
            out.append((_score_obj(s2, v2), h2, b2))
        return out

    def best_two(self, hand, long long hands_left, long long discards_left, long long deck_len,
                 int round_types, int mouth_hand):
        """The two best plays of one hand (pool indices): [(score, hand type, subset index)], best first.
        Subset indices refer to subset_patterns(len(hand)). Scores as predict_many returns them."""
        return self._best_two(self._load(hand, hands_left, discards_left, deck_len, round_types, mouth_hand))

    def best_two_many(self, hands, long long hands_left, long long discards_left, long long deck_len,
                      int round_types, int mouth_hand):
        """best_two for a batch of hands that share the round counters (one call for all of them)."""
        return [self._best_two(self._load(h, hands_left, discards_left, deck_len, round_types, mouth_hand))
                for h in hands]

    def score_all(self, hand, long long hands_left, long long discards_left, long long deck_len,
                  int round_types, int mouth_hand):
        """[(score, hand type)] for every subset of one hand, in subset_patterns order; the same values as
        predict_many(subset_patterns(len(hand))) on that hand."""
        cdef int n = self._load(hand, hands_left, discards_left, deck_len, round_types, mouth_hand)
        cdef int off = PAT_OFF[n], m = PAT_N[n], t, i, k, h
        cdef int pos[5]
        cdef double v, f
        cdef bint viol
        out = []
        for t in range(m):
            k = PAT_K[off + t]
            for i in range(k):
                pos[i] = PAT_POS[5 * (off + t) + i]
            v = self._score(pos, k, &h, &viol)
            f = floor(v)
            _check_finite(f)
            out.append((_score_obj(f, viol), h))
        return out

    def score_ext(self, subsets):
        """[(score, hand type, chips, mult)] for plays of the loaded hand (in any card order): predict_many's
        values plus the final chips and mult, as scoring.score_hand leaves them in ctx.chips / ctx.mult."""
        cdef int pos[5]
        cdef int k, i, hand
        cdef bint viol
        cdef double v, f
        out = []
        for s in subsets:
            k = len(s)
            if k < 1 or k > 5:
                raise ValueError("a play has 1 to 5 cards")
            for i in range(k):
                pos[i] = s[i]
                if pos[i] < 0 or pos[i] >= self.ncards:
                    raise IndexError("hand position out of range")
            v = self._score(pos, k, &hand, &viol)
            f = floor(v)
            _check_finite(f)
            out.append((_score_obj(f, viol), hand, self.last_chips, self.last_mult))
        return out

    # ------------------------------------------------------------------ joker state helpers
    cdef inline double _get(self, Ctx* x, int j, double default) noexcept:
        return x.val[j] if x.hv[j] else default

    cdef inline void _set(self, Ctx* x, int j, double v) noexcept:
        x.val[j] = v
        x.hv[j] = 1

    cdef inline int _enh(self, Ctx* x, int i) noexcept:
        return x.ov[i] if x.ov[i] >= 0 else self.cards[x.pos[i]].enh

    cdef inline void _add(self, Ctx* x, int kind, double amt) noexcept:
        if kind == K_MULT:
            x.mult += amt
        elif kind == K_CHIPS:
            x.chips += amt
        else:
            x.mult *= amt

    # ------------------------------------------------------------------ hooks
    cdef void _before(self, Ctx* x, int j) noexcept:
        cdef int kind = self.jk[j].bk, p, i, n
        cdef double v
        cdef const Crd* c
        cdef bint face
        cdef int most
        if kind == BK_BUS:
            face = False
            for p in range(x.nsc):
                if is_face(&self.cards[x.pos[x.scoring[p]]], self.par):
                    face = True
                    break
            if face:
                self._set(x, j, 0)
            else:
                self._set(x, j, self._get(x, j, 0) + 1)
        elif kind == BK_GREEN:
            self._set(x, j, self._get(x, j, 0) + 1)
        elif kind == BK_RUNNER:
            if x.contains & (1 << STRAIGHT):
                self._set(x, j, self._get(x, j, 0) + 15)
        elif kind == BK_SQUARE:
            if x.k == 4:
                self._set(x, j, self._get(x, j, 0) + 4)
        elif kind == BK_TROUSERS:
            if x.contains & (1 << TWO_PAIR):
                self._set(x, j, self._get(x, j, 0) + 2)
        elif kind == BK_LOYALTY:
            v = self._get(x, j, 5)
            self._set(x, j, v - 1 if v > 0 else 5)
        elif kind == BK_OBELISK:
            most = 0
            for i in range(12):
                if self.hplayed[i] > most:
                    most = self.hplayed[i]
            if most == 0:
                most = -1
            if most >= 0 and self.hplayed[x.hand] == most:
                self._set(x, j, 1.0)
            else:
                self._set(x, j, self._get(x, j, 1.0) + 0.2)
        elif kind == BK_VAMPIRE:
            n = 0
            for p in range(x.nsc):
                i = x.scoring[p]
                if self._enh(x, i) != E_NONE and not self.cards[x.pos[i]].deb:
                    x.ov[i] = E_NONE
                    n += 1
            self._set(x, j, self._get(x, j, 1.0) + 0.1 * n)
        elif kind == BK_MIDAS:
            for p in range(x.nsc):
                i = x.scoring[p]
                if is_face(&self.cards[x.pos[i]], self.par):
                    x.ov[i] = E_GOLD

    cdef void _card(self, Ctx* x, int j, int i, bint allow_copy) noexcept:
        """Per-card effect of joker j (its own state) on played card i."""
        cdef int kind = self.jk[j].ck, p, first, t
        cdef const Crd* c = &self.cards[x.pos[i]]
        cdef int r = c.rank
        cdef bint st = is_stone(c.enh)
        if kind == CK_NONE:
            return
        elif kind == CK_SUIT:
            if has_suit(c, self.jk[j].p_suit, self.sm):
                self._add(x, self.jk[j].p_kind, self.jk[j].p_amt)
        elif kind == CK_SCARY:
            if is_face(c, self.par):
                x.chips += 30
        elif kind == CK_EVEN:
            if not st and r <= 10 and r % 2 == 0:
                x.mult += 4
        elif kind == CK_ODD:
            if not st and (r == 14 or (r <= 10 and r % 2 == 1)):
                x.chips += 31
        elif kind == CK_SCHOLAR:
            if not st and r == 14:
                x.chips += 20
                x.mult += 4
        elif kind == CK_WALKIE:
            if not st and (r == 10 or r == 4):
                x.chips += 10
                x.mult += 4
        elif kind == CK_SMILEY:
            if is_face(c, self.par):
                x.mult += 5
        elif kind == CK_PHOTO:
            first = -1
            for p in range(x.nsc):
                if is_face(&self.cards[x.pos[x.scoring[p]]], self.par):
                    first = x.scoring[p]
                    break
            if first == i and x.photo[j] != x.trig:
                x.photo[j] = x.trig
                x.mult *= 2
        elif kind == CK_FIB:
            if not st and (r == 14 or r == 2 or r == 3 or r == 5 or r == 8):
                x.mult += 8
        elif kind == CK_BLOOD:
            if has_suit(c, 1, self.sm):
                x.mult *= 1 + (1.5 - 1) * self.p2
        elif kind == CK_IDOL:
            if not st and r == self.jk[j].st_rank and has_suit(c, self.jk[j].st_suit, self.sm):
                x.mult *= 2
        elif kind == CK_ANCIENT:
            if has_suit(c, self.jk[j].st_suit, self.sm):
                x.mult *= 1.5
        elif kind == CK_WEE:
            if not st and r == 2:
                self._set(x, j, self._get(x, j, 0) + 8)
        elif kind == CK_TRIB:
            if not st and (r == 12 or r == 13):
                x.mult *= 2
        elif kind == CK_COPY:
            t = self.jk[j].target
            if allow_copy and t >= 0:
                self._card(x, t, i, False)

    cdef int _retrig(self, Ctx* x, int j, int i, int p, bint allow_copy) noexcept:
        cdef int kind = self.jk[j].rk, t
        cdef const Crd* c = &self.cards[x.pos[i]]
        if kind == RK_CHAD:
            return 2 if p == 0 else 0
        elif kind == RK_HACK:
            return 1 if (not is_stone(c.enh) and 2 <= c.rank <= 5) else 0
        elif kind == RK_DUSK:
            return 1 if self.hands_left == 1 else 0
        elif kind == RK_SOCK:
            return 1 if is_face(c, self.par) else 0
        elif kind == RK_SELZER:
            return 1
        elif kind == RK_COPY:
            t = self.jk[j].target
            if allow_copy and t >= 0:
                return self._retrig(x, t, i, p, False)
        return 0

    cdef void _held(self, Ctx* x, int j, int h, bint allow_copy) noexcept:
        cdef int kind = self.jk[j].hk, t
        cdef const Crd* c = &self.cards[h]
        if kind == HK_SHOOT:
            if not is_stone(c.enh) and c.rank == 12:
                x.mult += 13
        elif kind == HK_BARON:
            if not is_stone(c.enh) and c.rank == 13:
                x.mult *= 1.5
        elif kind == HK_COPY:
            t = self.jk[j].target
            if allow_copy and t >= 0:
                self._held(x, t, h, False)

    cdef void _main(self, Ctx* x, int j, bint allow_copy) noexcept:
        cdef int kind = self.jk[j].mk, p, i, h, low, t, wild, need, nclub, nother, ncards
        cdef const Crd* c
        cdef bint ok
        cdef double v
        cdef long long m
        if kind == MK_NONE:
            return
        elif kind == MK_CONST:
            self._add(x, self.jk[j].p_kind, self.jk[j].p_amt)
        elif kind == MK_CONTAINS:
            if x.contains & (1 << self.jk[j].p_hand):
                self._add(x, self.jk[j].p_kind, self.jk[j].p_amt)
        elif kind == MK_HALF:
            if x.k <= 3:
                x.mult += 20
        elif kind == MK_BANNER:
            x.chips += 30 * self.discards_left
        elif kind == MK_MYSTIC:
            if self.discards_left == 0:
                x.mult += 15
        elif kind == MK_RAISED_FIST:
            low = -1                      # lowest rank held; the last one among ties
            for p in range(x.nh):
                h = x.held[p]
                if is_stone(self.cards[h].enh):
                    continue
                if low < 0 or self.cards[h].rank <= self.cards[low].rank:
                    low = h
            if low >= 0 and not self.cards[low].deb:
                x.mult += (2 * chip_value(&self.cards[low])) if self.cards[low].rank != 14 else 22
        elif kind == MK_ABSTRACT:
            x.mult += 3 * self.njokers
        elif kind == MK_SUPERNOVA:
            x.mult += self.hplayed[x.hand] + 1
        elif kind == MK_VAL_MULT:
            x.mult += self._get(x, j, 0)
        elif kind == MK_VAL_CHIPS:
            x.chips += self._get(x, j, 0)
        elif kind == MK_VAL_X:
            v = self._get(x, j, 1.0)
            x.mult *= v if v > 1.0 else 1.0
        elif kind == MK_BLUE:
            x.chips += 2 * self.deck_len
        elif kind == MK_SWASH:
            x.mult += self.sell_total - self.jk[j].sell
        elif kind == MK_FORTUNE:
            x.mult += self.tarots
        elif kind == MK_CARD_SHARP:
            if self.hplayed_round[x.hand] > 0:
                x.mult *= 3
        elif kind == MK_BULL:
            x.chips += 2 * (self.money if self.money > 0 else 0)
        elif kind == MK_BOOTSTRAPS:
            x.mult += 2 * ((self.money if self.money > 0 else 0) // 5)
        elif kind == MK_ACROBAT:
            if self.hands_left == 1:
                x.mult *= 3
        elif kind == MK_BLACKBOARD:
            ok = True
            for p in range(x.nh):
                c = &self.cards[x.held[p]]
                if is_stone(c.enh) or not (has_suit(c, 0, self.sm) or has_suit(c, 2, self.sm)):
                    ok = False
                    break
            if ok:
                x.mult *= 3
        elif kind == MK_FLOWER:
            need = 15                     # bitmask of suits still missing
            wild = 0
            for p in range(x.nsc):
                c = &self.cards[x.pos[x.scoring[p]]]
                if c.enh == E_WILD:
                    wild += 1
                elif not is_stone(c.enh) and 0 <= c.suit < 4:
                    need &= ~(1 << c.suit)
            if ((need & 1) + ((need >> 1) & 1) + ((need >> 2) & 1) + ((need >> 3) & 1)) <= wild:
                x.mult *= 3
        elif kind == MK_SEEING:
            ncards = nclub = nother = 0
            for p in range(x.nsc):
                c = &self.cards[x.pos[x.scoring[p]]]
                if is_stone(c.enh):
                    continue
                ncards += 1
                if has_suit(c, 2, self.sm):
                    nclub += 1
                if has_suit(c, 0, self.sm) or has_suit(c, 1, self.sm) or has_suit(c, 3, self.sm):
                    nother += 1
            if nclub and nother and ncards >= 2:
                x.mult *= 2
        elif kind == MK_STENCIL:
            m = self.slots - self.njokers + self.stencils
            x.mult *= m if m > 1 else 1
        elif kind == MK_LOYALTY:
            if self._get(x, j, 5) == 0:
                x.mult *= 4
        elif kind == MK_STEEL:
            x.mult *= 1 + 0.2 * self.steel
        elif kind == MK_EROSION:
            m = self.start_len - self.full_len
            x.mult += 4 * (m if m > 0 else 0)
        elif kind == MK_STONE:
            x.chips += 25 * self.stone
        elif kind == MK_LUCKY_CAT:
            self._set(x, j, self._get(x, j, 1.0) + 0.25 * x.lucky)
            x.mult *= x.val[j]
        elif kind == MK_BASEBALL:
            x.mult *= pow(1.5, <double>self.rare2)
        elif kind == MK_THROWBACK:
            x.mult *= 1 + 0.25 * self.skipped
        elif kind == MK_DRIVERS:
            if self.enhanced >= 16:
                x.mult *= 3
        elif kind == MK_COPY:
            t = self.jk[j].target
            if allow_copy and t >= 0:
                self._main(x, t, False)

    # ------------------------------------------------------------------ scoring
    cdef double _score(self, int* pos, int k, int* hand_out, bint* viol) noexcept:
        cdef Ctx x
        cdef const Crd* pl[5]
        cdef int i, j, p, r, e, reps, smask, level, ch, mu, h, vh, vs, vc
        cdef const Crd* c
        cdef bint in_play, steel_held
        cdef double avg
        x.k = k
        for i in range(k):
            x.pos[i] = pos[i]
            pl[i] = &self.cards[pos[i]]
            x.ov[i] = -1
        x.nh = 0
        for h in range(self.ncards):
            in_play = False
            for i in range(k):
                if pos[i] == h:
                    in_play = True
            if not in_play:
                x.held[x.nh] = h
                x.nh += 1
        evaluate(pl, k, self.ff, self.sc, self.sm, &x.hand, &smask, &x.contains)
        x.nsc = 0
        for i in range(k):
            if self.splash or (smask & (1 << i)):
                x.scoring[x.nsc] = i
                x.nsc += 1
        x.chips = x.mult = x.lucky = 0.0
        x.trig = 0
        for j in range(self.nj):          # joker state is copied per prediction
            x.val[j] = self.jk[j].val
            x.hv[j] = self.jk[j].has_val
            x.photo[j] = 0

        for p in range(self.n_before):
            self._before(&x, self.l_before[p])

        level = self.levels[x.hand]
        if self.boss == B_ARM:
            level = level - 1 if level - 1 > 1 else 1
        if level < 1:
            level = 1
        ch = HB[x.hand][0] + HB[x.hand][2] * (level - 1)
        mu = HB[x.hand][1] + HB[x.hand][3] * (level - 1)
        if self.boss == B_FLINT:
            ch = <int>(ch / 2.0 + 0.5)
            if ch < 0:
                ch = 0
            mu = <int>(mu / 2.0 + 0.5)
            if mu < 1:
                mu = 1
        x.chips = ch
        x.mult = mu

        # scored cards
        for p in range(x.nsc):
            i = x.scoring[p]
            c = pl[i]
            if c.deb:
                continue
            e = self._enh(&x, i)
            reps = 1 + (1 if c.seal == S_RED else 0)
            for j in range(self.n_retrig):
                reps += self._retrig(&x, self.l_retrig[j], i, p, True)
            for r in range(reps):
                x.trig += 1
                x.chips += (50 if e == E_STONE else chip_value(c)) + c.extra
                if e == E_BONUS:
                    x.chips += 30
                elif e == E_MULT:
                    x.mult += 4
                elif e == E_LUCKY:
                    x.mult += 20 * self.p5
                    x.lucky += self.p5
                    x.lucky += self.p15
                if c.ed == ED_FOIL:
                    x.chips += 50
                elif c.ed == ED_HOLO:
                    x.mult += 10
                for j in range(self.n_card):
                    self._card(&x, self.l_card[j], i, True)
                if e == E_GLASS:
                    x.mult *= 2
                if c.ed == ED_POLY:
                    x.mult *= 1.5

        # held-in-hand effects
        steel_held = False
        for p in range(x.nh):
            if self.cards[x.held[p]].enh == E_STEEL:
                steel_held = True
        if self.n_held > 0 or steel_held:
            for p in range(x.nh):
                h = x.held[p]
                c = &self.cards[h]
                if c.deb:
                    continue
                reps = 1 + (1 if c.seal == S_RED else 0) + self.mime
                for r in range(reps):
                    if c.enh == E_STEEL:
                        x.mult *= 1.5
                    for j in range(self.n_held):
                        self._held(&x, self.l_held[j], h, True)

        # Observatory: held planets for this hand
        for r in range(self.obs[x.hand]):
            x.mult *= 1.5

        # jokers (editions wrap each joker)
        for p in range(self.n_main):
            j = self.l_main[p]
            if self.jk[j].ed == ED_FOIL:
                x.chips += 50
            elif self.jk[j].ed == ED_HOLO:
                x.mult += 10
            self._main(&x, j, True)
            if self.jk[j].ed == ED_POLY:
                x.mult *= 1.5

        if self.plasma:
            avg = (x.chips + x.mult) / 2
            x.chips = x.mult = avg

        # Game.violates_boss
        viol[0] = False
        if self.vboss == B_PSYCHIC and k < 5:
            viol[0] = True
        elif self.vboss == B_EYE or self.vboss == B_MOUTH:
            evaluate(pl, k, self.vff, self.vsc, self.vsm, &vh, &vs, &vc)
            if self.vboss == B_EYE and (self.round_types & (1 << vh)):
                viol[0] = True
            elif self.vboss == B_MOUTH and self.mouth_hand >= 0 and vh != self.mouth_hand:
                viol[0] = True
        hand_out[0] = x.hand
        self.last_chips, self.last_mult = x.chips, x.mult
        return x.chips * x.mult
