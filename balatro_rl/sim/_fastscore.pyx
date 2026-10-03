# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled score prediction: a port of scoring.score_hand + hands.evaluate + the jokers' scoring
hooks for the expected-value case (rng=None, commit=False), plus Game.violates_boss.

The Python code is the reference. fastscore.py turns a game state into the flat tables used here,
and tests/test_fastscore.py checks the two agree. Effects that only move money are left out
(they never change the score)."""
cimport cython
from libc.math cimport pow, floor, isinf, isnan
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcmp, memcpy
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
    BK_DNA = 10
    BK_COPY = 11
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
    CK_HIKER = 16
    CK_LUCKY_CAT = 17
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
    HK_RAISED_FIST = 4
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
    BK_MIDAS=BK_MIDAS, BK_DNA=BK_DNA, BK_COPY=BK_COPY,
    CK_NONE=CK_NONE, CK_SUIT=CK_SUIT, CK_SCARY=CK_SCARY, CK_EVEN=CK_EVEN, CK_ODD=CK_ODD,
    CK_SCHOLAR=CK_SCHOLAR, CK_WALKIE=CK_WALKIE, CK_SMILEY=CK_SMILEY, CK_PHOTO=CK_PHOTO, CK_FIB=CK_FIB,
    CK_BLOOD=CK_BLOOD, CK_IDOL=CK_IDOL, CK_ANCIENT=CK_ANCIENT, CK_WEE=CK_WEE, CK_TRIB=CK_TRIB,
    CK_COPY=CK_COPY, CK_HIKER=CK_HIKER, CK_LUCKY_CAT=CK_LUCKY_CAT,
    RK_NONE=RK_NONE, RK_CHAD=RK_CHAD, RK_HACK=RK_HACK, RK_DUSK=RK_DUSK, RK_SOCK=RK_SOCK,
    RK_SELZER=RK_SELZER, RK_COPY=RK_COPY,
    HK_NONE=HK_NONE, HK_SHOOT=HK_SHOOT, HK_BARON=HK_BARON, HK_COPY=HK_COPY, HK_RAISED_FIST=HK_RAISED_FIST,
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
cdef int FLOWER_ORDER[4]
FLOWER_ORDER[0], FLOWER_ORDER[1], FLOWER_ORDER[2], FLOWER_ORDER[3] = 1, 3, 0, 2   # Hearts, Diamonds, Spades, Clubs
cdef int SEEING_ORDER[4]
SEEING_ORDER[0], SEEING_ORDER[1], SEEING_ORDER[2], SEEING_ORDER[3] = 2, 3, 0, 1   # Clubs, Diamonds, Spades, Hearts
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
    int unc             # Uncommon (Baseball Card multiplies its effect)

cdef struct Ctx:
    double chips, mult
    double lucky_now    # Lucky hits of the card trigger being scored
    int trig, hand, contains, k, nsc, nh
    int ncopy           # cards DNA added in this pass (kept after the real cards in Scorer.cards)
    int pos[5]          # view index of each played card
    int scoring[5]      # indices into the played list
    int held[MAXC + MAXJ]   # view indices of held cards
    int ov[5]           # enhancement override per played card (-1 = none)
    double hik[5]       # Hiker's chips gained in this hand, per played card
    double val[MAXJ]
    int hv[MAXJ]


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
    if is_stone(c.enh) or c.deb:          # a debuffed card is not a face card
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

cdef inline bint flush_suit(const Crd* c, int s, bint sm) noexcept nogil:
    # flushes and Blackboard: a debuffed Wild card is not wild
    if is_stone(c.enh):
        return False
    if c.enh == E_WILD and not c.deb:
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
    cdef int wild = 0, best_s, distinct = 0
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

    # flush: the first suit that makes one (live Wild cards count for every suit)
    if nn >= need:
        ns[0] = ns[1] = ns[2] = ns[3] = 0
        for j in range(nn):
            i = normal[j]
            if cs[i].enh == E_WILD and not cs[i].deb:
                wild += 1
            else:
                ns[cs[i].suit] += 1
        if sm:
            ns[0] = ns[2] = ns[0] + ns[2]
            ns[1] = ns[3] = ns[1] + ns[3]
        best_s = -1
        for s in range(4):
            if ns[s] + wild >= need:
                best_s = s
                break
        if best_s >= 0:
            for j in range(nn):
                i = normal[j]
                if (cs[i].enh == E_WILD and not cs[i].deb) or (cs[i].suit % 2 == best_s % 2 if sm
                                                              else cs[i].suit == best_s):
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
    if top >= 2 and second >= 2:                   # two separate groups: Four / Five of a Kind don't count
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


# ------------------------------------------------------------------ hand-detection cache
# evaluate() for every subset pattern of a loaded hand, keyed on the hand's cards (rank, suit, enhancement,
# debuff: everything evaluate reads) and the flags. The pricer scores the same sampled hands for every option
# of a shop decision and the solver the same hands across futures, each with a new Scorer: the cache is
# module-level, so only the jokers' pass runs per option. Direct-mapped; a slot holds one packed int per
# subset: hand type (4 bits), the boss check's hand type (4), the scoring mask (5), contained hands (12).
cdef enum:
    CACHE_SLOTS = 2048
    NO_PRE = -1

cdef struct HSlot:
    int n
    int flags
    unsigned int cards[MAXHAND]
    unsigned int* pre          # PAT_N[n] packed entries, or NULL
    int cap

cdef HSlot* CACHE = <HSlot*> calloc(CACHE_SLOTS, sizeof(HSlot))
if CACHE == NULL:
    raise MemoryError()
cdef long long CACHE_HITS = 0, CACHE_MISSES = 0


def cache_stats():
    """(hits, misses) of the hand-detection cache."""
    return CACHE_HITS, CACHE_MISSES


cdef inline unsigned int _pack_card(const Crd* c) noexcept nogil:
    return (<unsigned int>c.rank) | ((<unsigned int>c.suit) << 4) | ((<unsigned int>c.enh) << 8) \
        | ((<unsigned int>(1 if c.deb else 0)) << 12)

cdef inline long long _pre_of(unsigned int v) noexcept nogil:
    return <long long>v

cdef inline int PRE_HAND(long long p) noexcept nogil:
    return <int>(p & 15)

cdef inline int PRE_VHAND(long long p) noexcept nogil:
    return <int>((p >> 4) & 15)

cdef inline int PRE_SMASK(long long p) noexcept nogil:
    return <int>((p >> 8) & 31)

cdef inline int PRE_CONTAINS(long long p) noexcept nogil:
    return <int>((p >> 16) & 4095)


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
    cdef Crd cards[MAXC + MAXJ]          # the hand, then room for the copies DNA makes while scoring
    cdef int ncards
    cdef Jkr jk[MAXJ]
    cdef int nj
    cdef int l_before[MAXJ]
    cdef int l_card[MAXJ]
    cdef int l_retrig[MAXJ]
    cdef int l_held[MAXJ]
    cdef int l_main[MAXJ]
    cdef int l_holo[MAXJ]                # active Holograms (they grow when DNA adds a card)
    cdef int n_before, n_card, n_retrig, n_held, n_main, n_holo
    cdef bint first_hand                 # no hand played yet this round
    cdef bint ff, sc, sm, par, splash
    cdef int mime
    cdef bint vff, vsc, vsm
    cdef int boss, vboss, round_types, mouth_hand
    cdef bint hook, blackboard           # The Hook is the boss; a Blackboard is among the jokers
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
    cdef unsigned int* hpre              # hand-detection cache entries of the loaded hand (pooled calls)
    # pruning bound (see _bound_setup): state-level parts
    cdef bint bnd_ok                     # the bound is exact for every joker on the table
    cdef double b_chips, b_mult, b_x, b_half
    cdef double b_cont_c[12]
    cdef double b_cont_m[12]
    cdef double b_cont_x[12]
    cdef double b_hand_m[12]             # per hand type: Supernova
    cdef double b_hand_x[12]             # per hand type: Card Sharp, Observatory
    cdef double b_base_c[12]
    cdef double b_base_m[12]
    cdef int b_chad, b_wee, b_cat, b_hiker
    cdef double b_cat_val[MAXJ]
    cdef int b_lucky[MAXHAND]
    cdef double b_rf                     # Raised Fist instances (held hook)
    cdef double b_shoot, b_baron
    cdef int b_dusk, b_selzer, b_hack, b_sock
    # per card of the loaded hand
    cdef double b_pc[MAXHAND]
    cdef double b_pm[MAXHAND]
    cdef double b_px[MAXHAND]
    cdef double b_hm[MAXHAND]
    cdef double b_hx[MAXHAND]
    cdef int b_r[MAXHAND]

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
             self.jk[i].has_val, self.jk[i].val, self.jk[i].sell, self.jk[i].unc) = j
        before, card, retrig, held, main, holo = lists
        self.n_holo = len(holo)
        for i, x in enumerate(holo):
            self.l_holo[i] = x
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
        self.boss, self.vboss, self.round_types, self.mouth_hand, self.hook, self.blackboard = boss
        levels, hplayed, hplayed_round, obs = arrays
        for i in range(12):
            self.levels[i] = levels[i]
            self.hplayed[i] = hplayed[i]
            self.hplayed_round[i] = hplayed_round[i]
            self.obs[i] = obs[i]
        self.first_hand = sum(hplayed_round) == 0
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
            v = self._score(pos, k, &hand, &viol, NO_PRE)
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
        self.hpre = self._lookup(n)
        return n

    cdef unsigned int* _lookup(self, int n) except NULL:
        """The hand-detection cache entries of the loaded hand (filled on a miss)."""
        global CACHE_HITS, CACHE_MISSES
        cdef unsigned int key[MAXHAND]
        cdef unsigned long long h = 1469598103934665603ULL
        cdef int i, t, k, off, m, flags, hand, smask, contains, vh, vs, vc
        cdef bint needv = self.vboss == B_EYE or self.vboss == B_MOUTH
        cdef int pos[5]
        cdef const Crd* pl[5]
        cdef HSlot* slot
        flags = ((1 if self.ff else 0) | (2 if self.sc else 0) | (4 if self.sm else 0) | (8 if self.vff else 0)
                 | (16 if self.vsc else 0) | (32 if self.vsm else 0) | (64 if needv else 0))
        for i in range(n):
            key[i] = _pack_card(&self.cards[i])
            h = (h ^ key[i]) * 1099511628211ULL
        h = (h ^ <unsigned long long>flags) * 1099511628211ULL
        h = (h ^ <unsigned long long>n) * 1099511628211ULL
        slot = &CACHE[(h >> 17) % CACHE_SLOTS]
        if (slot.pre != NULL and slot.n == n and slot.flags == flags
                and memcmp(slot.cards, key, n * sizeof(unsigned int)) == 0):
            CACHE_HITS += 1
            return slot.pre
        CACHE_MISSES += 1
        off, m = PAT_OFF[n], PAT_N[n]
        if slot.cap < m:
            if slot.pre != NULL:
                free(slot.pre)
            slot.pre = <unsigned int*> malloc(m * sizeof(unsigned int))
            if slot.pre == NULL:
                slot.cap = 0
                raise MemoryError()
            slot.cap = m
        slot.n, slot.flags = n, flags
        memcpy(slot.cards, key, n * sizeof(unsigned int))
        for t in range(m):
            k = PAT_K[off + t]
            for i in range(k):
                pl[i] = &self.cards[PAT_POS[5 * (off + t) + i]]
            evaluate(pl, k, self.ff, self.sc, self.sm, &hand, &smask, &contains)
            vh = 0
            if needv:
                evaluate(pl, k, self.vff, self.vsc, self.vsm, &vh, &vs, &vc)
            slot.pre[t] = ((<unsigned int>hand) | ((<unsigned int>vh) << 4) | ((<unsigned int>smask) << 8)
                           | ((<unsigned int>contains) << 16))
        return slot.pre

    cdef list _best_two(self, int n):
        """[(score, hand type, subset index)] for the two best plays, best first; ties keep the earlier
        subset (what a stable sort by score gives). The subsets are visited largest first (they tend to
        score highest) and a subset whose upper bound (_bound) is below the second best is skipped; the
        tie rule is applied explicitly, so the result is the same as scoring every subset in order."""
        cdef int off = PAT_OFF[n], m = PAT_N[n], t, i, k, hand, h1 = -1, h2 = -1, b1 = -1, b2 = -1
        cdef int pos[5]
        cdef double v, f, s1 = -1e308, s2 = -1e308, bound
        cdef bint viol, v1 = False, v2 = False
        cdef long long pre
        self._bound_setup(n)
        for t in range(m - 1, -1, -1):
            pre = _pre_of(self.hpre[t])
            k = PAT_K[off + t]
            for i in range(k):
                pos[i] = PAT_POS[5 * (off + t) + i]
            if b2 >= 0 and self.bnd_ok:
                bound = self._bound(pos, k, pre)
                if bound < s2:
                    continue
            v = self._score(pos, k, &hand, &viol, pre)
            f = floor(v)
            _check_finite(f)
            if viol:
                f = 0.0
            if b1 < 0 or f > s1 or (f == s1 and t < b1):
                b2, s2, h2, v2 = b1, s1, h1, v1
                b1, s1, h1, v1 = t, f, hand, viol
            elif b2 < 0 or f > s2 or (f == s2 and t < b2):
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
            v = self._score(pos, k, &h, &viol, _pre_of(self.hpre[t]))
            f = floor(v)
            _check_finite(f)
            out.append((_score_obj(f, viol), h))
        return out

    def bounds(self, hand, long long hands_left, long long discards_left, long long deck_len,
               int round_types, int mouth_hand):
        """[(upper bound, score)] per subset of one hand (pool indices), in subset_patterns order: the
        pruning bound _best_two uses against the real score (bound is inf when the bound is off). For
        tests: bound >= score must hold everywhere."""
        cdef int n = self._load(hand, hands_left, discards_left, deck_len, round_types, mouth_hand)
        cdef int off = PAT_OFF[n], m = PAT_N[n], t, i, k, h
        cdef int pos[5]
        cdef double v, f, b
        cdef bint viol
        cdef long long pre
        self._bound_setup(n)
        out = []
        for t in range(m):
            k = PAT_K[off + t]
            for i in range(k):
                pos[i] = PAT_POS[5 * (off + t) + i]
            pre = _pre_of(self.hpre[t])
            b = self._bound(pos, k, pre) if self.bnd_ok else float("inf")
            v = self._score(pos, k, &h, &viol, pre)
            f = floor(v)
            if viol:
                f = 0.0
            out.append((b, f))
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
            v = self._score(pos, k, &hand, &viol, NO_PRE)
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
    cdef void _dna(self, Ctx* x) noexcept:
        # first hand of the round, one card played: its copy is held while this hand scores, and every
        # active Hologram has already grown
        cdef int q, p, h
        if not self.first_hand or x.k != 1:
            return
        q = self.ncards + x.ncopy
        self.cards[q] = self.cards[x.pos[0]]
        self.cards[q].enh = self._enh(x, 0)
        x.ncopy += 1
        x.held[x.nh] = q
        x.nh += 1
        for p in range(self.n_holo):
            h = self.l_holo[p]
            self._set(x, h, self._get(x, h, 1.0) + 0.25)

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
            v = self._get(x, j, 6)            # hands to go; fires on hands 6, 12, 18 ...
            self._set(x, j, v - 1 if v > 0 else 5)
        elif kind == BK_OBELISK:
            most = 0                      # resets when this hand is played at least as often as any
            for i in range(12):           # other (the first hand of a run included)
                if self.hplayed[i] > most:
                    most = self.hplayed[i]
            if self.hplayed[x.hand] == most:
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
        elif kind == BK_DNA:
            self._dna(x)
        elif kind == BK_COPY:                 # of the "before" effects, only DNA's changes the score
            i = self.jk[j].target
            if i >= 0 and self.jk[i].bk == BK_DNA:
                self._dna(x)

    cdef void _card(self, Ctx* x, int j, int i, bint allow_copy) noexcept:
        """Per-card effect of joker j (its own state) on played card i. allow_copy is False when this is
        a Blueprint / Brainstorm copy of j: the copy gives the effect but never grows j."""
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
            if first == i:
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
            if allow_copy and not st and r == 2:
                self._set(x, j, self._get(x, j, 0) + 8)
        elif kind == CK_HIKER:
            x.hik[i] += 5
        elif kind == CK_LUCKY_CAT:
            if allow_copy and x.lucky_now != 0:
                self._set(x, j, self._get(x, j, 1.0) + 0.25 * x.lucky_now)
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
        cdef int kind = self.jk[j].hk, t, p, q, low
        cdef const Crd* c = &self.cards[h]
        if kind == HK_RAISED_FIST:
            low = -1                      # lowest-ranked held card, the rightmost one if tied
            for p in range(x.nh):
                q = x.held[p]
                if is_stone(self.cards[q].enh):
                    continue
                if low < 0 or self.cards[q].rank <= self.cards[low].rank:
                    low = q
            if low == h:
                x.mult += 2 * chip_value(c)
        elif kind == HK_SHOOT:
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
                if is_stone(c.enh) or not (flush_suit(c, 0, self.sm) or flush_suit(c, 2, self.sm)):
                    ok = False
                    break
            if ok:
                x.mult *= 3
        elif kind == MK_FLOWER:
            # non-Wild cards, then Wild cards; each fills the first empty suit it matches, in the order
            # Hearts, Diamonds, Spades, Clubs (suit indices 1, 3, 0, 2)
            need = 0                      # bitmask of suits filled
            for wild in range(2):
                for p in range(x.nsc):
                    c = &self.cards[x.pos[x.scoring[p]]]
                    if (c.enh == E_WILD) != (wild == 1):
                        continue
                    for i in range(4):
                        h = FLOWER_ORDER[i]
                        if not (need & (1 << h)) and has_suit(c, h, self.sm):
                            need |= 1 << h
                            break
            if need == 15:
                x.mult *= 3
        elif kind == MK_SEEING:
            # debuffed cards count for nothing; other non-Wild cards for every suit they match; then each
            # Wild card fills the first missing suit in the order Clubs, Diamonds, Spades, Hearts
            need = 0                      # bitmask of suits present
            wild = 0
            for p in range(x.nsc):
                c = &self.cards[x.pos[x.scoring[p]]]
                if c.deb or is_stone(c.enh):
                    continue
                if c.enh == E_WILD:
                    wild += 1
                    continue
                for h in range(4):
                    if has_suit(c, h, self.sm):
                        need |= 1 << h
            for p in range(wild):
                for i in range(4):
                    h = SEEING_ORDER[i]
                    if not (need & (1 << h)):
                        need |= 1 << h
                        break
            if (need & 4) and (need & 11):
                x.mult *= 2
        elif kind == MK_STENCIL:
            m = self.slots - self.njokers + self.stencils
            x.mult *= m if m > 1 else 1
        elif kind == MK_LOYALTY:
            if self._get(x, j, 6) == 0:
                x.mult *= 4
        elif kind == MK_STEEL:
            x.mult *= 1 + 0.2 * self.steel
        elif kind == MK_EROSION:
            m = self.start_len - self.full_len
            x.mult += 4 * (m if m > 0 else 0)
        elif kind == MK_STONE:
            x.chips += 25 * self.stone
        elif kind == MK_LUCKY_CAT:
            x.mult *= self._get(x, j, 1.0)
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

    # ------------------------------------------------------------------ pruning bound
    # An upper bound on chips x mult of a play, cheap enough to skip the full pass when it cannot beat the
    # second-best play found so far. Every add in the real pass is >= 0 and every multiplier >= 1 (Lucky
    # Cat's is clamped here), so whatever the order of adds and multiplies, the final mult is at most
    # (base + every add) x (every multiplier), and the chips at most base + every add. Conditional effects
    # count as if they fired; effects that depend on the cards are summed per card (as if every played card
    # scored with its most retriggers); the held effects of every card that is not played are included.
    # DNA is the one joker without a bound (it adds a held copy and grows Holograms): bnd_ok is off then.

    cdef inline int _copy_target(self, int j) noexcept:
        return self.jk[j].target

    cdef double _val_bound(self, int t, double default) noexcept:
        """The largest value joker t's "val" can have when its main effect reads it."""
        cdef double v = self.jk[t].val if self.jk[t].has_val else default
        cdef int bk = self.jk[t].bk
        if bk == BK_BUS or bk == BK_GREEN:
            v += 1
        elif bk == BK_RUNNER:
            v += 15
        elif bk == BK_SQUARE:
            v += 4
        elif bk == BK_TROUSERS:
            v += 2
        elif bk == BK_OBELISK:
            v = v + 0.2 if v + 0.2 > 1.0 else 1.0
        elif bk == BK_VAMPIRE:
            v += 0.5
        return v

    cdef void _main_bound(self, int t, int kind) noexcept:
        cdef int h, pk = self.jk[t].p_kind
        cdef double v, amt = self.jk[t].p_amt
        cdef long long m
        if kind == MK_CONST:
            if pk == K_MULT:
                self.b_mult += amt
            elif pk == K_CHIPS:
                self.b_chips += amt
            else:
                self.b_x *= amt if amt > 1 else 1
        elif kind == MK_CONTAINS:
            h = self.jk[t].p_hand
            if pk == K_MULT:
                self.b_cont_m[h] += amt
            elif pk == K_CHIPS:
                self.b_cont_c[h] += amt
            else:
                self.b_cont_x[h] *= amt if amt > 1 else 1
        elif kind == MK_HALF:
            self.b_half += 20
        elif kind == MK_BANNER:
            self.b_chips += 30 * self.discards_left
        elif kind == MK_MYSTIC:
            if self.discards_left == 0:
                self.b_mult += 15
        elif kind == MK_RAISED_FIST:
            self.b_mult += 22
        elif kind == MK_ABSTRACT:
            self.b_mult += 3 * self.njokers
        elif kind == MK_SUPERNOVA:
            for h in range(12):
                self.b_hand_m[h] += self.hplayed[h] + 1
        elif kind == MK_VAL_MULT:
            v = self._val_bound(t, 0.0)
            self.b_mult += v if v > 0 else 0
        elif kind == MK_VAL_CHIPS:
            v = self._val_bound(t, 0.0)
            self.b_chips += v if v > 0 else 0
            if self.jk[t].ck == CK_WEE:
                self.b_wee += 1
        elif kind == MK_VAL_X:
            v = self._val_bound(t, 1.0)
            self.b_x *= v if v > 1 else 1
        elif kind == MK_BLUE:
            self.b_chips += 2 * self.deck_len
        elif kind == MK_SWASH:
            m = self.sell_total - self.jk[t].sell
            self.b_mult += m if m > 0 else 0
        elif kind == MK_FORTUNE:
            self.b_mult += self.tarots if self.tarots > 0 else 0
        elif kind == MK_CARD_SHARP:
            for h in range(12):
                if self.hplayed_round[h] > 0:
                    self.b_hand_x[h] *= 3
        elif kind == MK_BULL:
            self.b_chips += 2 * (self.money if self.money > 0 else 0)
        elif kind == MK_BOOTSTRAPS:
            self.b_mult += 2 * ((self.money if self.money > 0 else 0) // 5)
        elif kind == MK_ACROBAT:
            if self.hands_left == 1:
                self.b_x *= 3
        elif kind == MK_BLACKBOARD or kind == MK_FLOWER:
            self.b_x *= 3
        elif kind == MK_SEEING:
            self.b_x *= 2
        elif kind == MK_STENCIL:
            m = self.slots - self.njokers + self.stencils
            self.b_x *= m if m > 1 else 1
        elif kind == MK_LOYALTY:
            v = self.jk[t].val if self.jk[t].has_val else 6
            if v == 1 or v == 0:
                self.b_x *= 4
        elif kind == MK_STEEL:
            self.b_x *= 1 + 0.2 * self.steel
        elif kind == MK_EROSION:
            m = self.start_len - self.full_len
            self.b_mult += 4 * (m if m > 0 else 0)
        elif kind == MK_STONE:
            self.b_chips += 25 * self.stone
        elif kind == MK_LUCKY_CAT:
            self.b_cat_val[self.b_cat] = self.jk[t].val if self.jk[t].has_val else 1.0
            self.b_cat += 1
        elif kind == MK_BASEBALL:
            self.b_x *= pow(1.5, <double>self.rare2)
        elif kind == MK_THROWBACK:
            self.b_x *= 1 + 0.25 * self.skipped
        elif kind == MK_DRIVERS:
            if self.enhanced >= 16:
                self.b_x *= 3

    cdef void _bound_setup(self, int n) noexcept:
        """State-level and per-card parts of the bound for the loaded hand."""
        cdef int j, t, p, i, h, kind, r, level, ch, mu, e, hiker = 0
        cdef const Crd* c
        cdef bint st
        cdef double pc, pm, px, hm, hx, reps
        self.bnd_ok = True
        self.b_chips = self.b_mult = self.b_half = 0.0
        self.b_x = 1.0
        for h in range(12):
            self.b_cont_c[h] = self.b_cont_m[h] = self.b_hand_m[h] = 0.0
            self.b_cont_x[h] = 1.0
            self.b_hand_x[h] = pow(1.5, <double>self.obs[h])
            level = self.levels[h]
            if self.boss == B_ARM:
                level = level - 1 if level - 1 > 1 else 1
            if level < 1:
                level = 1
            ch = HB[h][0] + HB[h][2] * (level - 1)
            mu = HB[h][1] + HB[h][3] * (level - 1)
            if self.boss == B_FLINT:
                ch = <int>(ch / 2.0 + 0.5)
                if ch < 0:
                    ch = 0
                mu = <int>(mu / 2.0 + 0.5)
                if mu < 1:
                    mu = 1
            self.b_base_c[h] = ch
            self.b_base_m[h] = mu
        self.b_chad = self.b_wee = self.b_cat = 0
        self.b_rf = self.b_shoot = self.b_baron = 0.0
        self.b_dusk = self.b_selzer = self.b_hack = self.b_sock = 0
        for p in range(self.n_before):
            j = self.l_before[p]
            kind = self.jk[j].bk
            if kind == BK_COPY:
                t = self._copy_target(j)
                kind = self.jk[t].bk if t >= 0 else BK_NONE
            if kind == BK_DNA:
                self.bnd_ok = False
        for p in range(self.n_card):
            j = self.l_card[p]
            kind = self.jk[j].ck
            if kind == CK_COPY:
                t = self._copy_target(j)
                kind = self.jk[t].ck if t >= 0 else CK_NONE
            if kind == CK_HIKER:
                hiker += 1
        self.b_hiker = hiker
        for p in range(self.n_retrig):
            j = self.l_retrig[p]
            kind = self.jk[j].rk
            if kind == RK_COPY:
                t = self._copy_target(j)
                kind = self.jk[t].rk if t >= 0 else RK_NONE
            if kind == RK_CHAD:
                self.b_chad += 1
            elif kind == RK_HACK:
                self.b_hack += 1
            elif kind == RK_DUSK:
                if self.hands_left == 1:
                    self.b_dusk += 1
            elif kind == RK_SOCK:
                self.b_sock += 1
            elif kind == RK_SELZER:
                self.b_selzer += 1
        for p in range(self.n_held):
            j = self.l_held[p]
            kind = self.jk[j].hk
            if kind == HK_COPY:
                t = self._copy_target(j)
                kind = self.jk[t].hk if t >= 0 else HK_NONE
            if kind == HK_RAISED_FIST:
                self.b_rf += 1
            elif kind == HK_SHOOT:
                self.b_shoot += 13
            elif kind == HK_BARON:
                self.b_baron += 1
        for p in range(self.n_main):
            j = self.l_main[p]
            if self.jk[j].ed == ED_FOIL:
                self.b_chips += 50
            elif self.jk[j].ed == ED_HOLO:
                self.b_mult += 10
            elif self.jk[j].ed == ED_POLY:
                self.b_x *= 1.5
            if self.jk[j].unc:
                self.b_x *= pow(1.5, <double>self.rare2)
            t = j
            kind = self.jk[j].mk
            if kind == MK_COPY:
                t = self._copy_target(j)
                kind = self.jk[t].mk if t >= 0 else MK_NONE
                if kind == MK_COPY:
                    kind = MK_NONE
            if kind != MK_NONE:
                self._main_bound(t, kind)
        # per card: retriggers, what each trigger adds, what it adds when held
        for i in range(n):
            c = &self.cards[i]
            self.b_r[i] = 0
            self.b_pc[i] = self.b_pm[i] = self.b_hm[i] = 0.0
            self.b_px[i] = self.b_hx[i] = 1.0
            self.b_lucky[i] = 0
            if c.deb:
                continue
            st = is_stone(c.enh)
            r = 1 + (1 if c.seal == S_RED else 0) + self.b_selzer + self.b_dusk
            if not st and 2 <= c.rank <= 5:
                r += self.b_hack
            if is_face(c, self.par):
                r += self.b_sock
            self.b_r[i] = r
            e = c.enh
            pc = chip_value(c) + c.extra
            pm = 0.0
            px = 1.0
            if e == E_BONUS:
                pc += 30
            elif e == E_MULT:
                pm += 4
            elif e == E_LUCKY:
                pm += 20 * self.p5
                self.b_lucky[i] = 1
            if e == E_GLASS:
                px *= 2
            if c.ed == ED_FOIL:
                pc += 50
            elif c.ed == ED_HOLO:
                pm += 10
            elif c.ed == ED_POLY:
                px *= 1.5
            if not st and c.rank == 2:
                pc += 8 * self.b_wee
            for p in range(self.n_card):
                j = self.l_card[p]
                t = j
                kind = self.jk[j].ck
                if kind == CK_COPY:
                    t = self._copy_target(j)
                    kind = self.jk[t].ck if t >= 0 else CK_NONE
                    if kind == CK_COPY or kind == CK_WEE or kind == CK_LUCKY_CAT:
                        kind = CK_NONE
                if kind == CK_SUIT:
                    if has_suit(c, self.jk[t].p_suit, self.sm):
                        if self.jk[t].p_kind == K_MULT:
                            pm += self.jk[t].p_amt
                        elif self.jk[t].p_kind == K_CHIPS:
                            pc += self.jk[t].p_amt
                        else:
                            px *= self.jk[t].p_amt if self.jk[t].p_amt > 1 else 1
                elif kind == CK_SCARY:
                    if is_face(c, self.par):
                        pc += 30
                elif kind == CK_EVEN:
                    if not st and c.rank <= 10 and c.rank % 2 == 0:
                        pm += 4
                elif kind == CK_ODD:
                    if not st and (c.rank == 14 or (c.rank <= 10 and c.rank % 2 == 1)):
                        pc += 31
                elif kind == CK_SCHOLAR:
                    if not st and c.rank == 14:
                        pc += 20
                        pm += 4
                elif kind == CK_WALKIE:
                    if not st and (c.rank == 10 or c.rank == 4):
                        pc += 10
                        pm += 4
                elif kind == CK_SMILEY:
                    if is_face(c, self.par):
                        pm += 5
                elif kind == CK_PHOTO:
                    if is_face(c, self.par):
                        px *= 2
                elif kind == CK_FIB:
                    if not st and (c.rank == 14 or c.rank == 2 or c.rank == 3 or c.rank == 5 or c.rank == 8):
                        pm += 8
                elif kind == CK_BLOOD:
                    if has_suit(c, 1, self.sm):
                        px *= 1 + (1.5 - 1) * self.p2
                elif kind == CK_IDOL:
                    if not st and c.rank == self.jk[t].st_rank and has_suit(c, self.jk[t].st_suit, self.sm):
                        px *= 2
                elif kind == CK_ANCIENT:
                    if has_suit(c, self.jk[t].st_suit, self.sm):
                        px *= 1.5
                elif kind == CK_TRIB:
                    if not st and (c.rank == 12 or c.rank == 13):
                        px *= 2
            self.b_pc[i], self.b_pm[i], self.b_px[i] = pc, pm, px
            # held (every card that is not played): Steel, Shoot the Moon, Raised Fist, Baron, each repeated
            # by a Red seal and Mime
            reps = 1 + (1 if c.seal == S_RED else 0) + self.mime
            hm = 0.0
            hx = 1.5 if c.enh == E_STEEL else 1.0
            if not st:
                hm += self.b_rf * 2 * chip_value(c)
                if c.rank == 12:
                    hm += self.b_shoot
                if c.rank == 13:
                    hx *= pow(1.5, self.b_baron)
            self.b_hm[i] = hm * reps
            self.b_hx[i] = pow(hx, reps)

    cdef double _bound(self, int* pos, int k, long long pre) noexcept:
        """Upper bound on chips x mult of the play `pos` of the loaded hand (after _bound_setup)."""
        cdef int hand = PRE_HAND(pre), smask = PRE_SMASK(pre), contains = PRE_CONTAINS(pre), i, q, h, r
        cdef double C, M, X, v, lucky = 0.0
        cdef long long played = 0
        cdef bint first = True
        C = self.b_base_c[hand] + self.b_chips
        M = self.b_base_m[hand] + self.b_mult + self.b_hand_m[hand]
        X = self.b_x * self.b_hand_x[hand]
        if k <= 3:
            M += self.b_half
        for h in range(12):
            if contains & (1 << h):
                C += self.b_cont_c[h]
                M += self.b_cont_m[h]
                X *= self.b_cont_x[h]
        for i in range(k):
            q = pos[i]
            played |= (<long long>1) << q
            if not (self.splash or (smask & (1 << i))):
                continue
            r = self.b_r[q]
            if first:                     # Hanging Chad: the first scoring card, if it is not debuffed
                first = False
                if r > 0:
                    r += 2 * self.b_chad
            if r == 0:
                continue
            C += r * self.b_pc[q]
            if self.b_hiker:
                C += 5.0 * self.b_hiker * r * (r - 1) / 2
            M += r * self.b_pm[q]
            if self.b_px[q] != 1.0:
                X *= pow(self.b_px[q], <double>r)
            if self.b_lucky[q]:
                lucky += r
        for q in range(self.ncards):
            if not (played & ((<long long>1) << q)):
                M += self.b_hm[q]
                X *= self.b_hx[q]
        for i in range(self.b_cat):
            v = self.b_cat_val[i] + 0.25 * (self.p5 + self.p15) * lucky
            X *= v if v > 1 else 1
        if self.plasma:
            v = (C + M * X) / 2
            v = v * v
        else:
            v = C * M * X
        return v * (1 + 1e-9) + 0.5

    # ------------------------------------------------------------------ scoring
    cdef double _score(self, int* pos, int k, int* hand_out, bint* viol, long long pre) noexcept:
        # chips x mult of one play. Under The Hook, when the held cards can change the score, the mean
        # over every pair of held cards it could discard first (scoring.hook_variants / _hook_expected).
        cdef int nh = self.ncards - k, a, b, h, i, n = 0
        cdef bint matters = False, in_play
        cdef double total = 0.0, chips = 0.0, mult = 0.0
        if self.hook and nh > 0:
            matters = self.n_held > 0 or self.blackboard
            if not matters:
                for h in range(self.ncards):
                    in_play = False
                    for i in range(k):
                        if pos[i] == h:
                            in_play = True
                    if not in_play and self.cards[h].enh == E_STEEL:
                        matters = True
        if not matters:
            return self._score1(pos, k, hand_out, viol, -1, -1, pre)
        if nh <= 2:
            return self._score1(pos, k, hand_out, viol, -2, -2, pre)
        for a in range(nh):
            for b in range(a + 1, nh):
                total += self._score1(pos, k, hand_out, viol, a, b, pre)
                chips += self.last_chips
                mult += self.last_mult
                n += 1
        self.last_chips, self.last_mult = chips / n, mult / n
        return total / n

    cdef double _score1(self, int* pos, int k, int* hand_out, bint* viol, int skip_a, int skip_b,
                        long long pre) noexcept:
        # One scoring pass. skip_a / skip_b: held cards (by their place among the held cards) The Hook
        # discarded first; -1 for none, -2 for all of them. pre: the hand-detection cache entry of this
        # play (NO_PRE: evaluate here).
        cdef Ctx x
        cdef const Crd* pl[5]
        cdef int i, j, p, r, e, reps, smask, level, ch, mu, h, vh, vs, vc, bc, hi = 0
        cdef const Crd* c
        cdef bint in_play, steel_held
        cdef double avg, hc, hm
        x.k = k
        for i in range(k):
            x.pos[i] = pos[i]
            pl[i] = &self.cards[pos[i]]
            x.ov[i] = -1
            x.hik[i] = 0.0
        x.nh = 0
        x.ncopy = 0
        for h in range(self.ncards):
            in_play = False
            for i in range(k):
                if pos[i] == h:
                    in_play = True
            if not in_play:
                if skip_a != -2 and hi != skip_a and hi != skip_b:
                    x.held[x.nh] = h
                    x.nh += 1
                hi += 1
        if pre == NO_PRE:
            evaluate(pl, k, self.ff, self.sc, self.sm, &x.hand, &smask, &x.contains)
        else:
            x.hand, smask, x.contains = PRE_HAND(pre), PRE_SMASK(pre), PRE_CONTAINS(pre)
        x.nsc = 0
        for i in range(k):
            if self.splash or (smask & (1 << i)):
                x.scoring[x.nsc] = i
                x.nsc += 1
        x.chips = x.mult = x.lucky_now = 0.0
        x.trig = 0
        for j in range(self.nj):          # joker state is copied per prediction
            x.val[j] = self.jk[j].val
            x.hv[j] = self.jk[j].has_val

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
                x.lucky_now = 0.0
                if e == E_STONE:
                    bc = 50
                elif c.enh == E_STONE:        # a Stone card Vampire stripped: its rank counts again
                    bc = 11 if c.rank == 14 else (c.rank if c.rank < 10 else 10)
                else:
                    bc = chip_value(c)
                x.chips += bc + c.extra + x.hik[i]
                if e == E_BONUS:
                    x.chips += 30
                elif e == E_MULT:
                    x.mult += 4
                elif e == E_LUCKY:
                    x.mult += 20 * self.p5
                    x.lucky_now += self.p5
                    x.lucky_now += self.p15
                if e == E_GLASS:              # the card: chips, mult, Glass, then its edition ...
                    x.mult *= 2
                if c.ed == ED_FOIL:
                    x.chips += 50
                elif c.ed == ED_HOLO:
                    x.mult += 10
                elif c.ed == ED_POLY:
                    x.mult *= 1.5
                for j in range(self.n_card):  # ... then the jokers' per-card effects
                    self._card(&x, self.l_card[j], i, True)

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
                hc, hm = x.chips, x.mult          # Red seals / Mime repeat it only if it had an effect
                if c.enh == E_STEEL:
                    x.mult *= 1.5
                for j in range(self.n_held):
                    self._held(&x, self.l_held[j], h, True)
                if c.enh != E_STEEL and x.chips == hc and x.mult == hm:
                    continue
                reps = (1 if c.seal == S_RED else 0) + self.mime
                for r in range(reps):
                    if c.enh == E_STEEL:
                        x.mult *= 1.5
                    for j in range(self.n_held):
                        self._held(&x, self.l_held[j], h, True)

        # jokers: edition Foil / Holo, effect, Baseball Card (per Baseball, on Uncommons), edition Polychrome
        for p in range(self.n_main):
            j = self.l_main[p]
            if self.jk[j].ed == ED_FOIL:
                x.chips += 50
            elif self.jk[j].ed == ED_HOLO:
                x.mult += 10
            self._main(&x, j, True)
            if self.jk[j].unc:
                for r in range(self.rare2):
                    x.mult *= 1.5
            if self.jk[j].ed == ED_POLY:
                x.mult *= 1.5

        # Observatory: held planets for this hand, after the last joker
        for r in range(self.obs[x.hand]):
            x.mult *= 1.5

        if self.plasma:                   # both become floor((chips + mult) / 2)
            avg = floor((x.chips + x.mult) / 2)
            x.chips = x.mult = avg

        # Game.violates_boss
        viol[0] = False
        if self.vboss == B_PSYCHIC and k < 5:
            viol[0] = True
        elif self.vboss == B_EYE or self.vboss == B_MOUTH:
            if pre == NO_PRE:
                evaluate(pl, k, self.vff, self.vsc, self.vsm, &vh, &vs, &vc)
            else:
                vh = PRE_VHAND(pre)
            if self.vboss == B_EYE and (self.round_types & (1 << vh)):
                viol[0] = True
            elif self.vboss == B_MOUTH and self.mouth_hand >= 0 and vh != self.mouth_hand:
                viol[0] = True
        hand_out[0] = x.hand
        self.last_chips, self.last_mult = x.chips, x.mult
        return x.chips * x.mult
