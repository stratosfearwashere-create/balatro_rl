# Gap study: v51 on Gold vs what winning human builds need (10-30)

Data and tools: `tools/gap_study.py` (collect / report / levers), traces in `traces/gap/`. Human targets from `GOLD_STAKE_RESEARCH.md`
and `GOLD_STAKE_HEURISTICS.md`. Policy: `checkpoints/ppo_gold_v51.pt` (greedy). Red Deck, Gold Stake, engine with the rental/interest
order fixed (rentals and held Gold cards before interest, as `state_events.lua` does).

## 1. Honest vs foresight

| 1000 Gold games, same seeds | blinds | wins |
|---|---|---|
| rounds by the solver that sees the draws (as in training) | 10.83 | 0 |
| honest rounds (5 hidden-future copies vote per move, `Env::hide_future`) | 7.64 | 0 |

A stronger honest player (9 or 15 copies, 1000-1500 nodes) does not move it (7.44-7.60 on 200 games): the loss is in the policy, whose
builds and money are sized for rounds it can see.

## 2. Builds at the start of each ante (honest games)

| ante | games | reach next | money med | >= $25 | rentals held | >= 1 / >= 2 xMult | main hand (share) | its level | deck size | enhanced cards | share vs boss med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 853 | 78% | 6 | 7% | 0.46 | 23% / 1% | Two Pair (51%) | 1.0 | 52 | 0.0 | 1.30 |
| 3 | 663 | 70% | 11 | 21% | 0.72 | 37% / 4% | Two Pair (52%) | 1.1 | 52 | 0.2 | 1.18 |
| 4 | 464 | 54% | 18 | 39% | 0.92 | 47% / 8% | Two Pair (53%) | 1.2 | 52 | 0.2 | 0.85 |
| 5 | 249 | 16% | 17 | 41% | 1.05 | 53% / 14% | Two Pair (47%) | 1.2 | 52 | 0.4 | 0.56 |
| 6 | 39 | 10% | 19 | 46% | 0.90 | 72% / 15% | Pair (38%) | 1.5 | 52 | 0.7 | 0.41 |

"Share vs boss" = 4 hands x the mean best single play (16 redrawn hands, exact scorer) / the ante's boss target. It falls below 1 at
ante 4: the wall is at antes 4-5, not 6.

Against the human targets:

| | human (research) | v51 |
|---|---|---|
| main hand level | 3-5 by antes 3-4, 6-9 by 5-6 (planet plan) | 1.2 at antes 4-5 |
| main-hand planet in the shop, affordable | buy | bought 1 of 66 at antes 4-5 |
| deck | thin (Hanged Man, Trading Card), copy the best card | always 52 cards, < 1 enhanced card |
| money | $25 held from ante 3 | median $11-19; 21-46% hold $25 |
| rentals | none before ante 5 | 30% of ante-1 joker buys are rentals; about 1 held from ante 3 |
| blind skips | only for Investment | about 1.1 skips per game at ante 1 |
| xMult | >= 1 by ante 6, >= 2 by ante 7 | 47% have one at ante 4; affordable xMult in the shop bought 40-50% of the time |
| hand | Pair (or planet Straight) | Two Pair half the time |

The few honest runs that reached ante 7 had, at ante 5, more money ($36 vs $24), more xMult (1.0 vs 0.7), fewer flat +mult jokers
(1.25 vs 1.85), higher best hand level (4.2 vs 1.9) and a boss share of 1.34 vs 0.55.

## 3. Levers: what each fix is worth (`levers`)

400 real positions at the first blind select of ante 4 and of ante 5 (from 3000 fresh v51 games); each edited one way, then played on
by v51 (foresight rounds) in the same 8 redrawn futures per position (3200 games per lever, paired). Gain = extra blinds beaten vs the
unedited position.

| lever | ante 4: gain +- se | wins / 3200 | reach ante 8 | ante 5: gain +- se | wins / 3200 | reach ante 8 |
|---|---|---|---|---|---|---|
| none | 0 (3.74 blinds after) | 2 | 0.8% | 0 (2.23 after) | 7 | 1.8% |
| main hand +3 levels | +0.57 +- 0.04 | 5 | 1.1% | +0.43 +- 0.03 | 11 | 2.9% |
| main hand +6 levels | +1.27 +- 0.04 | 13 | 1.9% | +1.00 +- 0.04 | 22 | 4.3% |
| +$25 | +0.41 +- 0.03 | 5 | 1.1% | +0.24 +- 0.02 | 12 | 2.5% |
| one x3 joker (Cavendish) in a flat joker's place | **+2.55 +- 0.05** | 31 | 5.5% | **+1.71 +- 0.05** | 43 | 9.1% |
| Hanging Chad in a flat joker's place | +0.09 +- 0.04 | 2 | 0.8% | -0.43 +- 0.03 | 5 | 1.4% |
| 10 lowest plain cards destroyed | +0.42 +- 0.04 | 5 | 1.5% | +0.29 +- 0.04 | 16 | 2.5% |
| 13 cards changed to the main suit | +0.16 +- 0.03 | 5 | 1.2% | +0.08 +- 0.03 | 14 | 2.5% |
| rental stickers removed | +0.16 +- 0.02 | 4 | 0.9% | +0.11 +- 0.02 | 9 | 1.8% |
| +3 levels, x3 joker, thin 10 | **+3.63 +- 0.06** | 55 | 11.9% | **+2.85 +- 0.05** | 108 | 17.1% |

Reading:

- **One multiplier is worth more than everything else.** A single static x3 at ante 4 is worth +2.5 blinds and multiplies the win
  rate by about 15. It matches the human rule "one x3 beats ante 8" and the deep-run profile above.
- **Hand levels are the second lever** and scale with the dose: +3 levels about half a blind, +6 about one blind and more than doubles
  reaching ante 8. v51 almost never buys its main hand's planet.
- **Money, thinning, suit focus and rentals** are each worth 0.1-0.4 blinds on their own; they matter as enablers (money buys the
  multiplier and the planets), not as score.
- **Hanging Chad alone does nothing** for these builds (no enhanced cards to retrigger); it is a piece of the glass/steel plan, not a
  standalone buy.
- **Together** (+3 levels, x3, thin 10) the ante-5 positions reach ante 8 17% of the time (from 1.8%) and win 3.4% (from 0.2%), with v51
  still making every later decision. The policy's own later play stays the limit: even boosted builds rarely win.

## 4. What this means for the plan-based shop player

1. Its first job is a **multiplier by ante 4**: value the xMult jokers it can reach (shop, Buffoon packs, rerolls with $25 banked)
   above flat jokers from ante 3, and keep money for them.
2. Second, **level one hand** with planets (Celestial packs, the shop's planets, Blue seals) as a standing plan; pick the hand from what
   the deck and jokers already favour (Pair by default; Straight when planets / Shortcut / Four Fingers come).
3. Economy rules as enablers: $25 from ante 3, no rentals before ante 5, no blind skips except Investment.
4. Train and evaluate with honest rounds (or close the honest gap): the policy loses 3.2 blinds when it cannot see the draws.
