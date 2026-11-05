# Balatro Mechanics and Numbers Behind Gold Stake Strategy

Version basis: unless noted, numbers are for the current PC build (1.0.1f and later, 1.0.1o per the wiki Jokers page). Main web sources: balatrowiki.org (wiki.gg) and patch-note coverage. Corroboration comes from this project's C++ engine, `C:/balatro_rl/balatro_rl/sim/cpp/engine/game.hpp` and `tables_gen.hpp`. Per the project memory, the engine's keyed RNG was checked against the real game. The engine numbers stakes from 0 (White = 0, Gold = 7), so "stake >= 7" in its code means Gold. Local engine citations are marked [engine].

## 1. What each stake adds up to Gold, with exact sticker rules and version history

### Takeaway
Gold Stake includes every lower stake's modifiers:
- no Small Blind reward
- Purple-level ("scaling 3") blind requirements
- −1 discard
- Eternal and Perishable stickers: one shared roll, 30% each
- Rental sticker: an independent 30% roll; the joker costs $1 to buy and charges $3 at the end of every round
- Perishable jokers become debuffed after 5 rounds

Before 1.0.1, Orange Stake raised pack costs and Gold Stake gave −1 hand size. Version 1.0.1 replaced these with Perishable and Rental.

### Cited Findings
- Stake ladder, cumulative:

  | # | Stake | Adds |
  |---|---|---|
  | 1 | White | No modifiers |
  | 2 | Red | Small Blind gives no reward money |
  | 3 | Green | Required score scales faster |
  | 4 | Black | Eternal jokers can appear in the shop (can't be sold or destroyed) |
  | 5 | Blue | −1 discard |
  | 6 | Purple | Required score scales faster again |
  | 7 | Orange | Perishable jokers (debuffed after 5 rounds) |
  | 8 | Gold | Rental jokers ($3 per round) |

  — [Balatro Wiki: Stakes](https://balatrowiki.org/w/Stakes)
- Scaling level by stake: White and Red use scaling 1, Green through Blue use scaling 2, Purple through Gold use scaling 3 (`scaling(): stake>=5 ? 3 : stake>=2 ? 2 : 1`, 0-indexed). Base discards are `3 + Red Deck − (Blue stake or higher)`. — [engine] `game.hpp` lines 573, 733
- Sticker roll, as implemented: one poll decides Eternal versus Perishable.
  - Poll > 0.7 gives Eternal (Black Stake and up), a 30% chance.
  - 0.4 < poll ≤ 0.7 gives Perishable (Orange and up), a 30% chance.
  - A separate poll > 0.7 gives Rental (Gold), a 30% chance.
  - Stickers are rolled for shop jokers and pack jokers (pack streams use the keys "packetper" and "packssjr").
  - Jokers marked as incompatible refuse Eternal or Perishable.
  - A Perishable joker starts with 5 rounds. A Rental joker's price is set to $1.

  — [engine] `game.hpp` lines 2266–2288, 2515–2523
- The wiki confirms each sticker's base chance is 30%. At Orange and Gold there is a 40% chance of neither Eternal nor Perishable, and at Gold a 70% chance of no Rental. A joker can't be both Eternal and Perishable, and every joker can take Rental. — [Balatro Wiki: Stickers](https://balatrowiki.org/w/Stickers)
- Patch notes for 1.0.1:
  - Perishable: "30% chance … disabling them after 5 rounds".
  - Rental: 30% chance, "cost $1 up front and $3 every round", stacks with Eternal and Perishable.
  - Orange Stake's increasing pack cost was scrapped.
  - Gold Stake's −1 hand size was scrapped.
  - Eternal now applies to jokers in Buffoon packs.
  - On Gold Stake, random seeds make the first Legendary a joker you haven't yet won with on Gold.

  — [Shacknews, 1.0.1c notes](https://shacknews.com/article/139415/balatro-update-101c-patch-notes); also [Press Start summary](https://press-start.com.au/news/2024/04/08/the-latest-balatro-update-has-some-massive-changes-that-should-make-winning-easier/)
- 1.0.1 also made some non-retroactive scaling jokers incompatible with Perishable, for example Ceremonial Dagger, Glass Joker and Hologram. — [Press Start](https://press-start.com.au/news/2024/04/08/the-latest-balatro-update-has-some-massive-changes-that-should-make-winning-easier/)
- The Rental fee of −$3 is charged in the end-of-round payout, in the same step that ticks Perishable down. A Perishable joker is debuffed when its counter reaches 0; it is not destroyed. — [engine] `game.hpp` lines 2170–2175
- Unlocks: winning on Red unlocks the Zodiac Deck, Green unlocks Painted, Black unlocks Anaglyph, Blue unlocks Plasma, and Orange unlocks Erratic. Gold unlocks nothing. — [Balatro Wiki: Stakes](https://balatrowiki.org/w/Stakes)

### Inferences
- Odds for each shop joker at Gold, ignoring incompatibility:

  | Outcome | Chance |
  |---|---|
  | Eternal | 30% |
  | Perishable | 30% |
  | Rental | 30% (independent of the above) |
  | No sticker at all | 0.4 × 0.7 = 28% |
  | Eternal + Rental | 9% |
  | Perishable + Rental | 9% |

- From Black through Purple, only Eternal can appear (30%). The 0.4–0.7 band of the poll gives nothing on those stakes.
- A Rental joker costs $1 to buy, so its sell value should be $1 (floor of cost/2, minimum $1). Keeping it for k rounds costs about $3k, plus the interest lost on that money.
- Rental jokers are cheap entry points early, but $3 a round is close to the $5 interest cap, so most players sell them before long.

### Gaps
- No single primary source gives the exact build (1.0.0h or another) in which Stakes first shipped. The wiki Blinds page says Stake scaling values were "added" in 1.0.0h and revised in 1.0.1f.
- The official Steam text of the full 1.0.1f patch notes was not retrieved. The 1.0.1c beta notes differ from the final 1.0.1f values (see §2).

## 2. Blind chip requirements: base tables, multipliers, bosses and endless

### Takeaway
Gold uses the Purple ("scaling 3") table: 300, 1,000, 3,200, 9,000, 25,000, 60,000, 110,000 and 200,000 for antes 1–8. Small Blind is ×1, Big Blind ×1.5 and Boss ×2. The Wall is ×4, The Needle ×1 and Violet Vessel ×6. The Ante 8 boss at Gold therefore needs 400,000 chips (1.2M for Violet Vessel).

### Cited Findings
- Base chips per ante, post-1.0.1f:

  | Ante | White (1) | Green (2) | Purple / Gold (3) |
  |---|---|---|---|
  | 0 and below | 100 | 100 | 100 |
  | 1 | 300 | 300 | 300 |
  | 2 | 800 | 900 | 1,000 |
  | 3 | 2,000 | 2,600 | 3,200 |
  | 4 | 5,000 | 8,000 | 9,000 |
  | 5 | 11,000 | 20,000 | 25,000 |
  | 6 | 20,000 | 36,000 | 60,000 |
  | 7 | 35,000 | 60,000 | 110,000 |
  | 8 | 50,000 | 100,000 | 200,000 |

  — [Balatro Wiki: Blinds and Antes](https://balatrowiki.org/w/Blinds_and_Antes). The engine has the same table: `BLIND_BASE` in [engine] `tables_gen.hpp` line 639.
- Endless base chips:

  | Ante | White | Green | Purple |
  |---|---|---|---|
  | 9 | 110k | 230k | 460k |
  | 10 | 560k | 1.1M | 2.2M |
  | 11 | 7.2e6 | 1.4e7 | 2.9e7 |
  | 12 | 3.0e8 | 6.0e8 | 1.2e9 |
  | 13 | 4.7e10 | — | 1.8e11 |
  | 16 | 8.6e20 | — | 3.4e21 |

  At Ante 39 the requirement overflows to NaN. — [Balatro Wiki: Blinds and Antes](https://balatrowiki.org/w/Blinds_and_Antes)
- Multipliers and rewards:

  | Blind | Chip multiplier | Reward |
  |---|---|---|
  | Small | ×1 | $3 (none from Red Stake up) |
  | Big | ×1.5 | $4 |
  | Boss | ×2 | $5 |
  | The Wall | ×4 | $5 |
  | The Needle | ×1 (one hand only) | $5 |
  | The Flint | ×2 (base chips and mult halved) | $5 |
  | Showdown bosses (Ante 8, 16, …) | ×2 | $8 |
  | Violet Vessel | ×6 | $8 |

  Chicot sets The Wall and Violet Vessel back to ×2. — [Balatro Wiki: Blinds and Antes](https://balatrowiki.org/w/Blinds_and_Antes)
- Minimum antes for bosses:
  - Any ante: Hook, Club, Psychic, Goad, Window, Manacle, Pillar, Head
  - Ante 2: House, Wall, Wheel, Arm, Fish, Water, Mouth, Needle, Flint, Mark
  - Ante 3: Eye, Tooth
  - Ante 4: Plant
  - Ante 5: Serpent
  - Ante 6: Ox
  - Showdown bosses: Amber Acorn, Verdant Leaf, Violet Vessel, Crimson Heart, Cerulean Bell (Ante 8, 16, …)

  The game picks only from the bosses that have appeared least, so no boss repeats until all have been seen. Director's Cut and Retcon reroll the boss for $10 each. — [Balatro Wiki: Boss Blinds](https://balatrowiki.org/w/Boss_Blinds)
- History of the scaling values:
  - 1.0.1c beta: White Ante 3 went 2800→2000 and Ante 4 6000→5000. Green Ante 2 went 1000→900, Ante 3 3200→2400, Ante 4 9000→7000. Purple Ante 2 went 1200→1000, Ante 3 3600→3000, Ante 4 10000→8000, Ante 5 25000→22000. — [Shacknews 1.0.1c](https://shacknews.com/article/139415/balatro-update-101c-patch-notes)
  - Final 1.0.1f: Green Ante 3 is 2,600 and Ante 4 is 8,000; Purple Ante 3 is 3,200, Ante 4 is 9,000 and Ante 5 is 25,000. The wiki attributes these to 1.0.1f, so 1.0.1f partly reverted the 1.0.1c cuts. — [Balatro Wiki: Blinds and Antes](https://balatrowiki.org/w/Blinds_and_Antes)

### Inferences
- Gold Stake chip targets (Small / Big / Boss, with the variant boss value in brackets):

  | Ante | Small | Big | Boss | Variant boss |
  |---|---|---|---|---|
  | 1 | 300 | 450 | 600 | — |
  | 2 | 1,000 | 1,500 | 2,000 | Wall 4,000 |
  | 3 | 3,200 | 4,800 | 6,400 | Wall 12,800 |
  | 4 | 9,000 | 13,500 | 18,000 | Wall 36,000 |
  | 5 | 25,000 | 37,500 | 50,000 | Wall 100,000 |
  | 6 | 60,000 | 90,000 | 120,000 | Wall 240,000 |
  | 7 | 110,000 | 165,000 | 220,000 | Wall 440,000 |
  | 8 | 200,000 | 300,000 | 400,000 | Violet Vessel 1,200,000 |

- Compared with White, Gold needs 1.8× more at Ante 4, about 2.3× at Ante 5, 3× at Ante 6, about 3.1× at Ante 7 and 4× at Ante 8. The steepest jumps against White come at Antes 5–6.
- Endless formula. The wiki's rendering is garbled; this form reproduces its tables (my arithmetic check):

  base(a) = base8 · (1.6 + (0.75·(a−8))^(1+0.2·(a−8)))^(a−8)

  The result is floored to 2 significant figures. For White Ante 9: 50,000 × (1.6 + 0.75^1.2) = 50,000 × 2.308 ≈ 115k, which floors to 110k. For Ante 10: 50,000 × (1.6 + 1.5^1.4)² ≈ 566k, which floors to 560k. Both match the table.

### Gaps
- The Lua source of `get_blind_amount` was not fetched directly. The endless formula above is inferred from the wiki formula plus a numeric check.

## 3. Red Deck and base run parameters

### Takeaway
Red Deck gives +1 discard per round. At Gold, this cancels Blue Stake's −1, so a Gold Red Deck run plays with 4 hands and 3 discards per round, 8 hand size, 5 joker slots, 2 consumable slots and $4 starting money.

### Cited Findings
- The Red Deck gives "+1 discard every round" and offsets the −1 discard from Blue Stake and above. — [Balatro Wiki: Red Deck](https://balatrowiki.org/w/Red_Deck)
- Engine defaults:
  - Money is $4 (plus $10 with Yellow Deck).
  - Jokers 5, consumables 2, hand size 8, hands 4.
  - Discards are `3 + (Red Deck) − (Blue stake or higher)`.

  — [engine] `game.hpp` lines 565–573

### Inferences
- On Gold, Red Deck is the only deck that keeps the White-stake discard count of 3. This is the main reason Red Deck is a common Gold Stake choice.

### Gaps
- The wiki Red Deck page doesn't list the base hands, discards or money. Those numbers come from the engine only.

## 4. Shop mechanics: slots, card-type weights, rarity, editions, rerolls, packs, prices

### Takeaway
A shop has 2 card slots (+1 each for Overstock and Overstock Plus), 2 pack slots and 1 voucher slot. Each card slot has these weights:
- Joker 20, Tarot 4, Planet 4, giving 71.4% / 14.3% / 14.3%
- Playing card 4 with Magic Trick
- Spectral 2 with Ghost Deck

Jokers roll 70% Common, 25% Uncommon and 5% Rare. Rerolls start at $5 and rise $1 each, resetting every shop.

### Cited Findings
- Slots:
  - 2 random card slots, 2 booster packs and 1 voucher.
  - Overstock and Overstock Plus each add 1 card slot.
  - The first shop of a run always has a Normal Buffoon Pack.
  - Rerolls do not restock packs or the voucher.
  - The voucher restocks after each boss is defeated.

  — [Balatro Wiki: Shop](https://balatrowiki.org/w/Shop)
- Card-slot weights:
  - Joker 20, Tarot 4, Planet 4.
  - Playing card 4 with Magic Trick. With Illusion, 40% of them are enhanced and 20% have an edition.
  - Spectral 2 with Ghost Deck.
  - Tarot Merchant raises the Tarot weight to 9.6; Tarot Tycoon raises it to 32. Planet Merchant and Planet Tycoon do the same for Planets.

  — [Balatro Wiki: Shop](https://balatrowiki.org/w/Shop); [engine] `game.hpp` lines 2336–2337
- Joker rarity: 70% Common, 25% Uncommon, 5% Rare. Legendaries never appear in the shop. The rarity is rolled first, then a uniform pick within that rarity, excluding jokers already owned or locked. — [Balatro Wiki: Jokers](https://balatrowiki.org/w/Jokers); [engine] `game.hpp` line 2245 (`r>0.95→Rare, r>0.7→Uncommon`)
- Joker pool size: 150 jokers in total — 61 Common, 64 Uncommon, 20 Rare and 5 Legendary (as of 1.0.1o). 105 are unlocked from the start. — [Balatro Wiki: Jokers](https://balatrowiki.org/w/Jokers)
- Joker editions, by default:

  | Edition | Chance | Effect | Price |
  |---|---|---|---|
  | Negative | 0.3% | +1 slot | +$5 |
  | Polychrome | 0.3% | ×1.5 Mult | +$5 |
  | Holographic | 1.4% | +10 Mult | +$3 |
  | Foil | 2% | +50 Chips | +$2 |

  — [Balatro Wiki: Editions](https://balatrowiki.org/w/Editions)
- How the edition roll works:
  - The poll uses cumulative thresholds: 1−0.003 for Negative, then 1−0.006·r, 1−0.02·r and 1−0.04·r for Polychrome, Holographic and Foil.
  - r = 1 by default, 2 with Hone and 4 with Glow Up.
  - So Hone gives Polychrome 0.9%, Holographic 2.8% and Foil 4%. Glow Up gives 2.1%, 5.6% and 8%. Negative stays at 0.3% because vouchers don't change it.

  — [engine] `game.hpp` lines 2221–2235; the values match the wiki rows "0.9/2.8/4" and "2.1/5.6/8" on [Editions](https://balatrowiki.org/w/Editions)
- Rerolls:
  - $5 to start, +$1 per reroll, and the price resets each shop.
  - Reroll Surplus takes $2 off, and Reroll Glut another $2.
  - Chaos the Clown gives 1 free reroll per shop. The D6 Tag sets the starting reroll price to $0.

  — [Balatro Wiki: Shop](https://balatrowiki.org/w/Shop); [engine] `game.hpp` lines 2758, 2899
- Prices:
  - Base cost by type: Common jokers $1–6, Uncommon $4–8, Rare $7–10, Tarot and Planet $3, Spectral $4, Voucher $10, packs $4 / $6 / $8.
  - Edition surcharges are added to the base cost.
  - Clearance Sale takes 25% off and Liquidation 50%.
  - Sell value is floor(cost/2), minimum $1.

  — [Balatro Wiki: Shop](https://balatrowiki.org/w/Shop)
- Voucher unlock conditions (meta-progression, not in-run requirements): Liquidation needs 10 vouchers redeemed in one run, Reroll Glut needs 100 total rerolls, and Money Tree needs interest maxed for 10 consecutive rounds. — [Balatro Wiki: Vouchers](https://balatrowiki.org/w/Vouchers)
  - The Shop page phrases these as in-run requirements, which is misleading.
- Pack weights and chance per pack slot:

  | Pack | Normal | Jumbo | Mega |
  |---|---|---|---|
  | Arcana | weight 4 (17.84%) | 2 (8.92%) | 0.5 (2.23%) |
  | Celestial | 4 (17.84%) | 2 (8.92%) | 0.5 (2.23%) |
  | Standard | 4 (17.84%) | 2 (8.92%) | 0.5 (2.23%) |
  | Buffoon | 1.2 (5.35%) | 0.6 | 0.15 |
  | Spectral | 0.6 (2.68%) | 0.3 | 0.07 |

  By category: Arcana, Celestial and Standard 28.99% each, Buffoon 8.7%, Spectral 4.33%. — [Balatro Wiki: Booster Packs](https://balatrowiki.org/w/Booster_Packs)
- Pack sizes (cards shown, cards picked): Arcana, Celestial and Standard show 3 / 5 / 5 and pick 1 / 1 / 2. Buffoon and Spectral show 2 / 4 / 4 and pick 1 / 1 / 2. — [Balatro Wiki: Booster Packs](https://balatrowiki.org/w/Booster_Packs)
- Contents of Standard Pack cards:
  - 40% are enhanced (poll > 0.6), with a uniform pick among the 8 enhancements.
  - Editions are rolled with ×2 rate and no Negative: Polychrome 1.2%, Holographic 2.8%, Foil 4%, so about 8% total.
  - 20% have a seal, split evenly between Red, Blue, Gold and Purple.

  — [engine] `game.hpp` lines 2415–2428
  - Conflict: the wiki Editions page lists a "1.2%" default edition chance for pack playing cards. That matches the Polychrome share only. [Editions](https://balatrowiki.org/w/Editions)
- The Soul (in Arcana and Spectral packs) and Black Hole (in Celestial and Spectral packs) each have a 0.3% chance per card. The shop never sells them. — [Balatro Wiki: Spectral Cards](https://balatrowiki.org/w/Spectral_Cards); [engine] `game.hpp` lines 3417–3421
- With Omen Globe, each Arcana slot has a 20% chance to become a Spectral card. — [Balatro Wiki: Vouchers](https://balatrowiki.org/w/Vouchers); [engine] `game.hpp` line 2439
- Vouchers: a voucher can't be bought twice, an upgrade needs its base voucher, and Blank fills the slot once the pool is empty. — [Balatro Wiki: Vouchers](https://balatrowiki.org/w/Vouchers)

### Inferences
- Interaction with stickers: in the shop the 30% sticker rolls apply on top of the rarity roll. A specific joker at Gold therefore has about a 28% chance of arriving with no sticker, and a 30% chance of being Eternal, which is only a problem if you later need to sell it.
- Rental makes any joker cost $1 up front. This makes Rental Rares and Uncommons attractive for builds early in the run.

### Gaps
- Exact pack contents for Buffoon packs (I assume the same 70/25/5 rarity) aren't stated on the wiki pages fetched.
- The exact spectral pool weights (all equal except The Soul and Black Hole) weren't confirmed from a primary source.

## 5. Money: interest, caps and end-of-round payout

### Takeaway
Interest is $1 per $5 held at the end of a round:
- capped at $5 per round by default (reached at $25)
- $10 with Seed Money (at $50)
- $20 with Money Tree (at $100)

The end-of-round payout is the blind reward, plus $1 per unused hand, plus interest, minus $3 for each Rental joker. Small Blind pays $0 from Red Stake up.

### Cited Findings
- Interest:
  - $1 per $5 held, with a $5 cap.
  - Seed Money raises the cap to $10; Money Tree to $20.
  - To the Moon adds an extra $1 per $5, uncapped by that cap. With one To the Moon the maximum is $10, $20 or $40.
  - Interest is computed at round end. Gold cards held in hand and Reserved Parking count toward it; Golden Joker and Rocket pay out after and don't count.
  - Green Deck: no interest, but $2 per remaining hand and $1 per remaining discard.

  — [Balatro Wiki: Interest](https://balatrowiki.org/w/Interest)
- Engine payout formula:
  - Small Blind reward is $3 on White only, Big Blind $4, Boss $5.
  - Plus $1 × hands left.
  - Plus (1 + To the Moon count) × min(floor(money/5), cap).
  - Plus $3 per Gold card held, Blue Seal planets, and end-of-round jokers.
  - Minus $3 per Rental joker.

  — [engine] `game.hpp` lines 2141–2171, 759 (cap 5 / 10 / 20)
- Showdown bosses pay $8. — [Balatro Wiki: Blinds and Antes](https://balatrowiki.org/w/Blinds_and_Antes)
- In 1.0.1, the Investment Tag skip reward went from $15 to $25, and tag jokers became free. — [Windows Central 1.0.1f](https://www.windowscentral.com/gaming/balatro-update-patch-101f) (from search snippet)

### Inferences
- At Gold, an ante with no skips and no interest yields $0 + $4 + $5 = $9 in blind rewards, plus about $1 per spare hand. A White run gets $12. Interest of up to $5 per round (up to $15 per ante) is therefore worth more than the blind rewards themselves at the default cap.

### Gaps
- No web source fetched states the default "$1 per remaining hand". It comes from the engine only, though the wiki Green Deck description ($2 per hand) implies a smaller default.

## 6. Poker hands: base values, level-up gains, planets

### Takeaway
Each Planet adds fixed chips and mult per level. Straight, Four of a Kind and Straight Flush gain the most mult per level (+3/+3/+4). Pair and High Card gain only +1. Several planet gains were buffed in 1.0.1.

### Cited Findings
- Hand values:

  | Hand | Base | Per level | Planet |
  |---|---|---|---|
  | High Card | 5 × 1 | +10 / +1 | Pluto |
  | Pair | 10 × 2 | +15 / +1 | Mercury |
  | Two Pair | 20 × 2 | +20 / +1 | Uranus |
  | Three of a Kind | 30 × 3 | +20 / +2 | Venus |
  | Straight | 30 × 4 | +30 / +3 | Saturn |
  | Flush | 35 × 4 | +15 / +2 | Jupiter |
  | Full House | 40 × 4 | +25 / +2 | Earth |
  | Four of a Kind | 60 × 7 | +30 / +3 | Mars |
  | Straight Flush (incl. Royal) | 100 × 8 | +40 / +4 | Neptune |
  | Five of a Kind | 120 × 12 | +35 / +3 | Planet X |
  | Flush House | 140 × 14 | +40 / +4 | Ceres |
  | Flush Five | 160 × 16 | +50 / +3 | Eris |

  1.0.0k set the minimum hand level to 1 (The Arm). Flush Five became a separate hand in Jan 2024. — [Balatro Wiki: Poker Hands](https://balatrowiki.org/w/Poker_Hands)
- Changes in 1.0.1: Saturn's mult per level went +2→+3, Neptune +3→+4, Ceres +3→+4, and Eris chips +40→+50. — [Shacknews 1.0.1c](https://shacknews.com/article/139415/balatro-update-101c-patch-notes)
- Telescope makes every Celestial Pack contain the Planet for your most-played hand; ties go to the higher-tier hand. Observatory gives ×1.5 Mult for each copy of that hand's Planet held in the consumable area. Observatory applies after all jokers, including Polychrome. — [Balatro Wiki: Vouchers](https://balatrowiki.org/w/Vouchers); [Balatro Wiki: Scoring](https://balatrowiki.org/w/Scoring)
- Blue Seal (1.0.1) creates the Planet for the last hand played in the round. Black Hole raises every hand by 1 level. — [Shacknews](https://shacknews.com/article/139415/balatro-update-101c-patch-notes); [Spectral Cards](https://balatrowiki.org/w/Spectral_Cards)

### Inferences
- Level n of a hand gives base + (n−1) × gain. For example:
  - Flush at level 10: (35 + 135) × (4 + 18) = 170 × 22.
  - Pair at level 10: 145 × 11.
  - Straight at level 10: 300 × 31.
- Planets alone can't reach Ante 8 Gold scores (400k). The planet base still matters a great deal, because xMult multiplies it.

### Gaps
- I found no source data on "typical" hand levels in Gold wins. That question belongs to the community-strategy notes.

## 7. Tarot and Spectral cards for deck manipulation

### Takeaway
The main tools for thinning or fixing a deck:
- Hanged Man destroys up to 2 cards
- Death turns the left card into a copy of the right one
- Strength raises the rank of up to 2 cards
- Each suit tarot converts up to 3 cards (Star = Diamonds, Moon = Clubs, Sun = Hearts, World = Spades)
- Cryptid makes 2 copies of a card
- Familiar, Grim and Incantation each destroy 1 card and add 3 enhanced face cards, 2 enhanced Aces or 4 enhanced number cards
- Sigil and Ouija recolour or re-rank the whole hand; Ouija costs −1 hand size

### Cited Findings
- Tarots:
  - Magician (2 cards → Lucky), Empress (2 → Mult), Hierophant (2 → Bonus).
  - Lovers (1 → Wild), Chariot (1 → Steel), Justice (1 → Glass), Devil (1 → Gold), Tower (1 → Stone).
  - Strength: up to 2 cards gain +1 rank.
  - Hanged Man: destroys up to 2 cards.
  - Death: the left card becomes the right card.
  - Star, Moon, Sun, World: up to 3 cards change suit.
  - Hermit: doubles money, max +$20.
  - Temperance: pays the jokers' total sell value, max $50.
  - Wheel of Fortune: 1 in 4 chance to add Foil, Holographic or Polychrome to a random joker.
  - Judgement: creates a random joker. High Priestess: 2 Planets. Emperor: 2 Tarots. Fool: copies the last Tarot or Planet used.
  - Tarots cost $3 since 0.9.3g.

  — [Balatro Wiki: Tarot Cards](https://balatrowiki.org/w/Tarot_Cards)
- Magician enchanted 1 card before 1.0.1 and 2 after. When every tarot is blocked, the fallback tarot changed from Fool to Strength. — [Shacknews 1.0.1c](https://shacknews.com/article/139415/balatro-update-101c-patch-notes)
- Spectrals:
  - Familiar: destroy 1 random card in hand, add 3 random enhanced face cards.
  - Grim: destroy 1, add 2 enhanced Aces.
  - Incantation: destroy 1, add 4 enhanced number cards.
  - Talisman, Deja Vu, Trance, Medium: add a Gold, Red, Blue or Purple seal to 1 card.
  - Aura: add Foil, Holographic or Polychrome to 1 card.
  - Wraith: a random Rare joker, and money set to $0.
  - Sigil: all cards in hand become one random suit.
  - Ouija: all cards in hand become one random rank, −1 hand size.
  - Ectoplasm: a random joker becomes Negative, −1 hand size.
  - Immolate: destroy 5 random cards in hand, +$20.
  - Ankh: copy a random joker and destroy the rest (the copy loses Negative).
  - Hex: a random joker becomes Polychrome, the rest are destroyed.
  - Cryptid: 2 copies of 1 selected card.
  - The Soul: a Legendary joker. Black Hole: all hands +1 level.

  — [Balatro Wiki: Spectral Cards](https://balatrowiki.org/w/Spectral_Cards)
- Ectoplasm's hand-size penalty grows with each use: −1, then −2, and so on (`hand_size -= 1 + ecto_count`). — [engine] `game.hpp` lines 3283–3292

### Inferences
- Hanged Man (−2 cards per use) and Immolate (−5 cards, +$20) are the fastest deck thinners. Death and Cryptid concentrate the deck on a key card without shrinking it. For flush builds, suit tarots convert 3 cards for $3, which is the cheapest suit fixing.

### Gaps
- No web source fetched confirms when Ectoplasm's growing penalty was introduced (believed to be 1.0.1) or how Strength wraps ranks (Ace to 2 is believed but unverified here).

## 8. Scoring order

### Takeaway
Score = Chips × Mult, applied in this fixed order:
1. Boss effects
2. "On played" jokers
3. Each scored card, left to right: rank chips, then enhancement, seal and edition, then "on scored" jokers; repeated for retriggers, Red Seal first
4. Held-in-hand effects (Steel, Baron and others; Red Seal and Mime retrigger these)
5. Jokers left to right: joker Foil and Holographic editions, then independent jokers, then Baseball Card, then joker Polychrome
6. Observatory planets
7. Plasma Deck balancing

Joker order matters because +Mult must come before ×Mult.

### Cited Findings
- The activation sequence is as above. Scored-card modifiers go enhancement → seal → edition. Retriggers re-run from base chips through the scored-card jokers, Red Seal first and then retriggering jokers left to right. Independent jokers aren't retriggered. Polychrome on jokers applies at that joker's step, in the order the wiki describes. — [Balatro Wiki: Scoring](https://balatrowiki.org/w/Scoring)

### Inferences
- Order consequences: put +Mult jokers left of ×Mult jokers. Polychrome on a joker multiplies everything scored before it.
- Card chip values are face value for 2–10, 10 for J/Q/K and 11 for Ace. This is standard knowledge and wasn't stated on the Scoring page fetched.

### Gaps
- Retrigger sources (Hack, Sock and Buskin, Dusk, Seltzer, Hanging Chad) and their exact timing weren't detailed on the fetched page. All of them act in the scored-card retrigger step.

## 9. Probability math for finding specific jokers

### Takeaway
A given shop slot shows a specific Common joker about 0.82% of the time, a specific Uncommon about 0.28% and a specific Rare about 0.18%. Without rerolls, a player sees about 6 card slots per ante, so the chance of seeing one particular Rare by Ante 8 is only about 8%. Gold strategy is therefore built around flexible, common or uncommon engines and economy for rerolls, not around a specific Rare.

### Cited Findings
- The game first picks a rarity, then a joker uniformly within it. Each ante has its own joker RNG stream (a per-ante "queue"). A community back-of-envelope estimate is about 1/140 per roll for a specific joker under uniform odds. — [Steam discussion](https://steamcommunity.com/app/2379780/discussions/0/4308326918582404174); stream keys include the ante: [engine] `game.hpp` line 2264
- Inputs: the rarity split (70/25/5) and pool sizes (61/64/20) from [Balatro Wiki: Jokers](https://balatrowiki.org/w/Jokers); the joker slot weight of 20/28 from [Balatro Wiki: Shop](https://balatrowiki.org/w/Shop)

### Inferences
These are my calculations for a fully unlocked pool, with no Showman and nothing owned.

- Chance a single slot shows a specific joker:
  - Common: (20/28)·0.70/61 = 0.82%
  - Uncommon: (20/28)·0.25/64 = 0.28%
  - Rare: (20/28)·0.05/20 = 0.18%
- Over 48 slots (2 slots × 3 shops × 8 antes, no rerolls):
  - specific Common: 1−(1−0.0082)^48 ≈ 33%
  - specific Uncommon: ≈ 12.5%
  - specific Rare: ≈ 8%
- Each $5+ reroll adds 2 more slots. With Overstock (3 slots), base exposure rises by 50%.
- Buffoon packs, with 2–4 jokers each but only an 8.7% slot share, add a small amount. Judgement and Wraith (Rare) add more draws.
- A typical "any Uncommon in a slot" chance is (20/28)·0.25 ≈ 17.9%. "Any Rare" is about 3.6%.
- At Gold, multiply by about 0.28 for a chance of the joker arriving with no sticker at all, or by 0.7 for a chance of no Rental.

### Gaps
- No Reddit or primary "probability by ante" analysis was found beyond the Steam back-of-envelope estimate. The figures above are my own derivations.
