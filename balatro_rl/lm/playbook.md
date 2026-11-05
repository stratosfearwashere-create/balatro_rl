# Gold Stake playbook (Yellow Deck)

Sources: DrSpectred's Gold Stake videos (66-win streak), the community research in GOLD_STAKE_RESEARCH.md, and our own measurements on the
exact engine (the gap study). Rules marked [measured] come from our experiments; the rest are expert rules.

## The numbers

Yellow Deck at Gold: start with $14 (Yellow's +$10), 4 hands, 2 discards (Gold's -1 discard), hand size 8, 5 joker slots, 2 consumable
slots. Small Blind pays $0, Big $4, Boss $5, plus $1 per unused hand. Interest $1 per $5 held at round end, max $5 (at $25; $10 with Seed Money, $20 with Money Tree). Rentals charge $3 each round.

| Ante | Small | Big | Boss | Boss per hand (4 hands) |
|---|---|---|---|---|
| 1 | 300 | 450 | 600 | 150 |
| 2 | 1,000 | 1,500 | 2,000 | 500 |
| 3 | 3,200 | 4,800 | 6,400 | 1,600 |
| 4 | 9,000 | 13,500 | 18,000 | 4,500 |
| 5 | 25,000 | 37,500 | 50,000 | 12,500 |
| 6 | 60,000 | 90,000 | 120,000 | 30,000 |
| 7 | 110,000 | 165,000 | 220,000 | 55,000 |
| 8 | 200,000 | 300,000 | 400,000 (Violet Vessel 1,200,000) | 100,000 (Vessel 300,000) |

The Wall is x2 a normal boss, Violet Vessel x3. Ante 5 to 6 is the steepest step (x2.4).

## What wins, in order of measured value [measured]

Positions at ante 4 (measured on Red Deck; the order holds for any deck), each edited one way and played on, 3,200 games per edit:
1. **One xMult joker (x3) in place of a flat joker: +2.5 blinds, wins x15.** Nothing else comes close. The run is decided by getting a
   multiplier by ante 4-5.
2. **Main hand +6 levels: +1.3 blinds** (+3 levels: +0.6). Planets for your main hand are the second engine.
3. **+$25, thinning 10 cards, suit focus, no rentals: +0.1 to +0.4 each.** These matter as enablers (money buys the multiplier and the
   planets), not on their own.
4. All three together (x3 joker, +3 levels, thin 10) at ante 5: reach ante 8 17% of the time instead of 1.8%.

Our previous agent's failures: it reached ante 4-5 with hand level ~1.2, almost never bought its main hand's planet, never touched the
deck, bought rentals at ante 1, skipped blinds, held $11-19 instead of $25, and left affordable xMult in the shop half the time.

## The plan by ante

1. **Antes 1-2: survive.** Take whatever makes ordinary hands win now: cheap +Mult (Jolly, Half Joker, Banner, Mystic Summit, Abstract,
   Popcorn, Gros Michel, Blackboard) or any chips. One scoring joker by ante 2, about 2-3 by ante 3. One early planet on the hand you play
   can carry antes 1-2. Ante 2 (1,000 / 1,500 / 2,000) is the first real wall.
2. **Then one value generator** (Golden Joker, Rebate, Business Card, To the Moon, Trading Card, Hallucination, Cartomancer) if the next
   boss is already safe. The deciding question is always: can I beat the next boss without spending on scoring?
3. **Antes 2-4: reach $25 and hold it from ante 3.** Spend only the surplus above $25 unless the next blind would fail.
4. **Antes 3-5: build the engine.** Look for the multiplier (xMult joker, or Hanging Chad + Photograph / glass) and level one hand with
   planets. Take the first scaling joker you see and feed it (Castle, Wee, Square, Red Card, Ride the Bus, Green Joker, Constellation,
   Runner). Use surplus money for rerolls and packs to find xMult.
5. **Antes 5-6: convert.** Sell economy jokers and spent flat jokers for xMult. Buy a boss reroll voucher (Director's Cut) by ante 6.
6. **Antes 7-8: insurance.** Check whether the build clears Violet Vessel (1.2M in 4 hands). If not, buy extra xMult (Flower Pot,
   Ancient, Bloodstone, static xMult like Blackboard, Card Sharp, Cavendish). Slow starters (fresh Hologram, Constellation, Lucky Cat,
   Obelisk) are too late at ante 6-7.

Targets: at least 1 xMult joker by ante 5-6, at least 2 by ante 7. Main hand level 2-3 by ante 2, 3-5 by antes 3-4, 6-9 by antes 5-6,
about 12 by ante 7 for planet plans. Winning boards have 3-4 working jokers: about 1 chips source, 1-2 +Mult, 2-3 xMult (xMult on the
right).

## Which hand

- **Default: Pair** (or High Card with a big flat chips joker such as Stuntman). Always drawable, discard five to dig, lets you hold
  blue-seal, gold and steel cards.
- **Straights need an enabler with only 2 discards** (one fewer discard is what pushes Gold players onto pairs; straights without help
  land 50% of the time or less): commit only with Shortcut / Four Fingers or heavy planet supply (Saturn, Orbital tag, Telescope, blue
  seals). With that supply a level-12 Straight scores about 6x a level-12 Pair.
- **Retrigger plan:** Hanging Chad (always take it) + Photograph, or glass cards; Seltzer is a Violet Vessel answer.
- Commit on a signal, not on a fixed ante: Stuntman -> High Card/Pair; Spare Trousers -> Two Pair; early Ouija -> Four of a Kind; blue seal
  or Telescope -> commit harder to the levelled hand. Two Pair scores like two separate pairs; prefer Pair.
- By ante 7 plan one very good hand, not two. Flushes stop scaling late; flush planets rarely win Gold on their own.

## Economy

- **Yellow Deck:** bank the extra $10: you start at $14 and can reach $25 by round 2-3. The head start makes early pack gambles
  affordable (a miss still leaves you solvent) and makes one rental and Overstock easier to carry than on other decks.

- $25 banked is $5 a round, more than all blind rewards combined. Every $5 below it costs $1 every round from now on.
- Do not open packs or reroll below about $20-25 in the early antes. Never drop an interest tier for nothing ($14 and $11 earn the same).
- With one rental hold $28, with two $31.
- Sell a spare consumable or a finished joker when the $1-2 crosses an interest threshold, before the round ends.
- Rerolls: $5-7 fine; $8-9 only with a reason; $10+ only when rich ($60+). Spread rerolls across shops. Holding Death or Fool: do not reroll.
  Carrying two rentals: stop rerolling. No rerolls for 2-3 antes means the run is in trouble.
- Survival overrides interest: if the next blind would fail, spend what fixes it. "Money is useless if you die."
- Hermit: use it when you hold about $20 (it doubles up to $20). Temperance: take the cash.

## Stickers

- **Rental:** none before ante 5 (30% of our old agent's ante-1 buys were rentals: a named failure). At most one, only if it wins rounds
  now and income covers $3. Sell it the moment it is no longer the only way to win.
- **Eternal:** fine on generic early power (Abstract, Banner, Jolly) and on one value generator; bad on narrow or chips jokers. Never
  Eternal + Rental.
- **Perishable:** a bridge only; free from ante 7's small blind (it lasts to the ante-8 boss).

## Jokers

- Generic beats narrow: jokers that work on any hand survive hand-restricting bosses.
- Chips are the scarce half early; +Mult is everywhere. Take even weak early chips (Sly, Crafty, Banner, Foil, Scary Face, Odd Todd).
- One scaler is enough; feed it from the moment you buy it. Supernova scales retroactively, so it can come late.
- Joker order matters: chips and +Mult left, xMult right. Photograph before +Mult jokers; Hologram left of Brainstorm; Campfire rightmost;
  Blueprint copies the joker to its right, Brainstorm the leftmost.
- Blueprint/Brainstorm are only as good as what they copy.
- Retrigger jokers (Hack, Sock and Buskin, Dusk) rarely have targets on Gold; Hanging Chad is the exception.
- Do not get attached to the early joker: sell it as soon as levels or a scaler overtake it.
- Avoid anti-synergies: Ride the Bus with face cards, Green Joker with heavy discarding, Obelisk with a levelled main hand, Banner /
  Mystic Summit with discard-hungry jokers.

## Packs and consumables

- Pack priority when affordable: Buffoon early (for scoring jokers), then by plan: Pair takes Standard, Straight takes Celestial, flush
  takes Arcana. Open packs before rerolling and before buying jokers (you see replacements first). Small before jumbo; Mega (pick 2) beats
  two normal packs.
- Standard picks: blue seal first, then purple or gold seal, then gold card. Stop around ante 6 or once you have your seals.
- Celestial: open when you need the level or hold Telescope; always level your main hand. Do not buy random planets for a second hand.
- **Planets for the main hand in the shop: buy them** (our old agent bought 1 of 66 it could afford).
- Hanged Man: "usually yes"; remove low and off-plan cards, keep the last copy of needed ranks. Aim for a thin deck (about 30-45 cards).
- Death: copy the best card (blue seal, then steel, gold, glass). Strength: concentrate a rank. Justice (glass): on surplus or seal cards.
- Fool: save for Death, Hanged Man, Hermit, Temperance; not for a planet.
- Blue seal: hold it, never play it; a level per round. Black Hole: always.
- Hex: on the first non-sticker scoring joker. Wheel of Fortune: weak (75% nothing).
- Keep a consumable slot for the key tarot or planet.

## Vouchers

- Buy the voucher you want first; they compound. Almost always: Overstock, Clearance Sale, Grabber, Wasteful, Paint Brush, boss reroll
  (Director's Cut by ante 6). Seed Money only with momentum and a scaling source already found. Tarot Merchant usually yes.
- Telescope when committed to one hand; Observatory with Telescope. Skip all vouchers while you have no scoring joker yet.
- Hieroglyph (-1 ante, -1 hand): good for one-shot-hand builds or when about to die; bad for four-pair rounds.

## Blinds and skips

- Default: do not skip. A skip costs a shop and a round of scaling.
- The one skip to take: **Investment tag** at antes 1-2 (play the small blind, see shop 1, then skip the big blind) when the next blind is
  safe. Take one, never two. Never double-skip into a boss.
- Economy tag only when the skip is safe; never Coupon, Uncommon, Rare, Negative or Polychrome tags.
- Read the boss as soon as the ante starts. Reroll bosses that counter your hand (Needle, Eye, Mouth, Psychic, Plant for face builds,
  Flint without xMult). Keep the boss reroll for Violet Vessel at ante 8.
- Bosses that kill: Needle (hardest), Eye, Mouth, Psychic, Water, Arm, Wall, Violet Vessel. The answer is generic chips and mult or
  one big hand. Crimson Heart disables a random joker: do not depend on one joker. Arm: hold planets until after it.

## Mistakes to avoid

1. Greed: holding a card too long, digging when a miss is not affordable.
2. Skipping a blind for money and dying to the next one.
3. Forgetting to sell spent jokers and consumables before the round ends.
4. Using a planet or Fool in the wrong order.
5. Selling a scaler too early, or keeping an economy joker too long with no scoring.
6. Over-rerolling early.
7. Buying a second scaler you do not need.
8. Committing to a mediocre build instead of pivoting when the shops offer something better.

## Every decision: a quick check

1. Survival: does the build clear the next blind (and this ante's boss)? If not, fix that first.
2. Multiplier: is there an affordable xMult joker or a way toward one (Buffoon pack, reroll with surplus)? From ante 3 this is the top buy.
3. Levels: is there a planet for the main hand? Buy it.
4. Money: does this move cross an interest tier down? Is it worth $1 a round for the rest of the run?
5. Slots: is a weak flat joker blocking a better piece? Replace it.
6. Plan: does the move fit the plan note, or is the shop giving a clearly better direction?
