# Gold Stake heuristics from Balatro University (DrSpectred)

Decision rules extracted from 300 Gold Stake videos on
https://www.youtube.com/@drspectred (about 315 hours of commentary).

- **001-177**: the "Gold Stake Lessons" series, newest first. Sections 1-15 are built from
  these.
- **178-302**: his other Gold Stake videos (win streaks, deck runs, challenge runs, long
  streams, a few modded runs), newest first. Section 16 records what they add or change.
  186 and 188 have no captions and are not covered.

Per-item verdicts (138 jokers, vouchers, tags, consumables, packs, stickers) are in
`GOLD_STAKE_ITEM_VERDICTS.md`; that table covers 001-177 only.

## How this was made, and how far to trust it

- Source is YouTube's auto-generated captions, read by 36 subagents (about 10 videos each)
  that wrote about 5,400 per-video rules; this file is the merge. Numbers in brackets are video
  indices; the "Video index" section at the end maps each to its title and YouTube id.
- Captions have no punctuation and garble card names. Names were corrected where obvious
  (Grimbo = Green Joker, bus = Ride the Bus, pants = Spare Trousers, Todd = Odd Todd,
  Chad = Hanging Chad, "back in time" = Hieroglyph). A few stayed unidentified
  ("Bartolomeo", "Wayos") and are left out.
- Nothing here was checked against the video picture, only the commentary. Rules reflect
  what he says he does and why. Quoted thresholds are his words, not measured values.
- A rule with many citations recurred across independent readers and is solid. A rule with
  one citation is a single remark and should be treated as a hint.

## 1. The plan in one page

1. **Survive ante 1, then ante 2, then think long term.** The first three rounds are the
   most dangerous part of a Gold Stake run. Take whatever makes garbage hands win now,
   even if it is mediocre or stickered. [059, 060, 146, 044, 036]
2. **Then get one value generator** (money, seals, card removal), so every later shop has
   options. Cap eternal value generators at one, two at most. [141-148, 113, 127]
3. **Default hand is pairs** (or high card with a big flat chips joker). Spammable, does
   not care what is in the deck, lets you discard five at a time to dig, and lets you hold
   blue seals, gold and steel cards. Flushes are for the first round or with a specific
   enabler; straights need Shortcut, Four Fingers, The Order or heavy thinning.
   [132, 162, 126, 127, 141, 170, 110]
4. **Scoring needs one scaling source plus small flat support.** Scaling +mult (Ride the
   Bus, Green Joker, Red Card, Supernova), or scaling chips (Square, Runner, Castle, Bull,
   Wee, blue-seal planet levels), plus any flat partner. Chips are the scarce half; +mult
   is everywhere. [152, 153, 085, 104, 133, 080, 118]
5. **Ante 6 is the wall** (he quotes 60k / 90k / 120k). Flat +mult alone fails there. By
   then you need scaling mult, scaling chips and higher hand levels, or an xMult.
   [065, 017, 016]
6. **Throttle.** Score just enough each round and convert the surplus (hands, discards,
   cash) into value. Leftover hands are money. [121, 123, 128, 081, 036]
7. **Thin the deck and copy the best card.** Hanged Man, Trading Card, Death, Strength.
   Under about 30 cards you see almost the whole deck every round and every seal fires.
   [025, 171, 161, 169, 129]
8. **Do not skip blinds** unless the tag clearly beats the shop you give up and you
   survive the next blind anyway. [039, 074, 079, 128, 092]

## 2. Early game (antes 1-2)

- IF shop 1 has no points source THEN buy immediate scoring over scaling or economy:
  Banner, Jolly, Mystic Summit, Half Joker, Abstract, Popcorn, Gros Michel, Blackboard,
  Ramen, any early Foil or Holographic. [002, 072, 073, 080, 146-149, 051-056]
- IF scoring for the next boss is already safe THEN prefer the value generator instead
  (Golden Joker, Rebate, Business Card, To the Moon, Certificate, Hanging Chad, Trading
  Card). [151-159, 021-030, 164, 171] This is his one real flip-flop across videos; the
  deciding variable is always "can I beat the next boss without it".
- A scoring joker that makes you safe is also economy: you win in fewer hands, stop
  rerolling and stop buying bad jokers. [073, 092]
- First joker wanted is +mult; then chips or a hand level. One early planet level on the
  hand you play (Mercury for pairs, one Saturn for straights) can carry antes 1-2. Level-2
  pair is roughly 120. [025, 076, 103, 146]
- Early chips are rare. Take even weak chips (Sly/Crafty-type, Foil +50, Banner, bonus
  cards, Scary Face, Odd Todd) because flat chips plus any mult lets pairs replace
  five-card hands. [007, 113, 153, 170, 147]
- IF a mediocre known joker vs a Buffoon pack gamble THEN open the Buffoon pack first for
  information, but a known-good scoring joker beats a roughly 30% miss. [041, 043, 161]
- IF deck gives a head start (Checkered, Plasma, Painted, Black with spare power) THEN skip
  the ante-1 Buffoon pack and protect interest. Weak decks (Abandoned, Anaglyph) need
  jokers early. [030, 042, 083, 142, 144]
- Round 1 opening: go for the flush, not the straight; two discards usually find it. On
  the big blind, play two pair or the high pair first, then the five-card hand. If you hit
  a 300-point five-card hand, about 150 remain, which three pairs cover. With only three
  of a suit, pair out instead of chasing. [091-095, 142, 014, 057]
- Do not get attached to the early joker. Sell it the moment levels or a scaler overtake
  it. [031, 037, 006]
- Self-check he uses: "if I take it, will I regret it; if I don't, will I regret not
  taking it"; "play every run as if it will be difficult". [015]
- "Not enough mult usually means we're too picky or not generating money." [137]

## 3. Economy

### Interest
- $25 banked is the target ($5 per round). Do not open booster packs below roughly
  $20-25 in the early antes; saving for interest is the safest value generator. [015, 021,
  030, 071, 074, 147]
- Spending $3 costs $3 plus the compounding interest it delays. Refuse $3 buys that do not
  directly help the current plan, and never drop a tier for nothing ($14 and $11 earn the
  same, so do not go from 14 to 11 either). [036, 097, 129, 122]
- Exceptions he pays interest for: Hanged Man, Hermit, Temperance-grade tarots at about $3;
  a Fool; a Telescope or celestial pack when the early joker is perishable; hand-size or
  discard vouchers in flush/straight builds. [165, 083, 164]
- Sell a spare planet, tarot or finished joker when the $1-2 crosses an interest
  threshold, and do it before round end. He forgets this repeatedly and counts it as a
  mistake. [002, 005, 077, 112, 152, 161]
- "Money not in the bank is money gone forever"; not buying is "like a negative Golden
  Joker". [044, 030]
- Rule of thumb for packs: do not spend more than you earn per round (he estimates $8-10
  per round with a Golden Joker). [173]

### Rerolls
- Free, then $5-7 is fine; $8-9 is "double price"; $10-11 is too expensive. When rich
  ($60+ or large income) $12-15 is tolerated to find tarots or xMult. [014, 012, 053, 054,
  112, 145, 175]
- Spread rerolls across shops instead of rolling deep in one. [014, 082, 084]
- IF holding Death (or Fool) THEN do not reroll, so another can still appear. IF holding no
  tarots THEN reroll. [030, 145, 026]
- IF carrying two or three rentals THEN stop rerolling. [016, 095, 152]
- Going 2-3 antes without rerolling is a sign the run is in trouble; no money means no
  rerolls means you never find the finisher. [057, 093, 135]
- Bull and other money-hoarders are a trap if they stop you rerolling. [091, 116, 131]

### Hands as money
- Each unused hand is $1 ($2 on Green). One-shotting round 1 is +$1, which can be the
  difference between one purchase and two; on Black, $6 instead of $5 unlocks uncommons.
  [007, 092, 172]
- With per-hand scalers (Bus, Green, Square, Castle) on a deck with no income bonus, do
  not play every hand until interest is maxed. Bank first, then spend hands on scaling.
  [033, 071, 066]
- Exchange rates he quotes: Bus/Green/Supernova about $1 per mult through hands;
  Bootstraps 2 mult per $5 (bad); Bull 2 chips per $1; Red Card +3 to +6 per round. [080]

### Money sources
- Hermit: hold it until the bank is near $20, then double. Hermit early to fill joker
  slots; Hanged Man once slots are full. [159, 171, 114]
- Temperance: take the cash promptly; $40-50 mid-late with Gift Card. Take the money over
  a glass/steel card early; late, steel may be worth more. [009, 145, 167]
- Egg pays off after two Temperances and is unstable until the first one. [065, 075, 169]
- Seed Money: effective cost about $35 and about 7 rounds to repay. Only with momentum and
  a scaling source already found. He says he had been overvaluing it. [048, 049, 070, 121]
- Gold cards: +$3 each held; one cancels a rental fee. Do not win too fast if gold cards
  are still in the deck. [072, 076]

## 4. Stickers

- **Rental**: judge by "does it let me win in fewer hands, and can income cover $3". Real
  cost is more than $3 because it delays compounding interest. [091-098, 126, 157]
  - Cheaper on Green (no interest to lose), Yellow (reaches cap fast), Blue (extra hand
    pays it). Expensive on Red, Abandoned, Ghost, Painted. [033, 094, 111, 126, 157]
  - Take generic +mult/chips rentals you plan to swap out (Half Joker, Mystic Summit,
    Raised Fist, Popcorn, Runner). Avoid rentals that commit you to many hands (Green
    Joker, Supernova off Blue deck) or that score nothing yet. [094, 096, 100, 111-119]
  - Sell the rental the moment it stops being the only way to win; you save the fee and
    gain an interest dollar. [012, 055, 071, 092, 097]
  - One rental is workable, two usually breaks the economy, three means no rerolls.
    [012-020, 071-078, 101-105]
  - While carrying one: skip packs and rerolls, play the small blind for hand money, buy
    only the key voucher. [134, 135, 137, 177]
- **Eternal**: fine on generic early power (Abstract, Banner, Foil chips, Jolly) and on
  one value generator. Bad on narrow jokers and on chips jokers, which become dead weight
  once levels supply chips. Take a third eternal only if the first two are replaceable.
  [091, 097, 100, 122, 130, 131-139, 147]
- **Eternal + rental**: never. [012, 082, 141]
- **Perishable**: acceptable as a bridge with a plan to replace it; pick the one that
  lasts longer. Rarely worth it on slow money jokers or scalers. [161, 097, 141, 132]
- A big bank lets perishables expire before you replace them. [014]

## 5. Shop: what a build needs

- Slots by role: one chips source, one +mult, one xMult or flexible slot, plus value
  generators. Gold runs are won with 3-4 working jokers. [011, 113, 116, 141-148]
- Template he states: early stabiliser, a value generator, then scaling chips plus an
  xMult; "just times two is enough". [061]
- You need only one scaler; do not stack xMult for its own sake. "Most of the time you
  don't need xMult." [126, 151, 152, 167, 096]
- Generic beats narrow. Jokers that apply to any hand survive hand-restricting bosses and
  let you spam cheap hands. [091, 097, 143, 148, 177]
- Scaling chips beat scaling mult over a run because mult is plentiful. Two chips jokers
  are redundant outside Plasma. [104, 144, 118]
- Per-hand scalers (Runner, Supernova, Bus, Square) are "mini-games": scale them, then
  switch to the spammable hand. Runner does not mean you play straights. [073, 133, 177]
- Do not buy slow scalers late (Hologram, Lucky Cat, Obelisk at ante 6-7). [046, 048, 174]
- Blueprint/Brainstorm are only as good as what they copy; copying Abstract is another
  +15. [070]
- Retrigger jokers (Hack, Sock and Buskin, Dusk) rarely have targets on Gold. Hanging Chad
  is the exception and is always taken. [003, 063, 044, 171]
- "Eat your jokers" pieces (Madness, Ceremonial Dagger) change the shop: keep reroll money
  for fodder, buy eternals for safety, never feed a rental. [024, 098, 136, 171]
- Order of shop actions: buy the voucher you want first; reroll before opening Arcana;
  open packs before buying jokers so you see replacements; open Standard before Arcana for
  information; Arcana and Spectral before Celestial and before filling joker slots.
  [171, 055, 094, 098, 174]

Benchmarks he quotes for scalers: Green Joker 56 "not great", 65 "good enough", 70
"great"; Red Card usually ends 50-60; Bus/Green/Red about +60-70 by the end; Hologram
typically x3, x4-5 good; Castle 200 is enough on most decks; Runner 360 late is about x2 at
pair level 25; +25 flat mult early is "good enough". [143, 112, 087, 172, 144, 149, 147]

## 6. Booster packs

- Priority when affordable: Buffoon (early, for scoring), Arcana, Standard, Celestial.
  [021-030]
- Arcana: not at about $15; yes from about $19-22 with a floor near $15 and ideally $25
  after. Holding a Fool or a Golden Joker relaxes this. [030, 147, 058, 163]
- Standard: always look once money allows. Pick order blue seal, then purple or gold seal,
  then gold card; skip plain glass/foil/rank cards unless they fit. Stop around ante 6 or
  once you have your seals. A mega Standard shows 10 cards and yields about one seal.
  [121-128, 131, 145, 175]
- Celestial: open when you need the level to survive or own Constellation/Telescope; skip
  once blue seals supply levels. Use held planets before opening unless Observatory.
  [007, 041, 111-119, 173]
- Spectral: only if the boss is already safe; leave a joker slot for Hex/Wraith. [004, 174]
- Red Card: "pretend you don't have it". Do not skip packs you would open and do not open
  packs you would not. Always open for gold card, seals, Hermit, Temperance. [076, 104, 123]
- Small pack before large; planet odds are about 1 in 7 shop cards. [038, 111]

## 7. Consumables

- **Fool**: about 1 in 4 Arcana packs has one. Hold planets and tarots until after packs
  so Fool can copy the best one. Save Fool for Death, Hanged Man, Hermit, Temperance;
  spending it on a planet is a recurring regret. Track the last-used card before using
  anything. [171, 133, 151, 163, 061, 081]
- **Death**: copies rank, suit, enhancement and seal. Hold it for the best target (blue
  seal, then steel/gold/glass). Use it before discarding a purple seal. But do not hold it
  so long that it blocks finding more. [082, 062, 026, 121]
- **Hanged Man**: "usually yes". Remove low cards and off-plan ranks; keep the last copy
  of needed ranks. Think in cards not drawn. Removing two cards beats $20 late. [025, 102,
  125, 129, 171]
- **Death vs Hanged Man**: with many value cards and much garbage, remove garbage; copying
  only helps if you do not already see the card each round. [025]
- **Strength**: concentrate a rank (queens and jacks into kings). Keep one for emergencies
  in straight builds. [158, 090]
- **Glass (Justice)**: put it on surplus ranks or seal cards you can discard instead of
  play. Two or three in one hand is enough; one glass is about +70 mult late. Do not burn
  all Justice early. [031, 033, 030, 161]
- **Hex**: on the first non-sticker scoring joker; never on perishable/rental. On Ghost it
  makes rentals affordable (polychrome is about 50% more points). [102, 106, 119]
- **Planets**: level the hand you spam. Do not buy random planets for a second hand. Use
  random free levels instead of selling them for $1. Holding a planet also keeps it out of
  packs. [122, 049, 165]
- **Blue seal** is the strongest engine he names: a level per round, about 6x normal
  planet flow, worth about a Square Joker, makes chips jokers unnecessary. Hold it, never
  play it; one in hand is enough with Mime. A blue seal is the signal to commit to pairs
  or four of a kind and it is the Death target. He estimates it matters in about 30% of
  runs. You do not need it: Square, Castle, Bull, Spaceman substitute. [061-070, 095, 124,
  025, 176]
- **Purple seal**: tarot generator, better for straights and decks with discards; weaker
  on high stakes. **Gold seal** for pairs. [004, 035, 164]
- Saturn/level vs cash from a pack: take the level early ("momentum"), the money from
  ante 5. [031, 051]

## 8. Vouchers

- Buy the voucher you want before anything else in the shop; they compound. Carry cash
  into the boss so you can buy on arrival. [171, 015]
- Almost always: Overstock, Clearance Sale, Seed Money family when affordable, boss
  reroll. Hand size and extra discard over interest vouchers once the build holds cards.
  [152-158, 111, 137, 041, 046]
- Grabber (extra hand): first pick; works as an extra discard and +$1. [130, 134]
- Tarot Merchant/Tycoon: "usually yes" if it does not break interest. [102, 018]
- Crystal Ball over Telescope when you have blue seals. Telescope: decide the hand in
  round 1 and play it first. Observatory: yes with Telescope. [111, 141, 164]
- Magic Trick/Illusion: weak. Blank: mixed. [072, 084, 174]
- Skip Clearance Sale with Ceremonial Dagger (lower sell value). Skip all vouchers with no
  scoring joker yet; short-term cash matters more. [053, 055]
- **Hieroglyph** ("going back in time"): requirements roughly triple per ante early and
  double late, so -1 ante is like a free x2 at the cost of one hand (about 25% of a
  four-pair round). Take it for one-hand-per-round builds, when about to die, or when
  stable with a value generator to milk. Decline when spamming hands with Bus/Square.
  [009, 058, 062, 081, 088, 116, 118, 166]

## 9. Skip tags

- Default is do not skip. A skip costs a shop (worth $5-10 of pack refreshes), a round of
  scaling and, for per-round scalers, growth. Random-joker tags rarely fit. [039, 079,
  128, 121]
- Test: "would I pay this much to not see the next shop?" and "do I survive the next
  blind without it?" [128, 085, 093]
- **Investment** ($25 after boss): the one he usually takes at ante 1-2, like a slotless
  Golden Joker. Not on Black unless a joker is secured first; skip it if you already hold
  two value generators or Free Reroll. Take one, never both. [041, 044, 057, 095, 056,
  125, 128, 160, 162]
- Play the small blind, see shop 1, then skip the big blind. Never double-skip into a
  boss. [128, 131, 162]
- **Economy**: mostly declined; fine at ante 2-5 when the skip is safe and the shop is not
  holding your pieces. [061, 065, 111, 128, 151, 155]
- **Top-up**: good at ante 1-2 with no jokers (two clean commons, many of them scalers),
  only if you survive the next blind. [051, 061, 102, 175]
- **Orbital**: good when it shows your hand. **Standard** (mega): good early. **Coupon**:
  never ("free is not the same as useful"). **Uncommon/Rare/Negative/Polychrome**: no.
  **Handy/Charm**: only late with stacked Double tags. [039, 041, 152, 175]
- Costliest admitted error: skipping the ante-6 big blind for about $50 and losing the
  boss by 2k of 120k. "Money is useless if you die." [092, 093]

## 10. Choosing and changing the hand

- Pairs over high card: Mercury gives more than Pluto, about 50% more base. Pairs over two
  pair: easier, and two pair scores like two separate pairs. [127, 170, 120]
- High card when a big flat chips joker is present (Stuntman, Square, Wee). [096, 111, 087]
- Higher hands only pay with a planet source (Telescope, blue seals). Four of a kind is
  viable at about 9-11 copies of a rank in 52 plus blue seals and extra hands or discards;
  it lets you hold steel. [121, 124, 114, 052]
- Hands needing several plays per round (full house, straights) fail at Violet Vessel and
  high blinds; by ante 7 plan one very good hand, not two. But pure pair spam can also
  fail at Needle/Wall/Violet Vessel without held steel or an xMult. [075, 077, 087]
- Straights: choose a rank window, remove outside ranks, add inside ones. Shortcut is
  preferred over Four Fingers. [090, 141]
- Flushes: single suit only; be picky about off-suit adds. With pairs take every seal and
  enhancement. [013, 017]
- Pivot when a blue seal or planets support it; the cost is the levels left behind. Do not
  switch to Obelisk once a hand is highly levelled. [122, 160]
- Uniform decks finish runs (many of one rank plus Triboulet, Idol, Shoot the Moon, Hit
  the Road). [026, 027, 089, 099]
- Do not unlock five of a kind unless you plan it; extra planets pollute the pool. [152]

## 11. Playing the blind

- Start each round with a plan: how many hands of which type, how many discards. [113]
- Play throwaway hands first to dig or farm effects, the finisher last. Use discards first
  for Hit the Road, Rebate, Castle. [043, 071, 081]
- Play the weakest hand that wins and keep the strong cards. [001, 081, 111]
- Keep gold, steel and blue-seal cards in hand at round end. [033, 072]
- Dig for seals with discards only when a miss is affordable. Sacrifice the perfect hand
  to guarantee the blue seal. [091, 125]
- Do not throw away a made pair to chase better. [063]
- First discard: throw low cards and chase the flush; discarding low suited cards for a
  better flush is free if you need two hands anyway. [035, 066]
- With Green Joker mostly do not discard. Banner and Mystic Summit conflict with
  discard-hungry jokers; pick a discard policy. Burglar conflicts with digging. [108, 175,
  101]
- Extra hand size and extra hands act as extra discards. [151, 154]
- Even with a one-shot hand, keep the spare hand as a discard. [067]
- Joker order: Photograph triggers before +mult jokers; Hologram left of Brainstorm;
  Campfire rightmost. Copy Chad, not Photograph, with glass. [115, 172, 001, 106]

## 12. Bosses

- Killers are the hand-restricting ones: Needle (hardest), Eye, Mouth, Psychic, Water,
  Arm, Wall, Violet Vessel. Generic chips and mult, or a one-shot hand, is the answer.
  [040, 143, 148, 173]
- Eye: sequence the weakest hands first and keep the strong one for the close. [131, 081]
- Arm: hold planets until after. Worst at ante 4-5. [091-100, 143]
- Flint: needs xMult. Suit debuffs: suit tarots, rank hands, or glass. [123, 089, 040]
- Face-down cards: sort by suit/rank to identify, discard them first. [063, 081, 091]
- Plant punishes face builds; make non-face glass. Hook and random-discard bosses ruin
  precise digging; Hook and Manacle are the ante-1 killers on Black. [013, 016, 142]
- Tooth: budget cards played. Crimson Heart can disable the key joker; keep a glass/steel
  emergency set. [101, 117, 173]
- Save Justice, Death and suit tarots for the boss; keep a guaranteed fallback hand.
  [011-019]
- Use boss reroll when the boss is scary; he forgot twice and called it a mistake. [022]
- Campfire resets at the boss: feed before, save money after. [112, 114]

## 13. Decks

- **Yellow**: economic head start; easiest for rentals and Overstock. [086, 139, 177]
- **Blue**: extra hand is an extra discard and $1; pays rentals; pivot to pairs. [001, 096]
- **Red**: no income bonus, so bank before spending hands; extra discard suits straights.
  [033, 004]
- **Green**: no interest; $2 per hand and $1 per discard left. Win in few hands, do not
  hoard except for Hermit and vouchers, rentals are a flat $3. [053, 126, 137, 157]
- **Black**: harder; one less hand is one less discard and $1 less. Thinning matters. Do
  not take Investment without a joker. [098, 142, 162, 169]
- **Checkered**: skip the ante-1 Buffoon pack and go for the one-shot flush, but pairs are
  still the long plan. Castle is best here and on Abandoned. [042, 121, 030, 088]
- **Plasma**: chips are easier than mult; take chips jokers; play two pair or straights;
  skip early Buffoon packs. [144, 150, 118]
- **Painted**: hold seals and gold more easily; can survive early without jokers, so no
  unaffordable rentals. [069, 094]
- **Ghost**: Hex makes rentals safe; Hologram grows from spectral adds. [119, 103]
- **Anaglyph**: Top-up and Orbital early; stacked Double tags make late skips strong.
  [037, 175]
- **Abandoned**: Spare Trousers and Castle are better; be careful with spend. [088, 143]
- **Erratic**: you end up on pairs anyway. [067]

## 14. Mistakes he admits most often

1. Greed: holding a card or tarot too long, digging when a miss was not affordable,
   throwing away a made hand. [063, 173, 174, 159]
2. Using a planet or Fool in the wrong order. [042, 067, 109, 133, 163]
3. Forgetting to sell spent jokers and consumables before round end. [112, 152, 161]
4. Skipping a blind for money. [092, 093]
5. Selling a scaler or value joker too early, or keeping Hallucination/Egg too long with
   no scoring. [021, 028, 049, 166]
6. Over-rerolling early. [065]
7. Forgetting boss reroll and Red Card skips. [022, 006]
8. Buying a second scaler he did not need (Constellation, Supernova, Bull). [020, 116]
9. Locking a chips joker in as eternal. [100]

## 15. Where he contradicts himself

- **Scoring vs value generator in shop 1.** Videos 071-080 and 051-060 favour immediate
  scoring; 151-160 and 021-030 favour the value generator. Both are conditioned on
  whether the next boss is already beatable.
- **Rental The Tribe on Blue deck, ante 1.** Declined in 093, taken in 092 (the later
  video), where he calls the earlier pass a mistake.
- **Investment tag.** "For sure" in 069, 095, 173; declined in 056, 125; "usually lose" on
  Black in 160 but taken on Black in 057.
- **Seed Money.** "Buy immediately" in 112; "a trap" in 048; "I've been overvaluing it"
  in 070.
- **Hieroglyph.** "Usually worth it" in 058; "costs a third of scoring, not worth it" in
  116. Reconciled by build: good for one-shot hands, bad for four-pair rounds.
- **Blue seals.** "Busted" throughout, yet 105 and 156 note they added little once Wee,
  Castle or Square supplied chips.

## 16. What the other 123 Gold Stake videos add (178-302)

These are older and less curated than the Lessons series. Rules that only follow from a
challenge, a mod, a speedrun clock or gold-sticker hunting were tagged by the readers and
are left out here. 284-287 were played on the pre-release beta of the sticker stakes.

### Confirmed again, independently
- Pairs or high card as the default; straights are "50% or less" reliable with one fewer
  discard and are his most-admitted losing choice. [189-205, 226-231, 257-262, 277, 288-294]
- Blue seals plus levels as the main engine; Death on the seal card; Hanged Man for junk.
  [178-183, 190-204, 264-275]
- $25 interest line, no marginal packs below about $20-25, sell spent jokers to reach a
  breakpoint. [179, 199-205, 220-224, 240-254, 265-269, 281, 288-300]
- Rentals only when they win rounds or pay for themselves; one cheap rental at most; gold
  cards can pay the fee. [190-201, 222-232, 239-243, 257-263, 270-275, 279-287]
- Throttle and play the weakest winning hand. [181-185, 191-194, 240-245, 301-302]
- Hold unwanted planets to keep them out of shops and packs. [189-194, 249-254, 276-294]
- Investment tag is the only skip he routinely takes. [190-194, 215-221, 276-278]

### New rules
- **Scoring schedule**: one scoring joker by ante 2 and about 2.5-3 by ante 3. Ante 2
  (1200) is the first real wall; the ante-1 boss (600) is fine without a joker. Early
  +mult beats planet levelling. [276, 277, 295-297]
- **Commit to scalers early**: take the first scaling joker you see (Castle, Wee, Square,
  Dagger, Obelisk, Red Card) and commit; early acquisition matters more than which one.
  But do not pour money into slow ones. [264-269, 289-293, 301]
- **Photograph + Hanging Chad** is his "bailout" engine: one good card retriggered. Chad
  beats Dusk and Sock and Buskin because enhanced cards are few. [179-187, 234-237]
- **Hanged Man "early and often"**, taken over $20; stop thinning around 25 cards and cap
  Standard-pack additions at about 4. Deck sizes he aims for: 33-47 for flush/four of a
  kind plans. [276-282, 234-236, 270-275, 295-297]
- **Money first, deck second**: Hermit/Temperance/gold card early, Hanged Man once that is
  flowing. Take the first gold card even off-plan; value drops after about 3 held.
  [270-275, 247-256]
- **Tarot generators** (Hallucination, Cartomancer, Eight Ball, Superposition) are valued
  above Golden Joker as economy plus deck engine; Hallucination is his favourite common.
  Eight Ball conflicts with blue-seal planets for consumable slots. [206-214, 279-281]
- **Packs before rerolls**: open every pack you intend to open before rerolling; rerolls
  are lower value than packs. Buffoon first, small pack before jumbo. [280-282, 231]
- **Pack priority by build**: pairs take Standard, flushes take Arcana, straights take
  Celestial. [197-203, 282, 284-286]
- **Enhancement placement**: put gold, steel and seals on the rank/suit you hold and never
  discard; an off-plan held card is like -1 hand size. Stack buffs on one card, then
  Death-copy it. [283-286, 288-293, 234-237, 276-278]
- **Glass handling**: play one or two at a time, stockpile for the last hand; steel for
  early survival, glass for the endgame. [179-187, 247-254]
- **Joker order**: xMult to the right of +mult; Brainstorm copies the leftmost; Steel
  Joker leftmost. [288-292, 295-300]
- **Boss timing**: read the boss as soon as the small blind is beaten and delay buys and
  sells until you know it; reroll the boss early in the ante when it hits your hand type.
  [295-300, 288-292]
- **Violet Vessel** is the pairs/high-card killer and the reason not to skip to a boss at
  about 100k per hand. Keep two sources of points, or Luchador, or Hieroglyph in reserve.
  Random joker disable (Crimson Heart) is the other; do not depend on one joker.
  [190-193, 216-222, 246-255, 270-278, 209-212]
- **Safe over flashy**: take the two-hand win over a risky one-shot; play hands first and
  keep discards as insurance; count outs before discarding. [290-294, 296-300, 283-287]
- **Finisher pivot**: start on pairs or two pair, then move to a levelled finisher once
  the deck is thin (four of a kind from pairs; flush house on Abandoned; keep 3, discard
  5). Plain flushes stop scaling late. [215-221, 196-204, 276-281]
- **Avoid hand-size loss**: Ouija, Ectoplasm, Stuntman cost real consistency when the plan
  holds seals or gold. Hand size and discard sources are worth more than small scoring
  jokers. [193, 257-261, 276-278, 216-224, 296-300]
- **Late combo jokers** (Mime, Sock and Buskin, Idol, Baron) do not get slots until the
  economy and deck are built. [283, 284]
- **Marble Joker** is a temporary piece: hold about 3 antes with pairs/high card, then
  sell. [240-245]
- **Plasma**: aim for about 800 chips per hand; skip early shops; take Cloud 9. [190, 193]
- **Abandoned** is, by his account, his best Gold deck: flush house with almost no jokers,
  levelled by Telescope or blue seals. [279-281]
- **Ghost**: Hex the first joker, take Investment, avoid eternal+rental. [272, 274]
- **Yellow**: bank the extra money, reach $25 by round 2, and use it to gamble on early
  packs because a miss still leaves you solvent. [276-278]
- Stated attitude: "always play like you're losing"; Gold needs some high-rolls, so accept
  early risk and stop taking it around ante 6-7. [217, 218, 301, 302]
- He puts his own Gold Stake win rate at about 30% in the older streams. [279-282]

### Where the older videos disagree with the Lessons series
- **Coupon tag**: called the best ante-1 skip in 296 and 298 (the oldest videos); "never"
  from 041 onward. Treat "never" as his current view.
- **Skipping in general**: 288-294 allow foil/holo/negative/coupon skips early; 276-278
  and the Lessons series reject them because the joker arrives stickered or off-plan.
- **Ride the Bus**: rated weak in 279-282; ranked the best common +mult scaler in 191-195
  and 270-274 and taken almost always in the Lessons series.
- **Mime**: listed as weak in 279-282; an automatic take with blue seals later.
- **Seed Money**: "usually skipped on Gold" in 276-278; a priority voucher in 288-294 and
  267-269. Matches the split in section 15.
- **Finisher vs spam**: 215-224 and 242-244 prefer pivoting to four of a kind or flush
  house and call pairs the least efficient hand; the Lessons series leans to staying on
  pairs with blue seals. The older rule depends on having thinned the deck.
- **Wheel of Fortune targeting**: 299 and 300 contradict each other.

## Video index

Watch at `https://www.youtube.com/watch?v=<id>`.

| # | Title | Length | id |
|---|---|---|---|
| 001 | Which boss is the actual worst? (Gold Stake Lessons) | 59 min | jTlGi-hFZEs |
| 002 | Some of you owe Polar an apology (Gold Stake Lessons) | 49 min | YygA5Pi9b2s |
| 003 | How to make your own Luck (Gold Stake Lessons) | 50 min | BJhYQkTtiS8 |
| 004 | Purple Seal vs Gold Seal (Gold Stake Lessons) | 65 min | dnpW4mAt_-I |
| 005 | Ok, THIS was stressful (Gold Stake Lessons) | 56 min | IvdbSVW3hDI |
| 006 | This Seed defeated Balatro's winning-est players (Gold Stake Lessons) | 53 min | 5RFLCFhnXEw |
| 007 | One in the hand, two in the bush (Gold Stake Lessons) | 58 min | aMRm9dn9f30 |
| 008 | A Zodiac Deck palate cleanser (Gold Stake Lessons) | 49 min | _pYIxtL0E_w |
| 009 | The MOST unconventional Black Deck run (Gold Stake Lessons) | 56 min | h8f2_ArCYho |
| 010 | Surely we can afford a few Rentals (Gold Stake Lessons) | 35 min | M5Y8XyMRm_M |
| 011 | More than one path to victory, probably (Gold Stake Lessons) | 59 min | dxW85jAK3iU |
| 012 | I still stand by that choice (Gold Stake Lessons) | 42 min | TK-xVyEE0Sk |
| 013 | Always speculative Chad, only sometimes Photo (Gold Stake Lessons) | 64 min | 3CKz02U0WxU |
| 014 | C++ can be a triggering endeavor (Gold Stake Lessons) | 88 min | da8Rk2fhOpg |
| 015 | Canio be sleepin' (Gold Stake Lessons) | 91 min | 9RhyorM3Ors |
| 016 | Everything is bait (Gold Stake Lessons) | 48 min | PuAgZqDxTxQ |
| 017 | Pants big enough for my Telescope (Gold Stake Lessons) | 48 min | QAi7tTvPXhc |
| 018 | Trading Card makes it hard (Gold Stake Lessons) | 61 min | 9qiSgz2RDYE |
| 019 | I'm stuck in a time loop (Gold Stake Lessons) | 72 min | YR-dDLGVOBs |
| 020 | Is "pairslop" always the answer? (Gold Stake Lessons) | 39 min | 107KIYeyYy4 |
| 021 | We don't need to be comparing Obelisks (Gold Stake Lessons) | 49 min | HiGoHSArFVg |
| 022 | Think, Zaino, think (Gold Stake Lessons) | 45 min | zqINP6NEwkQ |
| 023 | Goated seed picker (Gold Stake Lessons) | 40 min | _52Db9GC4ng |
| 024 | THE Crash out King (Gold Stake Lessons) | 56 min | ydvlZPQ7GfY |
| 025 | It's sometimes an Idol seed (Gold Stake Lessons) | 72 min | 9zZvFu3tGoU |
| 026 | An egregious amount of pondering my orb (Gold Stake Lessons) | 76 min | JcwMwYOLins |
| 027 | This is why we prioritize Arcana Packs (Gold Stake Lessons) | 54 min | 0WTg0SvyPhI |
| 028 | I don't think I played this one correctly (Gold Stake Lessons) | 43 min | Nj_k0gWhRH4 |
| 029 | When should you start taking "value generators"? (Gold Stake Lessons) | 62 min | VHc7ESE160I |
| 030 | "Misprint tech" IYKYK (Gold Stake Lessons) | 72 min | OyAtJLpuJZI |
| 031 | Doc is a picky eater (Gold Stake Lessons) | 48 min | 0BPp5_FxuLI |
| 032 | Ok, THIS seed was BRUTAL (Gold Stake Lessons) | 47 min | 0Z6jR1uTy0E |
| 033 | Debt doesn't have to be forever (Gold Stake Lessons) | 62 min | PtRD2rpYyJ4 |
| 034 | This is why I love this Deck (Gold Stake Lessons) | 51 min | SfYQsQLdkA0 |
| 035 | I'm embarrassed to show this one (Gold Stake Lessons) | 59 min | 7Wh6j5Fdrvw |
| 036 | Well, that's just mean (Gold Stake Lessons) | 52 min | eAZ5vPJBTUo |
| 037 | Walkie Talkie counts as scaling (Gold Stake Lessons) | 59 min | ZQIrLBfpINc |
| 038 | The most CURSED Checkered Deck run (Gold Stake Lessons) | 69 min | BzMdXrJJGZI |
| 039 | The Anaglyph Deck dream (Gold Stake Lessons) | 49 min | S43RCz8q3s4 |
| 040 | Doomsday prepping, when playing Pairs (Gold Stake Lessons) | 48 min | gOdh-WJ3xoY |
| 041 | Coupon Tag is SO bad (Gold Stake Lessons) | 57 min | iNoj0zqaRZo |
| 042 | Can we agree this Joker is overpowered for its rarity? (Gold Stake Lessons) | 57 min | HKHcwzU_wDM |
| 043 | It's honest work (Gold Stake Lessons) | 56 min | KUN755Z5eqA |
| 044 | Just get Lucky, I guess (Gold Stake Lessons) | 67 min | TItHFGNgMOw |
| 045 | Do you even Ball? (Gold Stake Lessons) | 68 min | d6Rnd1EjVww |
| 046 | Satellite redemption or....? (Gold Stake Lessons) | 73 min | s_1IMzwU3Dk |
| 047 | The Doc HATES this Joker (Gold Stake Lessons) | 52 min | F13yqs2l5YU |
| 048 | Now we have proof how bad Satellite is (Gold Stake Lessons) | 59 min | L1fLFWYLcF4 |
| 049 | This seed got hands (Gold Stake Lessons) | 35 min | GQlrGySlYrU |
| 050 | We never talk about Satellite (Gold Stake Lessons) | 58 min | 7vMmCI1u3NA |
| 051 | Nobody likes you, Grimbo (Gold Stake Lessons) | 59 min | 7r2Fwo5F_dg |
| 052 | Zaino learned this move from ME (Gold Stake Lessons) | 42 min | JlRFOtTFWM0 |
| 053 | Stinky Deck, stinky Jokers? (Gold Stake Lessons) | 46 min | f1bCudltuHs |
| 054 | An unbelievable sequence of events (Gold Stake Lessons) | 47 min | Avn3tHFj_e4 |
| 055 | What does it mean "strong economy"? (Gold Stake Lessons) | 72 min | fbke3P922u0 |
| 056 | Honestly, an egregious amount of Jacks (Gold Stake Lessons) | 56 min | kXdD-awAmbg |
| 057 | The worst Castle of all time (Gold Stake Lessons) | 62 min | 7KmqIsLZ_-4 |
| 058 | A longer discussion about the Glyph Vouchers (Gold Stake Lessons) | 40 min | QRvxSAY-s1M |
| 059 | I can *almost* make this Joker worth it (Gold Stake Lessons) | 61 min | 89u0FbfBo3U |
| 060 | Sometimes you *can* do the fun thing (Gold Stake Lessons) | 55 min | 5N1Chl1wzM4 |
| 061 | Let me make a case for Stone cards (Gold Stake Lessons) | 52 min | kd_HiVz1fak |
| 062 | Flash Card is not *great* but still "underrated" (Gold Stake Lessons) | 56 min | s9BdU9BSAL8 |
| 063 | When there's "no good jokers" (Gold Stake Lessons) | 51 min | j8CCAZlNzBo |
| 064 | Another particularly rough start (Gold Stake Lessons) | 37 min | 7Eziem9JyyQ |
| 065 | A decent lesson in scaling and pacing (Gold Stake Lessons) | 53 min | lREwKHzOfCk |
| 066 | Telescope doesn't really support higher ranking Hands (Gold Stake Lessons) | 48 min | 0XjTIumE07A |
| 067 | Sixth Sense does something, sometimes, probably (Gold Stake Lessons) | 63 min | Rws2CYUpaJA |
| 068 | How much does it take to make Hit the Road work? (Gold Stake Lessons) | 53 min | ZgUmasv-N-Y |
| 069 | When is the Hand Size from the Painted deck actually useful? (Gold Stake Lessons) | 49 min | lXyiLAa_wcM |
| 070 | When is Blueprint "not enough?" (Gold Stake Lessons) | 59 min | WEHAuBmXtek |
| 071 | Managing Early/Eternal Rentals (Gold Stake Lessons) | 49 min | MKe6GC7Z4OI |
| 072 | Some musings about Risk and Reward (Gold Stake Lessons) | 38 min | 5WEEofSZOok |
| 073 | Take Overstock, I dare you (Gold Stake Lessons) | 49 min | tCBJo5ZIMrU |
| 074 | Don't threaten me with a good time (Gold Stake Lessons) | 52 min | p4mbEeW8668 |
| 075 | Is it ever worth it to build Full Houses? (Gold Stake Lessons) | 56 min | lBnfKrO4o6E |
| 076 | Let's not make a habit of this (Gold Stake Lessons) | 70 min | O0KQ2gtP8Jg |
| 077 | How many Rentals is too many? (Gold Stake Lessons) | 46 min | iTOTuXT4040 |
| 078 | Now THIS Joker is run-defining (Gold Stake Lessons) | 56 min | GGScX5CWANM |
| 079 | Making @seadubbs11 proud (Gold Stake Lessons) | 47 min | sZ_1dEijWLc |
| 080 | Why is this Joker so bad? (Gold Stake Lessons) | 64 min | QVTRCMRfH2U |
| 081 | Another gift from the Doc (Gold Stake Lessons) | 56 min | PZq6Uu9GBlk |
| 082 | A gift from the Doc (Gold Stake Lessons) | 52 min | _ZBkTKiLLsc |
| 083 | This run was interesting until the Blue Seal showed up (Gold Stake Lessons) | 52 min | gJVogwKQulY |
| 084 | This Joker NEEDS a rework (Gold Stake Lessons) | 44 min | fIUJr6BjzuQ |
| 085 | Tis the season for Credit Card (Gold Stake Lessons) | 36 min | WZiDcTYQQT8 |
| 086 | Just. Play. Pairs. (Gold Stake Lessons) | 48 min | qpCPl7Yo-pU |
| 087 | This one had some twists and turns (Gold Stake Lessons) | 48 min | vRsp4L4ZwiE |
| 088 | Are you still undervaluing this Joker? (Gold Stake Lessons) | 71 min | NplWtE4ObmU |
| 089 | A little Zodiac Deck, as a treat (Gold Stake Lessons) | 52 min | cY_ALLUArsE |
| 090 | STRAIGHTS stay winning (Gold Stake Lessons) | 58 min | wQ9U4Nw2KPs |
| 091 | ANOTHER chance at redemption (Gold Stake Lessons) | 49 min | AiqCCLJce2I |
| 092 | I replayed the ONE seed I lost, and THIS happened (Gold Stake Lessons) | 64 min | OVdk5v5uozo |
| 093 | Now THIS is the seed we've been looking for (Gold Stake Lessons) | 44 min | FWpbtvCdxJI |
| 094 | Some advice for the Painted Deck (Gold Stake Lessons) | 55 min | adSp51XGEgk |
| 095 | You're missing out on how fun this Joker can be (Gold Stake Lessons) | 45 min | SpAoTkiH9iY |
| 096 | Unused Hands are a viable source of income (Gold Stake Lessons) | 38 min | dgohZCByrS0 |
| 097 | Sorry, I can usually last longer than this (Gold Stake Lessons) | 25 min | PSD2zHaGpvo |
| 098 | I have a confession (Gold Stake Lessons) | 40 min | _2lcnYs_Biw |
| 099 | Some Jokers are less bad on the Magic Deck (Gold Stake Lessons) | 45 min | wuKivZwqHtE |
| 100 | They're not ready for the truth (Gold Stake Lessons) | 45 min | 8_UvdpwlWC4 |
| 101 | Whenever in doubt, Square Up (Gold Stake Lessons) | 42 min | zB8GQMN7NZ8 |
| 102 | 8 Ball is self enabling, right? (Gold Stake Lessons) | 48 min | leyis0O-oEs |
| 103 | This is among my *Splashiest* runs ever (Gold Stake Lessons) | 44 min | ndf16jXsDL0 |
| 104 | Win more games with an open mind (Gold Stake Lessons) | 49 min | BbeJIHd3-p8 |
| 105 | A big Wee [Joker] is all you need (Gold Stake Lessons) | 40 min | 5caXIRq1B6c |
| 106 | But you can only choose one (Gold Stake Lessons) | 43 min | a1YOxesUnRM |
| 107 | The rich get richer on the Ghost deck (Gold Stake Lessons) | 40 min | X1LxQ2f2wEU |
| 108 | Not all Gold Stake runs are difficult (Gold Stake Lessons) | 36 min | V6xazgveUMA |
| 109 | How much Mult does it actually take to beat Plasma Deck? (Gold Stake Lessons) | 49 min | 9OZdVKUXgcE |
| 110 | Ouija actually makes it *harder* to play Four of a Kind (Gold Stake Lessons) | 47 min | t0MGcuNgkBA |
| 111 | Square over Telescope, every time (Gold Stake Lessons) | 42 min | _MDwyFmIo5s |
| 112 | Red Card deniers in shambles (Gold Stake Lessons) | 54 min | W4SL6vuSFYg |
| 113 | How early is too early to Mime? (Gold Stake Lessons) | 70 min | EFUztSV_ilo |
| 114 | When is too early for a campfire? (Gold Stake Lessons) | 68 min | UCCPq-pkFgs |
| 115 | The TRUTH about Castle (Gold Stake Lessons) | 45 min | 5zi7Coveme4 |
| 116 | The Black Deck enables some quick runs (Gold Stake Lessons) | 32 min | arRqrivUIzo |
| 117 | I found a use for the Credit Card (Gold Stake Lessons) | 66 min | v-Da271jofU |
| 118 | This Joker is bait, too (Gold Stake Lessons) | 53 min | -DBnLZwSB6A |
| 119 | Don't fall for this obvious bait (Gold Stake Lessons) | 68 min | qrriD2_IDJA |
| 120 | Planning for the long game (Gold Stake Lessons) | 66 min | DTDgkp0vC9Y |
| 121 | What is even the point of playing Flushes? (Gold Stake Lessons) | 58 min | 6UV0xWdeiMs |
| 122 | Eternal? Or Betrayal? (Gold Stake Lessons) | 54 min | rjTuhBSdFa0 |
| 123 | With great Jokers comes great responsibility (Gold Stake Lessons) | 57 min | B9z5XMPG794 |
| 124 | Trust the process (Gold Stake Lessons) | 42 min | JAD9KeXi41U |
| 125 | The value of deck thinning (Gold Stake Lessons) | 53 min | Fe6hxt7ys2g |
| 126 | Vampire sucks /affectionate (Gold Stake Lessons) | 69 min | yl6Csf-qTAE |
| 127 | A case for nerfing the Blue Seal (Gold Stake Lessons) | 61 min | Xt8Gy6i1oLs |
| 128 | Let's talk about Skip Tags (Gold Stake Lessons) | 51 min | NXttotDW4LY |
| 129 | One particularly Lucky seed (Gold Stake Lessons) | 53 min | adidFE98AfU |
| 130 | The pants never fail (Gold Stake Lessons) | 61 min | -6YYQck2ifQ |
| 131 | A Matador run of all time (Gold Stake Lessons) | 43 min | lMCJFRECzr0 |
| 132 | When does it make sense to play Flushes? (Gold Stake Lessons) | 58 min | GT3j6Sp7PjY |
| 133 | I hate this card (Gold Stake Lessons) | 51 min | dmh61zuayvY |
| 134 | Madness is risky for sure (Gold Stake Lessons) | 43 min | fTdxQoolaYk |
| 135 | I picked this seed to be especially dubious (Gold Stake Lessons) | 51 min | luK3YJ7Tq5E |
| 136 | It gets bigger, I swear (Gold Stake Lessons) | 48 min | 9ucuEGMsAi0 |
| 137 | Where we're going, we don't need Mult (Gold Stake Lessons) | 55 min | vmf_iYPNSXs |
| 138 | Looks inside: The Needle (Gold Stake Lessons) | 55 min | CTscAuIb0J4 |
| 139 | Is Overstock ALWAYS worth it? (Gold Stake Lessons) | 69 min | 9vDE86eVa0Y |
| 140 | Can Straights win without Shortcut? (Gold Stake Lessons) | 59 min | KPhdFLy5UWs |
| 141 | Let's take a Shortcut (Gold Stake Lessons) | 60 min | ZfOttrUGjyY |
| 142 | The streak ender (Gold Stake Lessons) | 55 min | VpE_ar46aEw |
| 143 | Wow, another Castle so soon (Gold Stake Lessons) | 58 min | 4W3_mypSCwQ |
| 144 | Castle early and often (Gold Stake Lessons) | 60 min | ztX1Q6MF4eI |
| 145 | The benefits of a small deck (Gold Stake Lessons) | 75 min | YDU40I894IQ |
| 146 | More Nebula Deck (Gold Stake Lessons) | 46 min | HY1TPdCRgAs |
| 147 | Early Foils save runs (Gold Stake Lessons) | 54 min | 54RRyCyDfr0 |
| 148 | Money solves many problems (Gold Stake Lessons) | 68 min | Sh4C1kx11O4 |
| 149 | The mathematically optimal use of the Runner (Gold Stake Lessons) | 58 min | xrr8-Y1Kxts |
| 150 | *Some* advice for Plasma Deck (Gold Stake Lessons) | 60 min | d23RKYmdARU |
| 151 | Card Enhancements unlock useful xMult Jokers (Gold Stake Lessons) | 51 min | nVcvG5CQ2Go |
| 152 | Which Jokers are better on the Abandoned Deck? (Gold Stake Lessons) | 47 min | wlEfXDbaAwI |
| 153 | The importance of "generic" +Mult Jokers (Gold Stake Lessons) | 40 min | KpO3cV3FbbA |
| 154 | The importance of early planet cards (Gold Stake Lessons) | 47 min | 9RCcMBc9cJY |
| 155 | To suck or not to suck? (Gold Stake Lessons) | 44 min | FZm3yeYlHxY |
| 156 | Look for these winning Jokers (Gold Stake Lessons) | 47 min | FDJ-p0sksSA |
| 157 | Rentals are less punishing on the Green Deck (Gold Stake Lessons) | 46 min | LmfAa11Nie4 |
| 158 | One Polychrome does a lot (Gold Stake Lessons) | 60 min | nRqZD808bRc |
| 159 | Fortune Teller Son or 8 Ball Daughter? (Gold Stake Lessons) | 46 min | AOuGDu6Mz7c |
| 160 | This Joker is TOO strong (Gold Stake Lessons) | 39 min | rgtt7T0Mh5s |
| 161 | Adapting to her leaving you (Gold Stake Lessons) | 57 min | tu4EDzXR1bM |
| 162 | All bad choices on the Black Deck (Gold Stake Lessons) | 58 min | NRIqaGDUt00 |
| 163 | The Joker wins solo (Gold Stake Lessons) | 48 min | COIh9otUqoE |
| 164 | This cat has 9 lives, at least (Gold Stake Lessons) | 66 min | b2nZiul1etg |
| 165 | The perils of playing Flushes (Gold Stake Lessons) | 50 min | 7C93YBg3e2A |
| 166 | Navigating a dubious opener (Gold Stake Lessons) | 59 min | 1qxcT6QV4BA |
| 167 | How to force Pairs, even on the Checkered Deck (Gold Stake Lessons) | 41 min | _cm4N9FRGM0 |
| 168 | When and how to play Straights (Gold Stake Lessons) | 57 min | l70kPYu4cUE |
| 169 | Deckbuilding *wins* runs (Gold Stake Lessons) | 45 min | PqoTyLGgDvQ |
| 170 | *Another* reason to play Pairs (Gold Stake Lessons) | 49 min | x4shH6R_9w4 |
| 171 | Which +Mult is best? (Gold Stake Lessons) | 52 min | Em0sw0Ah7Fw |
| 172 | Which xMult is best? (Gold Stake Lessons) | 45 min | ikLzg6d9zsU |
| 173 | What makes the Nebula Deck difficult? (Gold Stake Lessons) | 44 min | CdjAitApPbg |
| 174 | Obelisk is for WINNERS only (Gold Stake Lessons) | 45 min | nKDzhycChEg |
| 175 | Which skips are even worth it? (Gold Stake Lessons) | 39 min | RvhjDvaZnHM |
| 176 | Who needs Blue Seals? (Gold Stake Lessons) | 42 min | 143d-zs8axE |
| 177 | Eternal Rental? No problem. (Gold Stake Lessons) | 60 min | vyU15NUg19U |
| 178 | The Avengers of Gold Stake Uncommons (C++, Black Deck ONLY) | 59 min | jqphoKxIa4U |
| 179 | Retrigger Jokers are just okay, on Gold Stake (C++, Painted Deck) | 49 min | TlgYfO-aKFY |
| 180 | Classic Gold Stake, not in the fun way (C++, Painted Deck) | 27 min | YpR4WuvVjbU |
| 181 | The Gold Stake all-stars (Lossless C++, Zodiac Deck) | 60 min | FXwhuP3NBpw |
| 182 | Digging for gold (Gold Stake, Red Bull Mod) | 56 min | TLZp_Eycv5I |
| 183 | Someday, my prince will (Gold Stake, Red Bull Mod) | 56 min | J22oor4ZdMk |
| 184 | Eye of Newt, Gall of Goat (Gold Stake, Red Bull Mod) | 74 min | VrYQDK2WjOs |
| 185 | Only YOU can prevent Pairslop (Gold Stake, Red Bull Mod) | 50 min | J4M_R0SFB7s |
| 187 | The HARDEST Gold Stake run (as determined by fans) | 51 min | rKUNZ_zgBEc |
| 189 | Gold Stake ALL Decks in less than 15 MIN each (Marathon Speedrun) | 223 min | jmieUtPvQek |
| 190 | Gold Stake ALL Decks in 4 hours (Balatro Marathon Speedrun [former] World Record) | 234 min | Q8_pDp19vWI |
| 191 | Gold Stake Balatro, Play-along Lesson (It's about sending a message) | 41 min | q2rkjPekIlM |
| 192 | Gold Stake Balatro, Play-along Lesson (Red Card vs the world) | 85 min | KwcBB0pDDJo |
| 193 | Gold Stake Balatro, Play-along Lesson (Squaring up all the way to the bank) | 42 min | Y4_xbFLZa2g |
| 194 | Gold Stake Balatro, Play-along Lesson (Greatness at any cost) | 60 min | -2-0nuP01T4 |
| 195 | Gold Stake Balatro, Play-along Lesson (Never stop playing Pairs) | 44 min | 0T7o_R_L9ZQ |
| 196 | Gold Stake Balatro, Play-along Lesson (Expensive Madness) | 35 min | duaSwioBhm4 |
| 197 | Gold Stake Balatro, Play-along Lesson (He SOLD the EGG???) | 68 min | DJiUOWOjRBk |
| 198 | Gold Stake Balatro, Play-along Lesson (RELEASE THE LIST) | 43 min | lHEKEywoiac |
| 199 | Gold Stake Balatro, Play-along Lesson (Oops, all Eternals again) | 50 min | KmPHTZigjpI |
| 200 | Gold Stake Balatro, Play-along Lesson (another lesson about Rentals) | 41 min | S_5YJ94SrIA |
| 201 | Gold Stake Balatro, Play-along Lesson (a couple dollaridoos) | 30 min | BFLXL2x8-aw |
| 202 | Gold Stake Balatro, Play-along Lesson (Rough Gem is fine) | 58 min | fACBci3TYQ0 |
| 203 | Gold Stake Balatro, Play-along Lesson (time to Square Up) | 57 min | 3vcIH7c_tOU |
| 204 | Let's go yass (Gold Stake, Erratic Deck) | 53 min | vQNf3DaiWrs |
| 205 | Are you undervaluing the Hiker? (Gold Stake, Plasma Deck) | 49 min | vOeyPlVCyyM |
| 206 | Fibonacci was for sure Ace (Gold Stake, Anaglyph Deck) | 39 min | 88wD98d72Js |
| 207 | Don't kink shame me (Gold Stake, Painted Deck) | 46 min | tR2sBf3OyL8 |
| 208 | He's just a funky little guy (Gold Stake, Zodiac Deck) | 55 min | 1NGIzObjRhg |
| 209 | Shortcut is my second favorite (Gold Stake, Checkered Deck) | 44 min | p7GdqzMR5wc |
| 210 | Don't be a h8ter (Gold Stake, Abandoned Deck) | 73 min | O_Ig07_wNNI |
| 211 | They told me this was the World Record (All Decks, Gold Stake Speedrun) | 292 min | wSmn07QIA7Q |
| 212 | Splash is the most $3-est Joker (Gold Stake, Ghost Deck) | 44 min | 30TSAeFCPKc |
| 213 | Which Legendary is worst? (Gold Stake, Nebula Deck) | 49 min | AYKX_Fjm6Xs |
| 214 | I don't usually pivot this way (Gold Stake, Magic Deck) | 52 min | yzkQNeZG5ik |
| 215 | With enough money, miracles are possible (Gold Stake, Green Deck) | 51 min | fOZuEBOqc3M |
| 216 | Nail-biter, 15-Deck Gold Stake Speedrun WR (for now) / Painted - Erratic Decks | 104 min | F1Xt9WUlsn4 |
| 217 | 15-Deck Gold Stake Speedrun WR (for now) / Nebula - Zodiac Decks | 118 min | AU3H2omlCwM |
| 218 | Sloppy 15-Deck Gold Stake Speedrun WR (for now) / Red - Magic Decks | 120 min | 1PVJau0aIS8 |
| 219 | "Stickerless" Gold Stake flow state (Yellow Deck) | 66 min | _pnbGTggzvY |
| 220 | A massive fumble on "Stickerless" Gold Stakes (Blue Deck) | 57 min | kQrlxqR42R8 |
| 221 | "Stickerless" Gold Stakes, proof of concept (Red Deck) | 53 min | fGwiqPUy3YE |
| 222 | A lesson about debt-to-income ratio (Gold Stake Balatro) | 48 min | eKJc4WzYfP4 |
| 223 | Black Deck gg ez (Gold Stake Balatro) | 45 min | Q2uuTB3pk-Y |
| 224 | Pair gaming, as a treat (Gold Stake Win Streak) | 57 min | 8DktEfPJlW8 |
| 225 | World's first! +Mult ONLY Plasma (Gold Stake / No xMult, No Chips, NO Planets) | 61 min | fTd-Ux1cGbs |
| 226 | Okay, this one had some twists and turns (Gold Stake Win Streak) | 52 min | GZfrjJE8yL4 |
| 227 | Not enough Jonkler slots (Gold Stake Win Streak) | 86 min | CuQ7rS0aoig |
| 228 | A tale of two xMult scalers (Gold Stake Win Streak) | 50 min | MQwak4QaChM |
| 229 | Who needs Discards anyway? (Gold Stake Win Streak) | 59 min | w5N4Jv_07aY |
| 230 | These Rentals are so worth it (Gold Stake Win Streak) | 65 min | VmjTmnXdUug |
| 231 | Just another day at BalatroU (Gold Stake Win Streak) | 66 min | -apU39ouRXU |
| 232 | Prepare for Trouble (Gold Stake Win Streak) | 70 min | YrjCh9jg3H8 |
| 233 | LIVE Gold Stake naneinf conclusion (random seed, no Legendaries) | 49 min | r45wxGwRue4 |
| 234 | LIVE Gold Stake naneinf (random seed, no Legendaries) | 159 min | Vgcu3RQN1do |
| 235 | Is Perkeo actually any good on Gold Stake? (High Stakes Balatro) | 139 min | Qt5Gj_LeA0s |
| 236 | Going even bigger on Gold Stake (High Stakes Balatro) | 65 min | 6l1fURwNjrc |
| 237 | Obviously the best Hand for Gold Stake  (Balatro Everyday 56) | 94 min | 4ru9k_pO6JU |
| 238 | Gold Stakes Pokemon (Modded Balatro) | 58 min | 2UMLFNONyys |
| 239 | Gold Stake Grudge Match (Twitch chat controls the game, Finale) | 221 min | q4xkpBmHOZQ |
| 240 | Bring me the Stones (Gold Stake Challenges 16) | 67 min | jnn8Za1whfo |
| 241 | The time he didn't get away with it (Gold Stake Challenges 15) | 48 min | 2rlwBNlZ4c0 |
| 242 | Vampire got me feeling a little Flushed (Gold Stake Challenges 14) | 62 min | GRRZ5HM1nA8 |
| 243 | That time I allowed myself to touch the Stones (Gold Stake Challenges 13) | 64 min | 1WyrFrqSwsA |
| 244 | Flush Houses, some case studies (Gold Stake Stream) | 140 min | _gOBi4XgjPQ |
| 245 | Say "no" to Medussy (Gold Stake Challenges 12) | 35 min | FIBWFGk04HE |
| 246 | Black Deck except it's Riff Raff every time (Gold Stake Stream) | 102 min | mmpjSqvwQFs |
| 247 | Driven Mad by dark Magic (Gold Stake Challenges 11) | 59 min | TdvZaHWsQW8 |
| 248 | Chatter tries to teach me what "roaching" means (Gold Stake Stream) | 48 min | noCpO9VtmFo |
| 249 | Dr. Acula helps grow my Wee [Joker] (Gold Stake Challenges 10) | 46 min | uNcTENavkHQ |
| 250 | I'm Blasting Off Again (Gold Stake Challenges 9) | 43 min | YAU2ax1qKCo |
| 251 | The thumbnail image is clickbait. (Gold Stake Challenges 8) | 49 min | NV651pq3_E4 |
| 252 | Blasting Off on Gold Stakes (Gold Stake Challenges 7) | 42 min | 0ngkc8jHmmg |
| 253 | Hey, I know a shortcut (Gold Stake Challenges 6) | 7 min | hcWt9dQjOe8 |
| 254 | Tell me again that Blue Seals don't need a nerf (Gold Stake Challenges 5) | 49 min | BXXvecc9XQ4 |
| 255 | *NO Discards* Golden Needle (Gold Stake Challenges 4) | 39 min | fEuoZ2MArUM |
| 256 | Unstoppable Force meets Immovable Object (Gold Stake Challenges 3) | 42 min | Q-dyPjNcAx0 |
| 257 | Balatro's easiest Challenge on the hardest Difficulty (Gold Stake Challenges 2) | 93 min | NEINe1_Yk7Y |
| 258 | Introducing: Gold Stake Challenge Mod! | 65 min | 0f0ceEklc1M |
| 259 | First Ever Chips Only *Gold Stake* without Plasma (ft. A Massive Wee [Joker]) | 86 min | izqD26lKa5A |
| 260 | Marble Maxing (Gold Stake, Ghost Deck) | 48 min | cXCA3yfe_jc |
| 261 | The Hiker is the Best Chips Joker??? (Gold Stake, Abandoned Deck) | 62 min | hzjfEX18C1M |
| 262 | I ate BoJangles before streaming and THIS happened (Gold Stake Stream) | 198 min | Yl1f6hpgdrk |
| 263 | Debt World Record (Gold Stake, Random Seed) | 32 min | Y0KMOj03etE |
| 264 | A particularly large Castle (Gold Stake) | 68 min | _Bk9BTbg_iQ |
| 265 | Are Rentals ever worth it? (Gold Stake, Abandoned Deck) | 79 min | _pHHi4GJg7w |
| 266 | Abandoned Deck Canio (No Planets Gold Stake) | 65 min | mW4ovpS2EFc |
| 267 | You *might* be right about the Dagger (Gold Stake with NO Planets, *again*) | 64 min | G94D6zs7zh8 |
| 268 | Obelisk skill issue (Gold Stake with NO Planets, again) | 58 min | o9BLaytGZWM |
| 269 | You're *still* wrong about Square Joker (Gold Stake with NO Planets) | 75 min | bDaNNfRpZ24 |
| 270 | You're wrong about Red Card (Gold Stake with only Common Jokers) | 92 min | AN6zMHHZ1JM |
| 271 | Kicking and Screaming (Gold Stake Black Deck) | 76 min | L_Zf4P1oTlw |
| 272 | Balatro Class: Some Parting Words (Gold Stake Ghost Deck) | 105 min | ORQ_mfJBUj0 |
| 273 | Guess how many Gold Stickers we got this run (Gold Stake Ghost Deck) | 32 min | 0aoGUSseBi0 |
| 274 | Don't Sleep on Thriller Bark (Gold Stake Ghost Deck) | 143 min | bP7XnYh0rwI |
| 275 | Balatro Class: Perkeo is not overpowered (on Gold Stakes) | 99 min | bSgSHMONgLQ |
| 276 | Balatro Class: Gold Stake until we lose (Yellow Deck) | 201 min | qxLK6KpYWEM |
| 277 | The clothes are coming off (Gold Stake Winstreaking again) | 167 min | VdaylZG8rPI |
| 278 | We're so back (Gold Stake Winstreaking) | 139 min | 9hGUd13YNx8 |
| 279 | Chill Gold Stakes Practice on my favorite deck | 143 min | ZZxH6i_99Pw |
| 280 | Gold Stakes, Abandoned and Checkered wins (Office Hours 20.2) | 115 min | zT34U1fwwc4 |
| 281 | Gold Stakes for Fun and Profit (Office Hours 20.1) | 117 min | 43NrenqU-0k |
| 282 | Gold Stake Money Management (Going for Gold ep4) | 49 min | QkHANP2e9i8 |
| 283 | Ante 15 on Gold Stake! (Going for Gold ep3) | 145 min | 4a_LYv9AFTc |
| 284 | Office Hours 3: Gold Stakes Warm-Ups (Abandoned Deck) | 110 min | 2IevsiM2x_I |
| 285 | State of the Meta (Gold Stakes on the Experimental Beta Patch) | 15 min | qx1H_ZIH9gQ |
| 286 | Balatro Daily ep56: Jokerless Gold Stake (Abandoned Deck) | 72 min | 9Le16J4thaY |
| 287 | New Gold Stake on Black Deck (Balatro 101 Patch Analysis) | 82 min | 1C7fDzdvunQ |
| 288 | Balatro Daily ep25: Hold R to restart (Gold Stake, Erratic Deck) | 95 min | mpfC85f9qlM |
| 289 | Balatro Daily ep24: Chips only speedrun (Gold Stake, Plasma Deck) | 34 min | z9TBHxXntyY |
| 290 | Balatro Daily ep23: Playing VERY loose (Gold Stake, Anaglyph Deck) | 74 min | 3-dvZZfQyp4 |
| 291 | Balatro Daily ep22: Proving Straights haters right (Gold Stake, Painted Deck) | 82 min | qpoShtGyCxo |
| 292 | Balatro Daily ep21: Now that's a High Card (Gold Stake, Zodiac Deck) | 50 min | ftCp7L4IqOc |
| 293 | Balatro Daily ep20: Pairs are for Squares (Gold Stake, Ghost Deck) | 40 min | JGwkIxEWXA4 |
| 294 | Balatro Daily ep19: Nebula Deck does not spark joy (Gold Stake) | 64 min | k-d9gDOHl0k |
| 295 | Balatro Daily ep18: DNA is just okay (Gold Stake, Magic Deck) | 70 min | REkVV0i2z5E |
| 296 | Balatro Daily ep17: How to get the most out of each Joker slot (Gold Stake, Black Deck) | 72 min | 3SrciYTwM9I |
| 297 | Balatro Daily ep16: Barely scraping by (Gold Stake, Green Deck) | 45 min | QXW_P-3iRVY |
| 298 | Balatro Daily ep15: Random Seed Perkeo (Gold Stake, Yellow Deck) | 60 min | yHLVxXvrQKo |
| 299 | Balatro Daily ep14: Gold Stake with XXL Pants (Blue Deck) | 109 min | eQHuzqW08F8 |
| 300 | Balatro Daily ep13: Gold Stake win streak with Pairs??? (Red Deck) | 70 min | sDrc8OCh0-Y |
| 301 | Balatro Daily ep12: Easy Straight Flushes on Gold Stake??? (14 Cards, Abandoned Deck) | 88 min | TKNVXaQOoo4 |
| 302 | Balatro Daily ep11: Easy Gold Stake win with High Card??? (Checkered Deck) | 82 min | xw3drHoIXbk |
