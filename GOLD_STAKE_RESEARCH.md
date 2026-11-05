# Gold Stake winners bank early, multiply late

Strong Gold Stake players win by following one plan and adapting it to the shop. Antes 1–2 are about surviving with whatever scores now. From antes 2–4 they build up to the $25 interest cap. Over antes 3–5 they commit to a scaling plan once the shop sends a clear signal. By antes 5–7 they convert to xMult or high hand levels, and they build for the hardest final boss (Violet Vessel, **1.2M**) rather than the usual **400k**. Most non-DrSpectred sources agree with him on that plan, on the $25 interest rule and on "generic beats narrow". The main dissent comes from nekoni, the only streak player whose detailed method could be read (best streak **11**). nekoni wins through **planet-levelled Straights** with Glass/Steel cards and calls Pair and High Card builds unstable at Gold. Community sources also add rules DrSpectred's summary lacks: exact sticker policies (with rentals, the interest threshold rises to **$28/$31**, by Steam players' account), ante-indexed hand-level targets, and a wider endgame board of **1 chip + 1–2 +Mult + 2–3 xMult**. The evidence on streaks is thin. The longest attributable non-DrSpectred streak is espaV's Twitch "PB: 41", which couldn't be checked further. Nothing anywhere documents a Gold streak in the hundreds. The longest streak in our whole evidence base is DrSpectred's own **66 straight Gold wins** in the local caption study, so his rules remain the best streak-validated rule set. For the agent, the practical result is a set of four candidate plans plus measurable per-ante targets (score per hand, money at shop exit, xMult count, main-hand level) for the plan-based shop player and for the gap study.

## Thin streak evidence, one readable streaker, and version traps

The source base is weak in specific ways, and every rule below should be weighted by where it comes from. Reddit couldn't be fetched at all, and TrueAchievements and Mobalytics returned HTTP 403, so their content survives only as search snippets. Of the videos and streams (DrSpectred, Balatro University, espaV, the "Road to 50" stream), only titles could be retrieved. The readable material is: the Balatro wiki, patch coverage, nekoni's 308-hour Japanese guide, The Mancunion's all-decks Gold guide, Addison Kindrick's Gold-specific tier addendum, two detailed Steam "Completionist++" threads, and a few general guides. This project's C++ engine supplies the mechanics, and per the project's notes its keyed RNG was checked against the real game.

| Source | What it is | Quality for this question |
|---|---|---|
| Balatro wiki (wiki.gg mirror) | Mechanics, tables | **High** for numbers; agrees with engine `tables_gen.hpp` |
| nekoni, note.com, Aug 2025 | 11-streak player, 308 h, all achievements twice | **Medium-high**: only detailed streaker method; one person's heuristics |
| The Mancunion, Sep 2025 | 200+ h, all 15 decks on Gold | **Medium**: deck-by-deck rules, no streak claim |
| Kindrick substack, Jan 2025 | Gold-specific tier addendum | **Medium** |
| Steam "Completionist++" threads, Dec 2024 / Jan 2025 | Players with full gold-sticker completions | **Medium**: concrete numbers, anecdotal |
| Steam threads, Apr 2024 | Pre- or just-post-1.0.1 | **Low-medium**: some reasoning outdated |
| TrueAchievements / Mobalytics / Choost | Snippets only | **Low** |
| Two Average Gamers, Jan 2026 | General combo guide | **Low**: says ante 8 needs "300 million" (wrong) |
| Switchblade Gaming, 2026 | Gold guide | **Low**: date conflicts, disputed sticker claims |

On streaks, espaV's recurring Twitch titles read "Gold Stake streaks [PB: 41]". The deck, the seed policy and the restart rules are unknown ([espaV VOD](https://m.twitch.tv/videos/2762394735); [channel](https://www.twitch.tv/espav)). A separate stream titled "Road to 50" may or may not be the same player ([YouTube](https://www.youtube.com/watch?v=gvNp5THUj_U)). Another channel ran a 9-streak attempt with **pairs banned** ([YouTube](https://www.youtube.com/watch?v=GQ68t7ZGbzM)). nekoni reports a best of **11** and says, without naming anyone, that some players win 50+ in a row ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). A "65-win streak on YouTube in its entirety" appears only as forum hearsay ([Coin-Return](https://forums.coin-return.org/threads/balatro-roguelike-poker-just-one-more-hand.782/page-2)). Our local caption study shows DrSpectred reaching **66 straight wins**, ending at 67, in the "Completionist++ with NO losses" playlist (local: `GOLD_STAKE_HEURISTICS.md` §17). That makes it likely, though not confirmed, that the hearsay refers to his run. Win-rate arithmetic puts these numbers in context. A 41-streak has a **1.3%** chance at a 90% win rate, **12%** at 95% and **44%** at 98%. A 66-streak has **26%** at 98% and **52%** at 99%. Documented streaks of this length therefore imply true win rates in the high 90s, possibly helped by restarts. A tracked "ordinary strong" player sat at **17.5%** over 104 hours ([lemm.ee](https://lemm.ee/post/30148259)), and 2024 posters guessed elite play peaks at **70/30** ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4294818016870729305)). No Red Deck–specific streak record was found.

Version matters for every rule. Patch 1.0.1 (final 1.0.1f) replaced Gold's −1 hand size with Rental jokers. It also replaced Orange's rising pack prices with Perishable jokers ([Press Start](https://press-start.com.au/news/2024/04/08/the-latest-balatro-update-has-some-massive-changes-that-should-make-winning-easier/); [Shacknews](https://shacknews.com/article/139415/balatro-update-101c-patch-notes)). Any guide that reasons from hand size or pack cost on Gold is quoting the old game, and that includes some current TrueAchievements text and YoruHawk's April 2024 "buy packs early" advice.

## Gold rules make a 400k finale on a $9 ante

Gold includes every lower stake's rules ([Stakes](https://balatrowiki.org/w/Stakes)):

- Small Blind pays no reward.
- Blind requirements scale at the Purple ("scaling 3") rate.
- **−1 discard**.
- Eternal stickers (from Black) and Perishable stickers (from Orange). A Perishable joker is debuffed, not destroyed, after **5 rounds**.
- Rental stickers: the joker costs **$1** to buy and charges **$3 every round**.

The engine rolls one poll for Eternal (30%) versus Perishable (30%), then a separate 30% roll for Rental (engine: `game.hpp` L2266–2288). That gives **28%** of shop jokers no sticker at all and **9%** Eternal+Rental ([Stickers](https://balatrowiki.org/w/Stickers)). Sticker choices therefore come up in almost every Gold shop.

The **Red Deck's +1 discard cancels Blue Stake's −1**, so a Gold Red run plays with **4 hands, 3 discards, hand size 8, 5 joker slots, 2 consumable slots and $4 starting money**. Red is the only deck that keeps three discards at Gold ([Red Deck](https://balatrowiki.org/w/Red_Deck); engine `game.hpp` L565–573). Under 1.0.1, random-seed Gold runs also make the first Legendary one you haven't yet won with on Gold ([Shacknews](https://shacknews.com/article/139415/balatro-update-101c-patch-notes)).

Blind targets come from the post-1.0.1f Purple table, which the engine matches ([Blinds and Antes](https://balatrowiki.org/w/Blinds_and_Antes)). The base is the Small Blind; Big Blind is ×1.5, Boss ×2, The Wall ×4 and Violet Vessel ×6. The per-hand columns below divide the boss by Red Deck's 4 hands ("minimum") and by 2 hands ("comfortable", meaning the build clears with spare hands and dollars). They are the targets the gap study should use.

| Ante | Small | Big | Boss | Wall / Vessel | Boss ÷ 4 hands | Boss ÷ 2 hands | × White |
|---|---|---|---|---|---|---|---|
| 1 | 300 | 450 | 600 | — | 150 | 300 | 1.0 |
| 2 | 1,000 | 1,500 | 2,000 | 4,000 | 500 | 1,000 | 1.25 |
| 3 | 3,200 | 4,800 | 6,400 | 12,800 | 1,600 | 3,200 | 1.6 |
| 4 | 9,000 | 13,500 | 18,000 | 36,000 | 4,500 | 9,000 | 1.8 |
| 5 | 25,000 | 37,500 | 50,000 | 100,000 | 12,500 | 25,000 | 2.3 |
| 6 | 60,000 | 90,000 | 120,000 | 240,000 | 30,000 | 60,000 | 3.0 |
| 7 | 110,000 | 165,000 | 220,000 | 440,000 | 55,000 | 110,000 | 3.1 |
| 8 | 200,000 | 300,000 | 400,000 | **1,200,000 (Vessel)** | 100,000 | 200,000 | 4.0 |

The step from ante 5 to ante 6 is **×2.4**, the steepest in the table. That supports DrSpectred's "ante 6 is the wall" without any source stating it in words. The economy is just as tight. A Gold ante with no skips pays **$0 + $4 + $5 = $9** in blind rewards, plus $1 per unused hand. Interest pays $1 per $5 held, capped at **$5 a round (reached at $25)**. The cap rises to $10 with Seed Money and $20 with Money Tree, and To the Moon adds a stacking extra ([Interest](https://balatrowiki.org/w/Interest); engine `game.hpp` L2141–2178). Interest (up to **$15 per ante**) is therefore worth more than all blind rewards combined. Steam players say the rental fee is taken *before* interest is counted, so holding one rental needs **$28** for full interest and two need **$31** ([Steam Completionist++](https://steamcommunity.com/app/2379780/discussions/0/596263931969820235)). **Our engine does the opposite.** It computes interest on the pre-fee balance and subtracts rentals afterwards (`game.hpp` L2147 vs L2171). If the community is right, the engine undercharges each rental by up to $1 a round. This should be checked against the Lua.

Shop odds explain why Gold strategy rests on generic engines rather than specific Rares. Each card slot is Joker 20 : Tarot 4 : Planet 4 (71/14/14%). Jokers come up **70% Common, 25% Uncommon, 5% Rare** from pools of 61/64/20 ([Shop](https://balatrowiki.org/w/Shop); [Jokers](https://balatrowiki.org/w/Jokers)). Any particular joker therefore shows in a given slot about **0.82% (Common), 0.28% (Uncommon) or 0.18% (Rare)** of the time. Over 48 slots without rerolls (2 slots, 3 shops, 8 antes) that is about **33%, 12.5% and 8%** of seeing it at all. Multiply by 0.28 for the chance it arrives with no sticker. These are our own calculations from the wiki rates; a Steam back-of-envelope gives the same order of magnitude ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4308326918582404174)). Rerolls cost **$5, rising $1 each**, and reset every shop. Pack slots are about 29% each for Arcana, Celestial and Standard, **8.7% for Buffoon** and 4.3% for Spectral ([Booster Packs](https://balatrowiki.org/w/Booster_Packs)). The Soul and Black Hole appear at **0.3%** per pack card ([Spectral Cards](https://balatrowiki.org/w/Spectral_Cards)).

Hand levels add fixed chips and mult per Planet ([Poker Hands](https://balatrowiki.org/w/Poker_Hands)). Saturn was buffed to +3 mult in 1.0.1 ([Shacknews](https://shacknews.com/article/139415/balatro-update-101c-patch-notes)).

| Hand | Base | Per level | Level 10 | Level 12 |
|---|---|---|---|---|
| Pair | 10 × 2 | +15 / +1 | 145 × 11 ≈ 1.6k | 175 × 13 ≈ 2.3k |
| Two Pair | 20 × 2 | +20 / +1 | 200 × 11 ≈ 2.2k | 240 × 13 ≈ 3.1k |
| Three of a Kind | 30 × 3 | +20 / +2 | 210 × 21 ≈ 4.4k | 250 × 25 ≈ 6.3k |
| Straight | 30 × 4 | +30 / +3 | 300 × 31 ≈ 9.3k | 360 × 37 ≈ 13.3k |
| Flush | 35 × 4 | +15 / +2 | 170 × 22 ≈ 3.7k | 200 × 26 ≈ 5.2k |
| Full House | 40 × 4 | +25 / +2 | 265 × 22 ≈ 5.8k | 315 × 26 ≈ 8.2k |
| Four of a Kind | 60 × 7 | +30 / +3 | 330 × 34 ≈ 11.2k | 390 × 40 ≈ 15.6k |

(Level values exclude card chips; our arithmetic.) This table drives nekoni's whole argument. A Straight's chips and mult both rise steeply, so its base grows roughly as **90n² + 30n** at level n. Its level-12 base of about 13k is **six times a level-12 Pair**. Scoring order also matters: +Mult jokers must sit left of xMult jokers, and joker Polychrome and Observatory apply late ([Scoring](https://balatrowiki.org/w/Scoring)).

## (1) Survive on output, bank to $25, commit on a signal

All the detailed sources share one spine. **Antes 1–2: take immediate output over plan purity.** nekoni takes Jolly Joker even off-plan, prefers jokers from Top-up Tag, Riff-Raff or Judgement because generated jokers carry no stickers, and wants one non-Rental chip or mult joker, or a Planet for the main hand, before the first boss ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). One Completionist++ player says survival beats the interest race: "building a functional joker setup to survive matters more" than reaching $25 fast. From Purple up, blinds grow too fast to spend antes preparing, so you build while surviving ([Steam](https://steamcommunity.com/app/2379780/discussions/0/596263931969820235)). One snippet-level source says flat +Mult of **15–25** "will carry you through ante 4" ([Mobalytics snippet](https://mobalytics.gg/blog/tier-lists/best-balatro-jokers/); confidence low).

**Antes 2–4: reach and hold $25.** Two Average Gamers and Switchblade target the cap by ante 3. The Mancunion says to reach it early and stay there. nekoni starts opening packs at about **$17 in ante 2** and holds $25 from antes 3–4 ([Two Average Gamers](https://www.twoaveragegamers.com/balatro-strategy-guide-the-joker-combos-that-actually-win-runs/); [Switchblade](https://www.switchbladegaming.com/strategy-games/balatro-gold-stake-guide/); [Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/); [nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)).

**Commit on a signal, not on a fixed ante.** nekoni lists explicit pivot triggers:

- Stuntman → High Card/Pair
- Spare Trousers → Two Pair
- early Ouija → Four/Five of a Kind
- Blue Seal or Perkeo early → commit harder to Planet/Straight and secure hand size and discards ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en))

Switchblade (low confidence) puts identification at antes 1–2 and commitment at **antes 4–6**, and says to check the upcoming boss before pivoting ([Switchblade](https://www.switchbladegaming.com/strategy-games/balatro-gold-stake-guide/)). Other sources warn against habit and against sticking. One Completionist++ player names "committing to a mediocre build instead of pivoting" as a leading cause of the **~80% of losses** he blames on himself ([Steam](https://steamcommunity.com/app/2379780/discussions/0/596263931969820235)). mangonaise says "don't decide quickly by heuristic", because "a blue seal is more powerful than many jokers" ([Steam](https://steamcommunity.com/app/2379780/discussions/0/604143165271744486)). Kindrick says the strongest runs are "emergent" rather than forced ([Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum)). The Mancunion's rule is to "play towards your deck" ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)).

**Antes 5–6: convert.** Sell economy jokers and add the first xMult (nekoni names Photograph). Buy Director's Cut (boss reroll). Two Average Gamers puts the same economy-to-scaling swap at antes 5–6 ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en); [Two Average Gamers](https://www.twoaveragegamers.com/balatro-strategy-guide-the-joker-combos-that-actually-win-runs/)).

**Ante 7: stockpile insurance.** nekoni collects extra xMult (Flower Pot, Ancient, Bloodstone) against Violet Vessel. At ante 8 he sells the surplus if the boss is not Vessel ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). mangonaise plans for the worst case: "assume Violet Vessel is the final boss", i.e. 1.2M, and keep enough Glass and Steel in a small enough deck to draw them ([Steam](https://steamcommunity.com/app/2379780/discussions/0/604143165271744486)).

Restarting is part of how streaks happen. 2024 posters said a "major streak" is unlikely without restarting for good ante-1 tags ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4346607305583788543/)), and The Mancunion restarts Black Deck runs until it gets an Investment Tag ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)). An agent that can't restart should expect lower win rates than streak numbers suggest.

## (2) Four main archetypes, and what each needs by ante

No source gives measured win rates by archetype at Gold. Every reliability claim is anecdotal, and the sources split into two camps. The majority (Steam players, YoruHawk, the "use pairs and high card" Completionist+ advice) holds that **Pair / Two Pair / High Card with generic scaling and xMult** is the most consistent route because those hands can always be drawn ([Steam YoruHawk](https://steamcommunity.com/app/2379780/discussions/0/4356744017103053782); [Steam](https://steamcommunity.com/app/2379780/discussions/0/4364626348105356076)). nekoni, the only streak player with a readable method, argues the opposite: **Pair and High Card need Chips, flat Mult and xMult all assembled in time, which is unstable at Gold**, so score should come from planet levels on a Straight ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). mangonaise, with two full completions, sits between them. He used **Straights (planet scaling) and Pairs (held-in-hand synergy with blue seals, Steel and Gold cards)**, found Pairs "superior for non-endless runs", and used Flush and High Card less ([Steam](https://steamcommunity.com/app/2379780/discussions/0/604143165271744486)).

| Archetype | Entry signal | Antes 1–2 | Antes 3–4 | Antes 5–6 | Antes 7–8 | Fails to | Confidence |
|---|---|---|---|---|---|---|---|
| **Pair / two-pair shell** | Any +Mult (Jolly, Half, Banner); Mercury | 1 flat +Mult; one Mercury carries | Chip source + one scaler (Green, Red Card, Runner, Square); $25 held | First xMult (Card Sharp, Blackboard, Cavendish, Duo) or a scaler near its endpoint | 1 chip + 1–2 +Mult + 2–3 xMult, xMult on the right; ≈100k per hand | Needle, Wall, Vessel without held steel or xMult; joker drought | Medium-high (majority view) |
| **Planet Straight** (nekoni) | Saturn / Orbital Tag, Telescope, blue seal, Perkeo, Shortcut / Four Fingers | 1 mult joker; Straight level **2–3** by ante 2 | Level **3–5**; ~2 mult jokers; hunt Blue Seals; $25 | Level **6–9**; xMult or Glass/Steel; sell economy; Director's Cut | Level **~12**; extra xMult for Vessel | Too few discards or too little hand size; Telescope pulled toward another hand; planet drought | Medium (one streaker) |
| **PhotoChad / glass-steel retrigger** | Hanging Chad (always take) + Photograph or glass | Chad on any lead card | Add Photograph; lead with a face card | Glass via Justice / Standard packs; steel on red- or blue-sealed cards | ≥×8 from two commons; glass retriggers vs Vessel | Plant, Mark, face debuffs; glass breaks (1 in 4) | Medium-high (broad praise) |
| **Baron/Mime steel kings** | Baron | High card to keep kings held | Chariot or DNA on kings; Hanged Man thinning | Mime; Blueprint/Brainstorm on Baron | Five steel kings held ≈ ×7.6 Baron alone, ~×11 with Mime (one guide) | Needs a specific Rare; disputed at high stakes | Low-medium |
| Flush | Checkered or suit enablers (Ancient + Lovers, Four Fingers) | First-round flush only | Suit tarots convert 3 cards for $3 | Ancient Joker with retriggers | "Flush planet levels rarely win Gold on their own" | Suit-debuff bosses | Low on Red Deck |
| Rank-uniform / 4oak / 5oak | Early Ouija, Erratic deck skewed to one rank | — | Death / Cryptid copies; Obelisk | Blue seals | Uniform-deck finishers | Pool pollution, rank debuffs | Low on Red Deck |

Sources for the table: nekoni's ante schedule ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)); the endgame board of 1 chip + 1–2 +Mult + xMult on the right ([Steam](https://steamcommunity.com/app/2379780/discussions/0/7252559508585171872)); PhotoChad as "at least x8" from two commons ([Two Average Gamers](https://www.twoaveragegamers.com/balatro-strategy-guide-the-joker-combos-that-actually-win-runs/)) and "highly rated" ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)), made possible by the 1.0.1 Hanging Chad buff to 2 retriggers ([Press Start](https://press-start.com.au/news/2024/04/08/the-latest-balatro-update-has-some-massive-changes-that-should-make-winning-easier/)); Baron/Mime figures from [games.gg](https://games.gg/balatro/guides/baron-mime-balatro-build/), with a Steam veteran calling it "weak at Orange" ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4356744017103053782)); the flush verdict from [Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum).

Arithmetic on the hand-level table shows what each plan needs at the ante 8 boss (100k per hand over 4 hands). A **level-12 Straight** gives about (360 + ~40 card chips) × 37 ≈ **15k** before jokers. Reaching 100k then takes about +30 Mult and roughly **×3.7 in combined xMult**, which is in line with nekoni's two or three xMult pieces. A **level-10 Pair** gives about (145 + ~22) × 11 ≈ **1.8k**. The joker board must supply roughly a **×55** factor through chips, +Mult and xMult, which is why every Pair source demands two or more xMult or held Steel. nekoni's example "1470 chips × 148 mult = 217,560 with no jokers" works out to a level-49 Straight, a showcase rather than a realistic ante-8 level. These numbers support the community's softer claim: Pair builds win through jokers, Straight builds through planets, and both need multiplicative pieces by ante 6–7.

The Mancunion's ante-8 recipe is wider than DrSpectred's "one scaler": one chip scaler (Runner), one +Mult scaler (Red Card) and **scaling** xMult (Hologram, Constellation, Vampire), plus Blueprint or Brainstorm moved onto the best joker ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)). 1.0.1 nerfed several of these engines: Vampire to X0.1 per enhancement, Steel Joker to X0.2, Campfire to X0.25, Bloodstone to 1/2 for X1.5. Glass Joker (X0.75) and Lucky Cat (X0.25) were buffed. Square Joker now starts at +0 ([Press Start](https://press-start.com.au/news/2024/04/08/the-latest-balatro-update-has-some-massive-changes-that-should-make-winning-easier/)). nekoni lists **Vampire and Square Joker as overrated** ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). Square triggers only on exactly 4 played cards, so a bare Pair doesn't feed it ([Steam](https://steamcommunity.com/app/2379780/discussions/0/596264096371887282)).

## (3) Economy rules priced in dollars

The economy rules are the most consistent across sources. They become sharper once Rental prices are included.

| Rule | Number | Sources / confidence |
|---|---|---|
| Interest target | $25 by ante 3; hold it from antes 3–4 | Mancunion, Two Average Gamers, nekoni — **High** |
| Softer floor | Leave **$15–20** at shop exit so the next shop opens at $20+ | [Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum) — Medium |
| Survival overrides interest | Skipping a needed (even Eternal) joker to protect interest is a named mistake; spend everything on a good Tarot if Needle or Water threatens | [Steam](https://steamcommunity.com/app/2379780/discussions/0/596263931969820235) — Medium |
| Greed also kills | A player lost ante 8 by 2% after buying 3 jokers instead of 2 | same — anecdote |
| Rental thresholds | Hold **$28** with one rental, **$31** with two (fee before interest) | same — Medium (engine disagrees, see above) |
| Rental timing | None early ("financial ruin"); at most **one, from ante 5–6**; Mancunion: only once the economy is stable; GAMES.GG: ante 7–8; Switchblade: not before ante 6 (~$30–36 cost by ante 8) | [nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en); [Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/); [GAMES.GG](https://games.gg/balatro/guides/balatro-economy-guide/); [Switchblade](https://www.switchbladegaming.com/strategy-games/balatro-gold-stake-guide/) — **High** on "not early" |
| Rental money jokers | Rental Golden Joker nets ~$1 a round; rental To the Moon "almost never worth it early" | [GAMES.GG](https://games.gg/balatro/guides/balatro-economy-guide/) — Medium |
| Hermit | Top Tarot; use *before* spending (max +$20) | [nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en); [GAMES.GG](https://games.gg/balatro/guides/balatro-economy-guide/) — High |
| Income jokers | Egg is the cheapest income; Matador pays $8 on a boss trigger, then sell it; Rocket ~$9 a round after 4 antes, Golden beats it for the first 3 antes | [nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en); [Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum) — Medium |
| Vouchers | Clearance Sale worth buying until ~ante 6; one Steam poster says an early voucher "is a good way to ruin a run" | nekoni; [Steam snippet](https://steamcommunity.com/app/2379780/discussions/0/4293691486113242157/?ctp=3) — conflicting |
| Never | Wheel of Fortune (75% nothing; 3.75% Polychrome) | [Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/) — Medium |
| Rerolling | Don't reroll to $0 or "all your money before a boss" without a boss-reroll voucher | [Steam](https://steamcommunity.com/app/2379780/discussions/0/4356744017103053782) — Medium |
| Skips | Almost never; exception: the **ante-1 Investment Tag** ($25 since 1.0.1); skip when the ante is likely lost anyway, to buy levels or packs | [Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/); [Steam](https://steamcommunity.com/app/2379780/discussions/0/596263931969820235); [Steam revlayle](https://steamcommunity.com/app/2379780/discussions/0/604143165271744486) — **High** |
| Lucky cards | ~$1.33 expected value each | nekoni — Low |

On Red Deck, unused hands are the only income on top of blinds and interest. DrSpectred's rule fits here: bank before spending hands on per-hand scalers (local §13). No non-DrSpectred source gives a Red-specific money rule. The Gold-specific twist on the general "$25 first" rule is that **every Rental raises the threshold by $3**. The Mancunion's "early economy joker even if Eternal" on Black Deck ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)) and a streamer's ante-1 Eternal Golden Joker or Chaos the Clown picks ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4356744017100525047)) show that Eternal is no reason to refuse an economy joker.

## (4) Hand choice: pairs by consensus, Straights by the one streaker

Most sources call High Card, Pair and Two Pair the easiest Gold hands. Some flush players level a flush just enough early, then move the deck to pairs or high card ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4356744017103053782)). After 1.0.1 one player believes almost any hand can win when forced, except straight flushes and the secret hands ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4424310816896517821)). Kindrick: "Flush planet levels rarely win a Gold Stake run on their own" ([Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum)).

The Red Deck changes this balance. The Mancunion's Red Deck rule is to **prioritise Full House, Straight and Flush, discard for better hands rather than playing the first decent one, and enhance cards because you play about 5 per turn**. It recommends Faceless, Mail-In Rebate, Trading Card, Burnt Joker and Castle ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)). nekoni calls −1 discard the hardest Gold handicap after Rentals and wants **hand size ≥9** for 5-card hands ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). Red's restored third discard therefore removes much of the case against 5-card hands. The local DrSpectred file agrees: the "extra discard suits straights" on Red (§13), and losing a discard is what "pushes everyone onto pairs" (§17). Red Deck is therefore where the Straight plan should most often beat the Pair plan.

**Levelling targets.** nekoni's are the only ante-indexed numbers found. His voucher order is **Paint Brush, Wasteful, Grabber, Clearance Sale, Telescope**, with a warning that Telescope can misfire if High Card gets played often. He uses Hieroglyph only when scaling is behind and avoids Petroglyph because it costs a discard ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)).

| Ante | Main-hand level |
|---|---|
| 2 | 2–3 |
| 3–4 | 3–5 |
| 5–6 | 6–9 |
| 7 | ~12 |

Planet supply routes: Telescope fills every Celestial pack with the main-hand planet, Observatory gives ×1.5 per held copy, Blue Seals make the planet for the last hand played, and Black Hole adds +1 to every hand ([Vouchers](https://balatrowiki.org/w/Vouchers); [Spectral Cards](https://balatrowiki.org/w/Spectral_Cards)). Perkeo + Observatory + Saturn is a named finisher ([Two Average Gamers](https://www.twoaveragegamers.com/balatro-strategy-guide-the-joker-combos-that-actually-win-runs/)). On Nebula, The Mancunion avoids Observatory ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)). No source gives Pair level targets by ante.

## (5) Deck manipulation turns tarots into joker-independent scoring

Every detailed source treats card upgrades as scoring that doesn't need the right joker to appear. nekoni builds **Glass and Steel cards "for scoring that doesn't depend on Jokers"**, puts Steel on Blue- or Red-Sealed cards, and rates these consumables highly ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)):

- Trance (Blue Seal)
- Sacrifice (+$20 and thinning)
- The Fool (to copy Planets)
- Death
- The Soul, always used, with Perkeo especially valued

mangonaise's summary is to score heavily from planets, enhancements and seals, keeping the deck small enough to draw Glass and Steel ([Steam](https://steamcommunity.com/app/2379780/discussions/0/604143165271744486)). The thinning tools are Hanged Man (−2 cards), Immolate (−5 cards, +$20), Death (copy) and Cryptid (two copies). Each suit tarot recolours 3 cards ([Tarot Cards](https://balatrowiki.org/w/Tarot_Cards); [Spectral Cards](https://balatrowiki.org/w/Spectral_Cards)). Hanged Man thins toward a King-heavy deck for Baron/Mime ([Two Average Gamers](https://www.twoaveragegamers.com/balatro-strategy-guide-the-joker-combos-that-actually-win-runs/)) and removes face cards that would reset Ride the Bus ([Steam](https://steamcommunity.com/app/2379780/discussions/0/596264096371887282)). Kindrick adds that Lovers + Ancient Joker makes "pseudo-Glass" cards, and Oops! All 6s makes Lucky cards reliable mult and money ([Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum)). Engine detail that matters for search: Ectoplasm's hand-size cost grows with each use (−1, then −2), and Standard-pack cards are **40% enhanced, 20% sealed and about 8% editioned** (engine `game.hpp` L3283–3292, L2415–2428). No source gives a deck-size target or ante timing for thinning. The only number is DrSpectred's "under about 30 cards" (local §1).

## (6) Gold joker priorities favour robust, sticker-tolerant pieces

Gold-specific lists differ from general ones in three ways. They reward jokers that work under boss debuffs and in small decks. They demote slow +Mult scalers and suit or face conditionals. They value guaranteed money.

**Hanging Chad has the broadest support.** mangonaise calls it "the clear winner" for Gold because it fits most builds, retriggering Lucky cards for money and Glass against Violet Vessel ([Steam](https://steamcommunity.com/app/2379780/discussions/0/604143165271744486)). It is nekoni's most-used joker ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). nekoni's other top picks are Hiker, Mime (with Blue/Steel seals), **Seltzer (for Vessel)**, Red Card grown by skipping packs at antes 3–4, and Chaos the Clown. His support jokers are Juggler, Turtle Bean, Troubadour, Drunkard and Merry Andy. Kindrick's Gold-built list rises from Marble, Certificate, Pareidolia, Flash Card and Obelisk through Oops! All 6s and Rocket to Ancient and, at the top, **Baseball Card**, which gives ×1.5 per Uncommon and raises the floor ([Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum)). mangonaise also says to prioritise Rare, Uncommon and consumable-generating jokers. The snippet-only Choost S-tier (Blueprint, Brainstorm, Cavendish, Triboulet, Throwback, Canio, Perkeo, DNA, Baron, Chicot) is a general list and mostly Rares or Legendaries ([Choost](https://choostgames.com/blog/balatro-jokers/)), so it matters less at Gold given the 8% chance of ever seeing a particular Rare.

**Three jokers are openly disputed.** TrueAchievements calls **Red Card, Ride the Bus and Supernova** Gold "traps" that "simply do not scale fast enough" ([TrueAchievements snippet](https://www.trueachievements.com/a424923/high-stakes-achievement)). The Mancunion calls Red Card a "reliable +Mult scaler" and lists Bus and Supernova for Blue Deck. nekoni rates Red Card highly. Kindrick calls Bus and Supernova riskier than Flash Card. The most likely resolution is deck context: per-hand scalers pay on Blue (+1 hand), Red Card pays when packs can be skipped cheaply. Flat hand-specific jokers (Lusty, Wrathful) become dead slots after ante 4 ([Two Average Gamers](https://www.twoaveragegamers.com/balatro-strategy-guide-the-joker-combos-that-actually-win-runs/)), and non-scoring utility is hard to justify in 5 slots ([Kindrick](https://addisonkindrick.substack.com/p/10-balatro-tier-list-addendum)).

**Sticker policy**, assembled from community sources, is concrete enough to encode. **Perishable** is free within 5 rounds of the ante-8 boss, i.e. from ante 7's small blind ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/)). **Rental**: never in the first shop; one player who did never got above $10 and lost ([Steam](https://steamcommunity.com/app/2379780/discussions/0/596263931969820235)). **Eternal** is acceptable on an early economy joker or a core you will keep, and Ghost Deck Hex makes a keeper the core. Sticker-chasing players win with 4 jokers and leave one slot as a dead sticker target ([Steam](https://steamcommunity.com/app/2379780/discussions/0/604143165271744486)), so **four working jokers are enough** at Gold, which matches DrSpectred's "3–4 working jokers".

## (7) Bosses: plan for Violet Vessel, reroll the rest

The consensus on bosses has four parts. Check the upcoming boss at the start of each ante (three shops to prepare). Buy **Director's Cut or Retcon** (boss reroll, $10) by antes 5–6. Keep Luchador to sell against a counter-boss. Size the endgame for Violet Vessel ([Mancunion](https://mancunion.com/2025/09/01/a-guide-to-gold-stake-balatro/); [Two Average Gamers](https://www.twoaveragegamers.com/balatro-strategy-guide-the-joker-combos-that-actually-win-runs/); [nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)). Rerolling every dollar before a boss without the voucher is a named mistake ([Steam](https://steamcommunity.com/app/2379780/discussions/0/4356744017103053782)). Bosses never repeat until all eligible ones have appeared, and minimum antes gate some (Plant from ante 4, Serpent 5, Ox 6) ([Boss Blinds](https://balatrowiki.org/w/Boss_Blinds)). That makes the late boss pool partly predictable from history.

nekoni's specific notes ([nekoni](https://note.com/teriburi/n/n9b9310f3f46b?hl=en)):

| Boss | Handling |
|---|---|
| The Ox | Avoid the most-played hand; keep Lucky cards as a money backup |
| The Flint | Reroll if possible |
| Burglar-type discard bosses | Sell Burglar first |
| Matador (joker) | Sell right after its boss |
| Amber Acorn | Harmless to Glass and xMult builds |
| Verdant Leaf | Ignorable at high hand levels |
| Cerulean Bell | Manage with discards |
| Violet Vessel | Seltzer, Glass retriggers or Boss Reroll |

On Wall and Vessel targets, Chicot resets both to ×2 ([Blinds and Antes](https://balatrowiki.org/w/Blinds_and_Antes)). Ante-8 boss misses by small margins are common in anecdotes (378k against 380k) ([Coin-Return](https://forums.coin-return.org/threads/balatro-roguelike-poker-just-one-more-hand.782/page-4)), which argues for building above the minimum target before the ante-8 boss.

## (8) Where non-DrSpectred sources disagree with his rules

| DrSpectred rule (local §1, §5, §10, §17) | Agreement | Disagreement | Verdict for the agent |
|---|---|---|---|
| Default hand is pairs; straights need Shortcut/Four Fingers/thinning | Steam majority, YoruHawk, Completionist+ advice | **nekoni**: Straight via planets is the default, Pair/High Card unstable; mangonaise: Straights first-class; Mancunion on Red: prioritise FH/Straight/Flush | Keep Pair as the default; **add Planet-Straight as a co-equal plan on Red Deck**, triggered by planet supply |
| One scaling source plus small flat support; "most of the time you don't need xMult" | DrSpectred's own §17 already softens this ("one x3 beats ante 8") | Mancunion: chip scaler + +Mult scaler + scaling xMult; Steam: 1 chip + 1–2 +Mult + rest xMult; nekoni: xMult at ante 5–6, stockpile at 7; Mobalytics snippet: xMult "almost required" | Target **≥2 xMult by ante 7** as the gap-study norm; one x3 only with a large base |
| Ride the Bus + Square alone beats ante 6 | Mancunion lists both for Blue | TrueAchievements: Bus a trap; Kindrick: Bus risky; nekoni: Square overrated; Square needs exactly 4 cards | Treat as deck-dependent (Blue), not a Red Deck rule |
| Retriggers rarely have targets except Hanging Chad | Chad praised everywhere | Seltzer (nekoni), Sock & Buskin/Hack with Ancient (Kindrick), Baron/Mime cited as winners | Mostly holds; add **Seltzer as a Vessel answer** |
| $25 before spending | Universal | Kindrick $15–20 floor; survival overrides; **$28/$31 with rentals** | Soft $25 floor, +$3 per rental |
| Don't skip unless the tag beats the shop | Universal | Explicit exceptions: ante-1 Investment, skip when the ante is lost anyway, nekoni's pack-skipping for Red Card | Compatible |
| Stickers (his §4) | Perishable as a bridge, rentals sold quickly | Community is stricter on rental timing (ante 5–8 only) and clearer that Perishable is free from ante 7 | Encode the community timing |
| Throttle (score just enough) | — | No other source discusses it | DrSpectred-only; unverified elsewhere |

## Comparison with GOLD_STAKE_HEURISTICS.md

**Confirmed.** The new research independently confirms the core of §1: survive antes 1–2 first, then one value generator, $25 interest, generic beats narrow, thin and copy, rarely skip (Investment the exception), Hanging Chad always. It also confirms §17's pacing (any +Mult early, then chips, then sell chips and money jokers for xMult), and that boss reroll is kept for Violet Vessel. The mechanics confirm his quoted ante-6 targets of 60k/90k/120k and that ante 5→6 is the steepest step (×2.4). His view that Red Deck's extra discard suits straights (§13) agrees with The Mancunion's Red rule. His Completionist++ streak of **66 wins** (§17) is longer than any other Gold streak we could document (espaV's 41 is next). The task assumed strong players beyond DrSpectred hold the best streak evidence, but on what we could find his rule set is actually the best streak-validated one.

**Added.** The new research adds several things the local file lacks:

- an exact blind and per-hand target table, sticker probabilities, and shop and joker odds;
- the rental interest thresholds of **$28/$31**, and an engine rental/interest ordering that differs from what players report;
- nekoni's **ante-indexed hand-level targets** (2–3 / 3–5 / 6–9 / ~12) and his signal-based pivot triggers (Stuntman, Spare Trousers, Ouija, Blue Seal/Perkeo);
- the Vessel-first sizing rule;
- the Perishable-free-from-ante-7 window;
- Seltzer and Baseball Card as Gold picks;
- the observation that Gold winners need only 4 working jokers.

**Contradicted or in tension.** Two items. First, the default hand: nekoni (11-streak) and mangonaise, with two completions, treat planet-levelled Straights as primary, against DrSpectred's §10 "higher hands only pay with a planet source". This is a difference in emphasis, since both name planets as the condition, but nekoni plays it as the default. Second, how much xMult is needed: other sources want two or three xMult by ante 7–8, while §5 says "most of the time you don't need xMult". §17's own later rule ("one x3 beats ante 8") narrows the gap. Smaller tensions concern Ride the Bus + Square at ante 6 and whether Square and Vampire are worth slots at all.

## Implications for the plan-based shop player and the gap study

**Candidate plans.** The shop player should carry four plans as standing hypotheses on Red Deck Gold, score each at every shop, and switch when a plan's ante target can't be met by its next checkpoint. The pivot triggers come from nekoni and DrSpectred.

| Plan | Entry / raise priority when | Ante 2 | Ante 4 | Ante 6 | Ante 8 board |
|---|---|---|---|---|---|
| **A. Pair shell** (default) | Any +Mult; Mercury; Stuntman → High Card variant; Spare Trousers → Two Pair | 1 +Mult joker; ≥1,000/hand | +Mult + chip source or scaler; Pair L3–4; ≥9k/hand comfortable | 1 xMult or scaler near endpoint; ≥30k/hand | 1 chip + 1–2 +Mult + ≥2 xMult (or Chad + glass); ≥100k/hand, Vessel plan |
| **B. Planet Straight** | Saturn/Orbital, Telescope, Blue Seal, Perkeo, Shortcut/Four Fingers, hand size ≥9 | Straight L2–3 + 1 mult joker | L3–5, 2 mult jokers, $25 | L6–9 + first xMult or Glass/Steel; Director's Cut | L~12, +~30 Mult, ×~3.7 combined xMult (≈100k/hand) |
| **C. Retrigger / enhancement** | Hanging Chad (always), Photograph, Justice/Glass, Seltzer, Mime + Blue/Steel seals | Chad on any lead | Photograph or 2+ Glass | Glass/Steel on sealed cards | Chad + Photograph (≥×8) or glass retriggers; Vessel answer |
| **D. Uniform-rank** (opportunistic) | Early Ouija, Baron, DNA, Death chains | — | Rank count rising | Blue seals, Steel kings | Rank finisher |

**Shared constraints and checks.** Several rules apply to every plan:

- Hold **$25 from ante 3**, plus $3 per rental.
- No rentals before ante 5. Perishables are free from ante 7. Accept Eternal on economy or core jokers.
- Sell economy jokers at antes 5–6.
- Buy boss reroll by ante 6.
- At the ante-7 shops, check whether the board clears **1.2M in four hands** and buy insurance xMult if not.

**Gap-study measures.** The gap study should log the same quantities for our agent's builds at each ante and compare them with the targets above:

- best single-hand score against the "boss ÷ 2" column;
- money at shop exit against $25 (+$3 per rental);
- count of xMult jokers (target ≥1 by ante 6, ≥2 by ante 7);
- main-hand level against nekoni's schedule (Straight plans);
- number of rentals and when each was bought;
- the joker-role mix (chip / +Mult / xMult / economy) against 1 / 1–2 / 2–3 / 0 at ante 8;
- whether the run had a Vessel answer.

One engine item belongs on the fix list before trusting the economy comparisons: confirm whether rental fees come out before interest. If the real game charges first, as players report, the engine is too generous to rentals.

## Conclusion

The biggest change in understanding is that "how strong humans win Gold" doesn't split into DrSpectred versus everyone else. It splits into **joker-driven Pair play versus planet-driven Straight play**, and Red Deck's restored third discard is exactly the condition that makes the second competitive. Both camps converge by ante 6 on the same requirement: a multiplicative layer (xMult jokers, Glass/Steel with retriggers, or a high-level 5-card hand) sized for Violet Vessel rather than the median boss. For an agent, this argues against hard-coding a single default. A plan-based shop player that keeps a Pair plan and a Straight plan open, scores each against explicit ante checkpoints, and switches on concrete signals (planet supply, Hanging Chad, Stuntman, Blue Seals) is closer to how the best documented players describe their choices.

The weakest link is still the streak evidence. Two of the three longest streaks in our base (espaV's 41, and the 50+/65 claims) are titles or hearsay, and the readable streaker managed 11. The gap study is therefore more trustworthy as a test of measurable requirements (score per hand, money, xMult count, hand level by ante) than of any one player's style. Those requirements follow from the blind table and the scoring formula, whoever's advice is followed.
