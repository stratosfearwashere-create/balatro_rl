"""Blinds replayed after a -1 Ante voucher (Hieroglyph/Petroglyph) don't count as progress."""
from balatro_rl.sim.game import Game


def test_furthest_blind_ignores_replays():
    g = Game(seed=1, stake="WHITE")
    g.ante, g.blind_idx = 2, 1
    g.win_round()                       # ante 2 big blind = blind 5
    assert (g.blinds_beaten, g.furthest_blind) == (1, 5)
    g.ante, g.blind_idx = 1, 2          # set back an ante, then beat the ante 1 boss
    g.win_round()
    assert (g.blinds_beaten, g.furthest_blind) == (2, 5)
    g.ante, g.blind_idx = 8, 2
    g.win_round()                       # ante 8 boss = blind 24 = a win
    assert g.furthest_blind == 24
