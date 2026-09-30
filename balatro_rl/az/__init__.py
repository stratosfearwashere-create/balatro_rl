"""Unified agent: one decision process for the whole run (blind select, cards, consumables, shop, packs).

    world.py     Action, World (a game plus loop counters), resampling hidden information
    actions.py   A: every legal action, with exact scores and joker-state / side-effect deltas
    solver.py    B: Monte Carlo round solver, P(clear) and expected score per play/discard
    features.py  C: state tokens and candidate-action features for the network
    net.py       C: transformer, residual policy head over candidates, value and auxiliary heads
    search.py    D: Gumbel tree search with chance nodes on the real simulator
    agent.py     E/F: priors, adaptive search budget, the guarded auto-play shortcut
    train.py     self-play with search, training, evaluation
"""
