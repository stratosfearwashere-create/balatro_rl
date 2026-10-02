"""Build value: V(build, ante, money, deck, stake) learned by Monte Carlo from the outcomes of many games.

records.py   compact build records and their outcome labels
gen.py       play games with a fixed agent and write the records (python -m balatro_rl.value.gen)
model.py     the network: ordinal P(reach ante k), P(win), blinds after
train.py     supervised training and calibration against Phi (python -m balatro_rl.value.train)
"""
