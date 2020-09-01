import numpy as np
import math


def logistic(x, a, b):
    return 1 / (1 + math.exp(-a * x + b))


def fair_coin(h, t):
    ''' returns P(D|H1) for a sequence of h heads and t tails'''
    return 1 / (2 ** (h + t))


def unfair_coin(h, t):
    '''returns P(D|H2) for a sequence of h heads and t tails'''
    # theta = np.random.uniform()
    # probability of h heads is theta ^ h and t tails is (1-theta) ^ t
    # so P(D|theta) = theta ** h * (1-theta) ** t
    res = 0
    space = np.linspace(0, 1, 100)
    for theta in space:
        # res += P(D|theta) * P(theta|H2) and P(theta|H2) is 1/number of slices
        res += theta ** h * (1 - theta) ** t * (1 / 100)
    return res


def log_posterior_odds(d_h1, d_h2):
    ''' return log posterior odds ration given P(D|H1) and P(D|H2)'''
    return log(d_h1 / d_h2)


sequence_1 = [6, 5, 2, 6, 5, 1, 7, 5, 1]
sequence_2 = [7, 5, 2, 6, 4, 1, 6, 3, 1]
sequence_3 = [5, 3, 1, 6, 4, 1, 6, 2, 1]
sequence_4 = [7, 3, 1, 6, 2, 1, 7, 4, 1]

c1 = (3, 2)
c2 = (1, 4)
c3 = (5, 0)
c4 = (4, 6)
c5 = (8, 2)
c6 = (0, 10)
c7 = (10, 16)
c8 = (21, 5)
c9 = (26, 0)

coins = [c1, c2, c3, c4, c5, c6, c7, c8, c9]

# model_fair = []
# model_unfair = []
model = []
for coin in coins:
    h, t = coin
    fair = fair_coin(h, t)
    unfair = unfair_coin(h, t)
    lpo = fair / unfair
    model.append(logistic(lpo, 1, 0.2))

print(model)
