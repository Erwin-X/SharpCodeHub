def estimate_gaes_and_returns(rewards, values, gamma=0.99, lamb=0.9):
    gaes_reversed = []

    last_gae = 0.
    for t in reversed(range(len(values))):
        next_value = values[t+1] if t < (len(values)-1) else 0
        delta = rewards[t] + gamma*next_value - values[t]
        gae = delta + gamma * lamb * last_gae
        gaes_reversed.append(gae)
        last_gae = gae
    gaes = gaes_reversed[::-1]
    returns = [a+v for a,v in zip(gaes, values)]
    return gaes, returns
