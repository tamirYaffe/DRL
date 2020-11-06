
# Algorithm
# 1. Create a lookup table containing the approximation of the Q-value for each state-action pair.
# 2. Initialize the table with zeros.
# 3. Choose initial values for the hyper-parameters: learning rate 𝛼, discount
#    factor 𝛾, decay rate for decaying epsilon-greedy probability.
# 4. Get initial state s
# 5. For k = 1, 2, ... till convergence (5000 episodes)
#       sample action a, get next state s' (sample actions using decaying 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦)
#       if s' is terminal: (max of 100 steps per episode)
#           target = R(s, a, s')
#           sample new initial state s'
#       else:
#           target = R(s, a, s') + ymaxQk(s', a')
#       Qk+1(s, a) = (1- alpha) * Qk(s, a) + alpha * (target)
#       s = s'
