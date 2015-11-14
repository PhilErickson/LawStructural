""" Model constants """

UNRANKED = 195
RBAR = 194
BETA = .99  # Discount factor
N_AESIMS = 1#3  # Number of simulations for first-stage sim estimation
N_SCHOOLS = 200  # Number of possible schools
N_ALTPOL = 4#1250  # Number of alternate policies to simulate
N_SIMS = 2#10  # Number of simulations per policy
N_PERIODS = 10#80  # Truncation period for value function
N_THREADS = 8  # Number of processes/threads for running in parallel
N_KNOTS = 10  # Number of knots for policy function splines
