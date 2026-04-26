import random
P = {
    0: {
        0: [(1.0, 0, -1, True)],
        1: [(1.0, 0, -1, True)],
    },
    1: {
        0: [(1.0, 0, -1, True)],
        1: [(1.0, 2, 0, False)],
    },
    2: {
        0: [(1.0, 1, 0, False)],
        1: [(1.0, 3, 0, False)],
    },
    3: {
        0: [(1.0, 2, 0, False)],
        1: [(1.0, 4, 1, True)],
    },
    4: {
        0: [(1.0, 4, 1, True)],
        1: [(1.0, 4, 1, True)],
    },
}

state=2
c=0
while True:
    action = int(input("Enter action (0 for left, 1 for right): "))
    prob, next_state, reward, terminal = outcomes = P[state][action][0]
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Terminal: {terminal}")
    state = next_state
    c+=1
    disc=(0.99**c)*reward
    if terminal:
        print("discount:",disc)
        print("Reached terminal state!")
        break