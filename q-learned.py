import gym
import pickle
import numpy as np
from time import sleep
from IPython.display import clear_output



env = gym.make("Taxi-v2").env
env.reset()

f = open('q_table.pckl', 'rb')
q_table = pickle.load(f)
f.close()

# Setting the number of iterations, penalties and reward to zero
epochs = 0
penalties, reward = 0, 0
frames = []
done = False
first_time = True
while not done:
    if first_time:
        action = env.action_space.sample()
        first_time = False
    else:
        action = np.argmax(q_table[state, :])
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into the dictionary for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    }
    )

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# Printing all the possible actions, states, rewards.
def printFrames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(1)


printFrames(frames)