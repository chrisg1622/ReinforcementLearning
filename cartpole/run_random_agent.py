import time
import numpy as np
import pyvirtualdisplay
from tf_agents.environments import suite_gym


display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)

print('Observation Spec:')
print(env.time_step_spec().observation)

print('Reward Spec:')
print(env.time_step_spec().reward)

print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)

env.reset()
for i in range(100):
    env.render()
    action = np.random.choice([0, 1])
    time_step, reward, discount, observation = env.step(action)
    print(f'\nNext time step: {i}')
    print(f'Time Step: {time_step}')
    print(f'Reward: {reward}')
    print(f'Observation: {observation}')
    time.sleep(0.1)



