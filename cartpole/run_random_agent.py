import time
import pyvirtualdisplay
from tf_agents.environments import suite_gym
from tf_agents.policies import random_py_policy


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

random_policy = random_py_policy.RandomPyPolicy(
        env.time_step_spec(),
        env.action_spec()
    )

env.reset()
for i in range(100):
    env.render()
    action = random_policy.action(time_step=time_step).action
    print(f'Action: {action}')
    time_step = env.step(action)
    step, reward, discount, observation = time_step
    print(f'\nNext time step: {i}')
    print(f'Step: {step}')
    print(f'Reward: {reward}')
    print(f'Observation: {observation}')
    time.sleep(0.1)



