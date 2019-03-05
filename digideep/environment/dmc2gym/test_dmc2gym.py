# TODO:
#    Detail on camera and rendering: https://github.com/deepmind/dm_control/issues/12

import digideep.environment.dmc2gym
# from digideep.environment.dmc2gym import register_dmc
import gym
from gym.envs.registration import registry
from pprint import pprint
from time import sleep

if __name__=="__main__":    
    pprint(registry.env_specs)
    env = gym.make("DMBenchHumanoidStand-v0")
    env.reset()

    for t in range(10000):
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        env.render()
        # print(t)
        # sleep(0.05)

# DMBenchAcrobotSwingup-v0
# DMBenchAcrobotSwingup_sparse-v0
# DMBenchBall_in_cupCatch-v0
# DMBenchCartpoleBalance-v0
# DMBenchCartpoleBalance_sparse-v0
# DMBenchCartpoleSwingup-v0
# DMBenchCartpoleSwingup_sparse-v0
# DMBenchCheetahRun-v0
# DMBenchFingerSpin-v0
# DMBenchFingerTurn_easy-v0
# DMBenchFingerTurn_hard-v0
# DMBenchFishSwim-v0
# DMBenchFishUpright-v0
# DMBenchHopperHop-v0
# DMBenchHopperStand-v0
# DMBenchHumanoidRun-v0
# DMBenchHumanoidStand-v0
# DMBenchHumanoidWalk-v0
# DMBenchManipulatorBring_ball-v0
# DMBenchPendulumSwingup-v0
# DMBenchPoint_massEasy-v0
# DMBenchReacherEasy-v0
# DMBenchReacherHard-v0
# DMBenchSwimmerSwimmer15-v0
# DMBenchSwimmerSwimmer6-v0
# DMBenchWalkerRun-v0
# DMBenchWalkerStand-v0
# DMBenchWalkerWalk-v0
