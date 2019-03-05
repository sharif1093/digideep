import digideep.environment.dmc2gym
import gym
import joblib

if __name__=="__main__":
    # env = gym.make('Acrobot-v1')
    env = gym.make("DMBenchHumanoidStand-v0")
    joblib.dump(dict(env=env), 'pickletest.pt')

