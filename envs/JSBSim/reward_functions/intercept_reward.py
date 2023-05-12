from .reward_function_base import BaseRewardFunction
from ..core.simulatior import MissileSimulator, waitMissileSimulator
import numpy as np
from ..core.catalog import Catalog as c

class InterceptReward(BaseRewardFunction):
    "拦截导弹的奖励，终端和过程 终端以距离zem=50cm为拦截成功，过程以角度计算"

    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
       #在拦截过程中需要重置的变量？
        return super().reset(task, env)



    def get_reward(self, task, env, agent_id):

        reward = 0
        min_dist = 50
        # 还需要改成导弹与导弹的相对位置计算
        if env.agents[agent_id].get_property_value(c.delta_altitude) <= min_dist: #终端
            reward +=100
        else:
            reward +=0

        reward_dist = 1*( -np.log(env.agents[agent_id].get_property_value(c.delta_altitude)**2+1e-8))#距离
        # reward_energy = 1*(-np.log(integrate(np.linalg.norm((ny,nz),(self._t,0))/(v_m*self._t*pow(abs(Rxyz)++0.5)))))
        print(reward_dist)
        # print(reward_energy)

        reward += reward_dist
                  # +reward_energy

        return self._process(reward, agent_id)