import numpy as np
from gym import spaces
from collections import deque

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from .singlecombat_with_missle_task import SingleCombatDodgeMissileTask
from ..reward_functions import InterceptReward, AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator, waitMissileSimulator
from ..termination_conditions import LowAltitude, Overload, Timeout, Angleview


class InterceptTask(SingleCombatDodgeMissileTask):
#     '''
#     Control target heading with discrete action space
#     '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            # PostureReward(self.config),
            # AltitudeReward(self.config),
            # EventDrivenReward(self.config),
            # ShootPenaltyReward(self.config),
            InterceptReward(self.config)#对拦截奖励
        ]
        self.termination_conditions = [  # 选取仿真结束条件
            # Angleview(self.config),
            # Overload(self.config),
            # LowAltitude(self.config),
            Timeout(self.config)
        ]



    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return super().get_obs(env, agent_id)

    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))

    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        super().reset(env)

    def step(self, env):  # 发射导弹
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])  # 导弹uid
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0],
                                            uid=new_missile_uid),
                    waitMissileSimulator.create(parent=agent, target=agent.enemies[0],
                                            uid=new_missile_uid)
                )  # 创建导弹指令、临时的（导弹发射不是一直的是有装弹过程），调用一个类MissileSimulator
                self.remaining_missiles[agent_id] -= 1  #