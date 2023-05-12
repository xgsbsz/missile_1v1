import numpy as np
from gym import spaces
from collections import deque
from ..core.catalog import Catalog as c

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from .singlecombat_with_missle_task import SingleCombatDodgeMissileTask
from ..reward_functions import InterceptReward, AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator, waitMissileSimulator
from ..termination_conditions import LowAltitude, Overload, Timeout, Angleview

from .task_base import BaseTask
# from ..envs.singlecontrol_env import SingleControlEnv


class InterceptTask(BaseTask):
#     '''
#     Control target heading with discrete action space
#     '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            # AltitudeReward(self.config),
            # EventDrivenReward(self.config),
            # ShootPenaltyReward(self.config),
            # InterceptReward(self.config)#对拦截奖励
        ]
        self.termination_conditions = [  # 选取仿真结束条件
            # # Angleview(self.config),
            # Overload(self.config),
            # LowAltitude(self.config),
            Timeout(self.config)
        ]


    @property
    def num_agents(self):
        return 4

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_heading  (unit: °)
            c.delta_velocities_u,               # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  # 3. altitude  (unit: m)
            c.attitude_roll_rad,                # 4. roll      (unit: rad)
            c.attitude_pitch_rad,               # 5. pitch     (unit: rad)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        # obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        # norm_obs = np.zeros(12)
        # norm_obs[0] = obs[0] / 1000  # 0. ego delta altitude (unit: 1km)
        # norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        # norm_obs[2] = obs[2] / 340  # 2. ego delta velocities_u (unit: mh)
        # norm_obs[3] = obs[3] / 5000  # 3. ego_altitude   (unit: 5km)
        # norm_obs[4] = np.sin(obs[4])  # 4. ego_roll_sin
        # norm_obs[5] = np.cos(obs[4])  # 5. ego_roll_cos
        # norm_obs[6] = np.sin(obs[5])  # 6. ego_pitch_sin
        # norm_obs[7] = np.cos(obs[5])  # 7. ego_pitch_cos
        # norm_obs[8] = obs[6] / 340  # 8. ego_v_north    (unit: mh)
        # norm_obs[9] = obs[7] / 340  # 9. ego_v_east     (unit: mh)
        # norm_obs[10] = obs[8] / 340  # 10. ego_v_down    (unit: mh)
        # norm_obs[11] = obs[9] / 340  # 11. ego_vc        (unit: mh)
        # norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        # return norm_obs
        return super().get_obs(env, agent_id)

    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))

    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        return super().reset(env)#原无 return


def step(self, env):  # 发射导弹
    BaseTask.step(self, env)
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