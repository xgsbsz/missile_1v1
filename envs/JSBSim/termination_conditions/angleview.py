from .termination_condition_base import BaseTerminationCondition
import numpy as np
from ..core.catalog import Catalog as c

class Angleview(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """

    def __init__(self, config):
        super().__init__(config)




    def get_termination(self, task, env, agent_id, info={}):
        """
        目标线与基准角夹角视线角范围限制
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """

        done = bool(env.agents[agent_id].get_property_value(c.delta_heading)>=135)

        if done:
            self.log(f"{agent_id} step limits! Total Steps={env.current_step}")
        success = False
        return done, success, info
