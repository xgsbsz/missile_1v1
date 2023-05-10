"""
A simplified version from OpenAI Baselines code to work with gym.env parallelization.
"""
import os
import contextlib
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle
    使用cloudpickle来序列化内容（否则多进程会尝试使用pickle）)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.
    If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    如果子进程有MPI环境变量，MPI会认为子进程和父进程一样是一个MPI进程，并做一些坏事，比如挂起。
    这个上下文管理器是一个临时清除这些环境变量的笨办法

    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    一个抽象的异步的、矢量的环境。
    用于从一个环境的多个副本中批处理数据，因此
    每个观察值成为一批观察值，而预期的行动是一批要应用于每个环境的行动被应用于每个环境。
    """
    closed = False

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        重置所有的环境，并返回一个数组的观察值，或者观察值数组的dict。

        如果step_async还在做工作，那么这个工作将被取消，并且在再次调用step_async()之前，不应调用step_wait()
        直到step_async()被再次调用。
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.

        告诉所有的环境，开始采取一个步骤用给定的动作。
        调用step_wait()来获得该步骤的结果。

        如果一个step_async运行正在进行中，你不应该调用这个函数。

        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        等待step_async()采取的步骤。
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects

          返回（obs, rews, dones, infos）：
         - obs：一个观察值数组，或者一个观察值数组的dict/观察值的数组。
         - rews：一个奖励的数组
         - dones：一个 "回合完成 "的布尔数组
         - infos：一个信息对象的序列
        """
        pass

    def close_extras(self):
        """
        Clean up the extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        同步地步入环境，为向后兼容提供。
        """
        self.step_async(actions)
        return self.step_wait()


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)

    VecEnv是按顺序运行多个环境的，步骤和重置命令是一次发送到一个环境中。
    在调试时和num_env==1时很有用（在后一种情况下、避免了通信开销)
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(self.envs), env.observation_space, env.action_space)

        self.actions = None
        self.num_agents = getattr(self.envs[0], "num_agents", 1)
        print(self.num_agents)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obss, rewards, dones, infos = map(list, zip(*results))
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obss[i] = self.envs[i].reset()
            elif isinstance(done, (list, tuple, np.ndarray)):
                if np.all(done):
                    obss[i] = self.envs[i].reset()
            elif isinstance(done, dict):
                if np.all(list(done.values())):
                    obss[i] = self.envs[i].reset()
            else:
                raise NotImplementedError("Unexpected type of done!")
        self.actions = None
        return self._flatten(obss), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        obss = [env.reset() for env in self.envs]
        return self._flatten(obss)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode, filepath):
        if mode == 'txt':
            self.envs[0].render(mode, filepath)

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)


def worker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """Maintain an environment instance in subprocess,
    communicate with parent-process via multiprocessing.Pipe.

    Args:
        remote (Connection): used for current subprocess to send/receive data.
        parent_remote (Connection): used for mainprocess to send/receive data. [Need to be closed in subprocess!]
        env_fn_wrappers (method): functions to create gym.Env instance.

        在子进程中保持一个环境实例,通过multiprocessing.Pipe与父进程通信。

    参数：
        remote（连接）：用于当前子进程发送/接收数据。
        parent_remote（连接）：用于主进程发送/接收数据。[需要在子进程中关闭！] 。
        env_fn_wrappers（方法）：用于创建gym.Env实例的函数。
    """
    def step_env(env, action):
        obs, reward, done, info = env.step(action)
        if 'bool' in done.__class__.__name__:
            if done:
                obs = env.reset()
        elif isinstance(done, (list, tuple, np.ndarray)):
            if np.all(done):
                obs = env.reset()
        elif isinstance(done, dict):
            if np.all(list(done.values())):
                obs = env.reset()
        else:
            raise NotImplementedError("Unexpected type of done!")
        return obs, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space)))
            elif cmd == 'get_num_agents':
                remote.send(CloudpickleWrapper((getattr(envs[0], "num_agents", 1))))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Args:
            env_fns: iterable of callables - functions that create environments to run in subprocesses. Need to be cloud-pickleable
            context (str, optional): Defaults to 'spawn'.
            in_series (int, optional): number of environments to run in series in a single process. Defaults to 1.
                (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        # create Pipe connections to send/recv data from subprocesses,
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv().x
        super().__init__(nenvs, observation_space, action_space)

        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv().x

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)  # [[tuple] * in_series] * nremotes => [tuple] * nenvs
        self.waiting = False
        obss, rewards, dones, infos = zip(*results)
        return self._flatten(obss), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obss = [remote.recv() for remote in self.remotes]
        obss = self._flatten_series(obss)
        return self._flatten(obss)

    def close_extras(self):
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)

    @classmethod
    def _flatten_series(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0
        assert all([len(v_) > 0 for v_ in v])

        return [v__ for v_ in v for v__ in v_]


class ShareVecEnv(VecEnv):
    """
    Multi-agent version of VevEnv, that is, support `share_observation_space` interface.
    """
    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        super().__init__(num_envs, observation_space, action_space)
        self.share_observation_space = share_observation_space


class ShareDummyVecEnv(DummyVecEnv, ShareVecEnv):
    """
    Multi-agent version of DummyVecEnv, that is, support `share_observation_space` interface.

    DummyVecEnv is a VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case, avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(self.envs), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None
        self.num_agents = getattr(self.envs[0], "num_agents", 1)

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos = map(list, zip(*results))
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i] = self.envs[i].reset()
            elif isinstance(done, (list, tuple, np.ndarray)):
                if np.all(done):
                    obs[i], share_obs[i] = self.envs[i].reset()
            elif isinstance(done, dict):
                if np.all(list(done.values())):
                    obs[i], share_obs[i] = self.envs[i].reset()
            else:
                raise NotImplementedError("Unexpected type of done!")
        self.actions = None
        return self._flatten(obs), self._flatten(share_obs), self._flatten(rews), self._flatten(dones), np.array(infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs = map(np.array, zip(*results))
        return obs, share_obs


def shareworker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """Maintain an environment instance in subprocess,
    communicate with parent-process via multiprocessing.Pipe.

    Args:
        remote (Connection): used for current subprocess to send/receive data.
        parent_remote (Connection): used for mainprocess to send/receive data. [Need to be closed in subprocess!]
        env_fn_wrappers (method): functions to create gym.Env instance.
    """
    def step_env(env, action):
        obs, share_obs, reward, done, info = env.step(action)
        if 'bool' in done.__class__.__name__:
            if done:
                obs, share_obs = env.reset()
        elif isinstance(done, (list, tuple, np.ndarray)):
            if np.all(done):
                obs, share_obs = env.reset()
        elif isinstance(done, dict):
            if np.all(list(done.values())):
                obs, share_obs = env.reset()
        else:
            raise NotImplementedError("Unexpected type of done!")
        return obs, share_obs, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].share_observation_space, envs[0].action_space)))
            elif cmd == 'get_num_agents':
                remote.send(CloudpickleWrapper((getattr(envs[0], "num_agents", 1))))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class ShareSubprocVecEnv(SubprocVecEnv, ShareVecEnv):
    def __init__(self, env_fns, context='spawn', in_series=1):
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        # create Pipe connections to send/recv data from subprocesses,
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv().x
        ShareVecEnv.__init__(self, nenvs, observation_space, share_observation_space, action_space)

        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv().x

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results) # [[tuple] * in_series] * nremotes => [tuple] * nenvs
        self.waiting = False
        obs, share_obs, rewards, dones, infos = zip(*results) 
        return self._flatten(obs), self._flatten(share_obs), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)
        obs, share_obs = zip(*results)
        return self._flatten(obs), self._flatten(share_obs)
