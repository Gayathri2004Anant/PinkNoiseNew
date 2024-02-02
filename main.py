"""Main script used for running all experiments, normally reads in a YAML file with parameters"""

from pathlib import Path
from types import MethodType

import d4rl  # noqa: F401
import dmc2gym
import gym
import mujoco_maze  # noqa: F401
import numpy as np
import oscillator  # noqa: F401
import tonic
import torch
from cluster.settings import read_params_from_cmdline, save_metrics_params
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from tonic.torch.agents import MPO

from cnrl import (MPO_CN, MPO_OU, BanditCNTrainer, BanditColoredNoise, ConstantColoredNoise,
                  MonitorCallback, MonitorLogger, OracleColoredNoise, RandomCNTrainer,
                  RandomColoredNoise, ScheduledCNTrainer, ScheduledColoredNoise,
                  SquashedDiagOUDistribution)


class GenEnv():
    """Generate gym environment from string with new seed each time"""
    def __init__(self, env, seed, sparse_reward, tonic=False):
        self.env = env
        self.seed = seed
        self.sparse_reward = sparse_reward
        self.tonic = tonic

    def __call__(self, get_policy=False):
        policy = "MlpPolicy"
        if self.tonic:
            if isinstance(self.env, str):
                env = lambda _: lambda: tonic.environments.Gym(self.env)    # noqa: E731
            elif isinstance(self.env[1], str):
                env = lambda seed: lambda: tonic.environments.ControlSuite('-'.join(self.env),    # noqa: E731
                    task_kwargs=dict(random=seed))
            env1 = tonic.environments.distribute(env(self.seed))
            env1.environments[0].seed(self.seed)
            self.seed += 1
            env2 = tonic.environments.distribute(env(self.seed))
            env2.environments[0].seed(self.seed)
            env = env1, env2
        elif isinstance(self.env, str):
            env = gym.make(self.env)
            env.seed(self.seed)
        elif isinstance(self.env[1], str):
            env = dmc2gym.make(*self.env, seed=self.seed)
        else:
            env = gym.make(self.env[0], **self.env[1])
            env.seed(self.seed)

        if self.sparse_reward:
            env.step_ = env.step

            def step(env, a):
                obs, _, done, info = env.step_(a)
                return obs, int(info['goal_achieved']), done, info

            env.step = MethodType(step, env)

        if env == 'FetchPickAndPlace-v1':
            env.reward_type = env.env.reward_type = 'dense'
            policy = "MultiInputPolicy"

        self.seed += 1    # Change seed such that it is different each time the environment is created

        if get_policy:
            return env, policy
        return env


if __name__ == "__main__":
    # Read settings from yaml file
    params = read_params_from_cmdline()
    params = params._mutable_copy()
    params.update(dict(params.conf))
    dir_ = Path(params.working_dir)
    name = f'{params.name}-{params.id}-{params.agent}-{params.noise}-{params.env}-{params.score}'
    monitor_kwargs = {x: params[x] for x in ['eval_every', 'n_eval']}
    mpo = params.agent == 'mpo'

    # Seeding
    seed = params.seed * 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Initialize environment
    env = params.env
    if env == 'oscillator':
        env = ('Oscillator-v0', dict(frequency=params.oscillator_f, quality=params.oscillator_Q))
    gen_env = GenEnv(env, seed, params.sparse_reward, mpo)
    env, policy = gen_env(get_policy=True)
    ep = env._max_episode_steps if not mpo else env[0].max_episode_steps

    if not mpo:
        # Initialize agent
        if params.agent == 'td3':
            agent = TD3(policy, env, tensorboard_log=(dir_ / 'tb'))
        elif params.agent == 'sac':
            agent = SAC(policy, env, tensorboard_log=(dir_ / 'tb'))

        n_actions = env.action_space.shape[-1] if not mpo else env[0].action_space.shape[-1]

        # Initialize noise
        callback = MonitorCallback(gen_env, dir_, **monitor_kwargs)
        if params.noise == 'wn' and params.agent == 'td3':
            # White noise (for TD3).
            agent.action_noise = NormalActionNoise(np.zeros(n_actions), params.noise_scale * np.ones(n_actions))
        elif params.noise == 'wn' and params.agent == 'sac':
            pass    # For SAC, 'wn' means unstructured noise, which is applied automatically.
        elif params.noise == 'ou' and params.agent == 'td3':
            # Ornstein-Uhlenbeck noise
            scale = params.noise_scale
            if params.ou_sc:
                scale *= np.sqrt((1 - (1 - params.theta*params.ou_dt)**2) / params.ou_dt)
            agent.action_noise = OrnsteinUhlenbeckActionNoise(
                np.zeros(n_actions), scale * np.ones(n_actions),
                params.theta, params.ou_dt)
        elif params.noise == 'ou' and params.agent == 'sac':
            scale = 1
            if params.ou_sc:
                scale *= np.sqrt((1 - (1 - params.theta*params.ou_dt)**2) / params.ou_dt)
            agent.actor.action_dist = SquashedDiagOUDistribution(n_actions, scale, params.theta, params.ou_dt)
        elif params.noise == 'oracle':
            # Oracle (=ICN: Iterated colored noise)
            callback = OracleColoredNoise(
                gen_env, dir_, params.colors, params.noise_scale, rng=rng, **params.oracle, **monitor_kwargs)
        elif params.noise == 'bandit':
            bandit_kwargs = params.bandit
            if params.bandit_method == 'cont':
                bandit_kwargs = params.bo
            elif params.bandit_method == 'ind':
                bandit_kwargs['dx'] = None
                bandit_kwargs['ls'] = None
            bandit_kwargs['window'] = params.memory // ep
            bandit_kwargs['rng'] = rng
            callback = BanditColoredNoise(
                gen_env, dir_, params.bandit_method, params.colors, params.noise_scale, params.score,
                ep, params.oracle_rollouts, rng, monitor_kwargs, bandit_kwargs)

        elif params.noise == 'random':
            callback = RandomColoredNoise(
                gen_env, dir_, params.colors, params.noise_scale,
                ep, params.random_method, rng, **monitor_kwargs)
        elif params.noise == 'constant':
            callback = ConstantColoredNoise(
                gen_env, dir_, params.beta, params.noise_scale,
                ep, rng, **monitor_kwargs)
        elif params.noise == 'schedule':
            callback = ScheduledColoredNoise(
                gen_env, dir_, (2, 0), params.schedule_method, params.noise_scale,
                ep, rng, **monitor_kwargs)

        # Train the agent
        agent.learn(total_timesteps=params.total_timesteps, log_interval=1, tb_log_name=name, callback=callback)

        # Save trained agent
        agent.save(dir_ / "agent")

        # Announce results to cluster-utils
        save_metrics_params({'final_eval_ret_mean': callback.evaluation_returns[-1][1].mean()}, params)
    else:
        # Initialze logger
        logger = MonitorLogger(dir_)
        tonic.logger.current_logger = logger

        # Default (constant noise) trainer
        trainer = tonic.Trainer(steps=params.total_timesteps, epoch_steps=params.eval_every,
            test_episodes=params.n_eval, save_steps=params.total_timesteps, replace_checkpoint=True)

        # Initialize agent and noise
        if params.noise == 'wn':
            agent = MPO()
            agent.initialize(env[0].observation_space, env[0].action_space, seed)
        elif params.noise == 'constant':    # (was 'const' by accident)
            agent = MPO_CN()
            agent.initialize(
                params.beta, env[0].max_episode_steps, rng, env[0].observation_space, env[0].action_space, seed)
        elif params.noise == 'ou':
            scale = 1
            if params.ou_sc:
                scale *= np.sqrt((1 - (1 - params.theta*params.ou_dt)**2) / params.ou_dt)
            agent = MPO_OU()
            agent.initialize(
                env[0].observation_space, env[0].action_space, seed, scale, params.theta, params.ou_dt)
        elif params.noise == 'random':
            agent = MPO_CN()
            agent.initialize(
                0, env[0].max_episode_steps, rng, env[0].observation_space, env[0].action_space, seed)
            trainer = RandomCNTrainer(params.random_method, params.colors, rng, dict(
                steps=params.total_timesteps, epoch_steps=params.eval_every, test_episodes=params.n_eval,
                save_steps=params.total_timesteps, replace_checkpoint=True))
        elif params.noise == 'schedule':
            agent = MPO_CN()
            agent.initialize(
                0, env[0].max_episode_steps, rng, env[0].observation_space, env[0].action_space, seed)
            trainer = ScheduledCNTrainer(params.schedule_method, dict(
                steps=params.total_timesteps, epoch_steps=params.eval_every,
                test_episodes=params.n_eval, save_steps=params.total_timesteps, replace_checkpoint=True))
        elif params.noise == 'bandit':
            agent = MPO_CN()
            agent.initialize(
                0, env[0].max_episode_steps, rng, env[0].observation_space, env[0].action_space, seed)

            bandit_kwargs = params.bandit
            if params.bandit_method == 'cont':
                bandit_kwargs = params.bo
            elif params.bandit_method == 'ind':
                bandit_kwargs['dx'] = None
                bandit_kwargs['ls'] = None
            bandit_kwargs['window'] = params.memory // ep
            bandit_kwargs['rng'] = rng

            trainer = BanditCNTrainer(params.bandit_method, params.colors, params.score, bandit_kwargs,
                logger, dict(steps=params.total_timesteps, epoch_steps=params.eval_every, test_episodes=params.n_eval,
                save_steps=params.total_timesteps, replace_checkpoint=True))

        # Train the agent
        trainer.initialize(agent, *env)
        trainer.run()

        # Save trained agent
        logger.save(dir_)

        # Announce results to cluster-utils
        save_metrics_params({'final_eval_ret_mean': logger.evaluation_returns[-1][1].mean()}, params)
