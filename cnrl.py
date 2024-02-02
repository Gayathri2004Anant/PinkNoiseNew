"""Main library file.

Includes all methods for using colored noise and OU noise with the RL
algorithms, as well as bandit and schedule implementations.
"""

import os
import pickle
import time

import colorednoise as cn
import gym
import numpy as np
import tonic
import torch as th
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.noise import ActionNoise, OrnsteinUhlenbeckActionNoise
from tonic import logger
from tonic.torch.agents import MPO
from tonic.utils.trainer import Trainer
from tqdm.auto import tqdm


class ColoredNoiseProcess():
    """Colored noise implemented as a process that allows subsequent samples.
    Implemented as a buffer; every "chunksize[-1]" items, a cut to a new time series starts.
    """

    def __init__(self, beta=1, scale=1, chunksize=32768, largest_wavelength=256, rng=None):
        self.beta = beta
        if largest_wavelength is None:
            self.minimum_frequency = 0
        else:
            self.minimum_frequency = 1 / largest_wavelength
        self.scale = scale
        self.rng = rng

        # The last component of chunksize is the time index
        try:
            self.chunksize = list(chunksize)
        except TypeError:
            self.chunksize = [chunksize]
        self.time_steps = self.chunksize[-1]

        # Set first time-step such that buffer will be initialized
        self.idx = self.time_steps

    def sample(self):
        self.idx += 1    # Next time step

        # Refill buffer if depleted
        if self.idx >= self.time_steps:
            self.buffer = cn.powerlaw_psd_gaussian(
                exponent=self.beta, size=self.chunksize, fmin=self.minimum_frequency, rng=self.rng)
            self.idx = 0

        return self.scale * self.buffer[..., self.idx]


class ColoredActionNoise(ActionNoise):
    """Action noise using colored noise processes (independent for each action dimension)."""
    def __init__(self, beta: np.ndarray, sigma: np.ndarray, seq_len, rng=None):
        super().__init__()
        self._beta = beta
        self._sigma = sigma
        self._gen = [ColoredNoiseProcess(beta=b, scale=s, chunksize=seq_len, largest_wavelength=None, rng=rng)
                     for b, s in zip(beta, sigma)]

    def __call__(self) -> np.ndarray:
        return np.asarray([g.sample() for g in self._gen])

    def __repr__(self) -> str:
        return f"ColoredActionNoise(beta={self._beta}, sigma={self._sigma})"


class SquashedDiagCNDistribution(SquashedDiagGaussianDistribution):
    """
    Colored Noise distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure
    bounds. Used for Soft Actor-Critic with colored noise exploration in lieu of SquashedDiagGaussianDistribution.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, beta: np.ndarray, seq_len, epsilon: float = 1e-6, rng=None):
        super().__init__(action_dim, epsilon)
        self.cn_processes = [ColoredNoiseProcess(beta=b, chunksize=seq_len, largest_wavelength=None, rng=rng)
                             for b in beta]

    def sample(self) -> th.Tensor:
        cn_sample = th.tensor([cnp.sample() for cnp in self.cn_processes]).float()
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev*cn_sample
        return th.tanh(self.gaussian_actions)


class SquashedDiagOUDistribution(SquashedDiagGaussianDistribution):
    """
    Ornstein-Uhlenbeck distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure
    bounds. Used for Soft Actor-Critic with colored noise exploration in lieu of SquashedDiagGaussianDistribution.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, scale, theta, dt, epsilon: float = 1e-6):
        super().__init__(action_dim, epsilon)
        self.oup = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim),
            scale*np.ones(action_dim), theta, dt)

    def sample(self) -> th.Tensor:
        ou_sample = th.tensor(self.oup()).float()
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev*ou_sample
        return th.tanh(self.gaussian_actions)


class MPO_CN(MPO):
    """MPO with colored noise exploration"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, beta, seq_len, rng, observation_space, action_space, seed):
        super().initialize(observation_space, action_space, seed)
        self.seq_len = seq_len
        self.rng = rng
        self.action_space = action_space
        self.set_beta(beta)

    def set_beta(self, beta):
        if np.isscalar(beta):
            beta = [beta] * self.action_space.shape[0]
        self.cn_processes = [
            ColoredNoiseProcess(beta=b, chunksize=self.seq_len, largest_wavelength=None, rng=self.rng) for b in beta]

    def _step(self, observations):
        observations = th.as_tensor(observations, dtype=th.float32)
        cn_sample = th.tensor([[cnp.sample() for cnp in self.cn_processes]]).float()
        with th.no_grad():
            loc = self.model.actor(observations).loc
            scale = self.model.actor(observations).scale
            return loc + scale*cn_sample


class MPO_OU(MPO):
    """MPO with Ornstein Uhlenbeck exploration"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, observation_space, action_space, seed, scale, theta, dt):
        super().initialize(observation_space, action_space, seed)
        action_dim = action_space.shape[0]
        self.oup = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim), scale*np.ones(action_dim), theta, dt)

    def _step(self, observations):
        observations = th.as_tensor(observations, dtype=th.float32)
        ou_sample = th.tensor([self.oup()]).float()
        with th.no_grad():
            loc = self.model.actor(observations).loc
            scale = self.model.actor(observations).scale
            return loc + scale*ou_sample


class MonitorLogger(tonic.logger.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_returns = []
        self.episode_lengths = []
        self.evaluation_returns = []
        self.monitor = {'episode_returns': self.episode_returns, 'episode_lengths': self.episode_lengths,
                        'evaluation_returns': self.evaluation_returns}

    def dump(self):
        self.episode_returns.extend(self.epoch_dict['train/episode_score'])
        self.episode_lengths.extend(self.epoch_dict['train/episode_length'])
        self.evaluation_returns.append(
            (self.epoch_dict['train/steps'][0], np.array(self.epoch_dict['test/episode_score'])))
        super().dump()

    def save(self, dir_):
        with open(dir_ / 'state.pkl', 'wb') as f:
            pickle.dump(self.monitor, f)


class MonitorCallback(BaseCallback):
    """Simple SB3 callback for monitoring episode returns and lengths. Also shows a progress bar."""
    def __init__(self, gen_env, save=False, eval_every=10_000, n_eval=5):
        super().__init__()
        self.save = save
        self.env = gen_env()
        self.eval_every = eval_every    # Every eval_every interactions, commence evaluation step
        self.n_eval = n_eval            # Number of evaluation episodes per evaluation step
        self.episode_returns = []
        self.episode_lengths = []
        self.evaluation_returns = []
        self.monitor = {'episode_returns': self.episode_returns, 'episode_lengths': self.episode_lengths,
                        'evaluation_returns': self.evaluation_returns}

    def _on_training_start(self):
        self.n_actions = self.training_env.action_space.shape[-1]
        self.episode_returns.append(0)
        self.episode_lengths.append(0)

        # Progress bar
        self.pbar = tqdm(total=self.locals['total_timesteps'])

    def _on_training_end(self):
        # Save data and close progress bar
        if self.save:
            with open(self.save / 'state.pkl', 'wb') as f:
                pickle.dump(self.monitor, f)
        self.pbar.close()

    def _on_step(self):
        self.pbar.update()

        # Store reward
        self.episode_returns[-1] += self.locals['rewards'].item()
        self.episode_lengths[-1] += 1
        if self.locals['dones'].item():
            self.logger.record('episode return', self.episode_returns[-1])
            self.logger.record('episode length', self.episode_lengths[-1])
            self.episode_returns.append(0)
            self.episode_lengths.append(0)

        # Evaluation Rollouts
        if self.num_timesteps == 1 or self.num_timesteps % self.eval_every == 0:
            returns = np.zeros(self.n_eval)
            for i in range(self.n_eval):
                obs = self.env.reset()
                done = False
                while not done:
                    action = self.model.predict(obs, deterministic=True)[0]
                    obs, reward, done, _ = self.env.step(action)
                    returns[i] += reward
            self.logger.record('evaluation return mean', returns.mean())
            self.logger.record('evaluation return std', returns.std())
            self.evaluation_returns.append((self.num_timesteps, returns))


class BanditColoredNoise(MonitorCallback):
    """Adjust Colored Noise Using Differential Reward Bandit Algorithm"""
    def __init__(self, gen_env, save=False, method='ind', colors=None, noise_scale=0.3, score='return',
                 len_rollout=100, oracle=0, rng=None, monitor_kwargs=None, bandit_kwargs=None):
        super().__init__(gen_env, save, **(monitor_kwargs if monitor_kwargs else {}))
        self.colors = colors
        self.noise_scale = noise_scale    # Colored noise process std
        self.len_rollout = len_rollout    # Number of interactions per rollout
        self.score = score                # Rollout statistic as bandit reward
        self.rollout = []                 # Rewards from current rollout for score calculation
        self.i = 0                        # Iteration counter
        self.rng = rng                    # Random number generator
        # self.method = method              # Method is either 'bandit' or 'bo' (for Bayesian optimization)
        self.method = method              # Method is one of 'ind', 'list', 'cont'

        # Initialize bandit algorithm
        self.bandit = (
            KMNBandit(self.colors[0], self.colors[-1], **(bandit_kwargs if bandit_kwargs else {})) if method == 'cont'
            else MNBandit(len(self.colors), **(bandit_kwargs if bandit_kwargs else {})))

        self.window = self.bandit.window
        self.returns = np.empty(self.window)    # Past returns to calculate info-score
        self.oracle = oracle    # Number of additional rollouts to get a more accurate score estimate.

        print("Bandit Configuration")
        print(self.bandit, self.bandit.__dict__)

        # Monitoring
        self.monitor['bandit'] = self.bandit.monitor

    def _on_training_start(self):
        super()._on_training_start()

        # Initialize agent noise
        self.k = self.bandit()
        self.beta = self.colors[self.k] if self.method != 'cont' else self.k
        self.set_beta()

    def set_beta(self):
        if isinstance(self.model, SAC):
            # State-dependent action noise is a bit more difficult
            self.model.actor.action_dist = SquashedDiagCNDistribution(
                self.n_actions, self.beta*np.ones(self.n_actions), self.len_rollout, rng=self.rng)
        else:
            # For TD3 or other non-state dependent action noise agents
            self.model.action_noise = ColoredActionNoise(
                self.beta*np.ones(self.n_actions), self.noise_scale*np.ones(self.n_actions),
                self.len_rollout, rng=self.rng)

    def sample_action_oracle(self, obs):
        # Sample action as per OffPolicyAlgorithm._sample_action
        action, _ = self.model.predict(obs, deterministic=False)
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = self.model.policy.scale_action(action)

            if self.model.action_noise is not None:
                action = np.clip(action + self.model.action_noise(), -1, 1)
            action = self.model.policy.unscale_action(action)
        return action

    def score_rollout(self, rollout):
        ret = sum(rollout)

        # Calculate rollout score
        if self.score == 'return':
            score = ret
        elif self.score == 'var-wss':
            # BUG: something is wrong here.. the scores are all negative? (not used in paper)
            R = np.asarray(rollout)
            T = len(R)
            m2 = (R.mean()**2)
            var = np.sum(R**2) - T*m2
            score = var + 2*sum((R[t]*R[t + tau] - (T - tau)*m2) for tau in range(1, T) for t in range(T - tau))
        elif self.score == 'info':
            N = min(self.window, self.i - 1)
            if N == 0:
                score = 1    # = probability of 1/e
            else:
                # KDE, bandwidth from Murphy: PML1, Sec. 16.3.3
                rs = self.returns[:N]
                mad = np.median(abs(rs - np.median(rs)))
                mad = max(mad, 1e-2)
                h = 1.4826 * mad * (4 / (3 * N))**(1/5)
                score = np.log(N * np.sqrt(2 * np.pi) * h) - logsumexp(-(ret - rs)**2/(2 * h**2))

        return score, ret

    def _on_step(self):
        super()._on_step()

        # Store reward
        self.rollout.append(self.locals['rewards'].item())

        # Check if rollout is finished
        i = self.num_timesteps // self.len_rollout
        if i == self.i:
            return
        self.i = i    # i = number of finished rollouts

        # If oracle > 0: Run additional rollouts with same beta
        rollouts = [self.rollout]
        for i in range(self.oracle):
            rollout = []    # rewards
            obs = self.env.reset()
            for j in range(self.len_rollout):
                action = self.sample_action_oracle(obs)
                obs, reward, done, _ = self.env.step(action)
                rollout.append(reward)
                if done:
                    obs = self.env.reset()

        # Score all rollouts (only one rollout if oracle = 0)
        scores, returns = zip(*[self.score_rollout(rollout) for rollout in rollouts])
        score = np.mean(scores)

        # Store (average) return for info-score
        t = (self.i - 1) % self.window
        self.returns[t] = np.mean(returns)

        # Select new beta, change noise process and reset rollout return
        beta = self.beta
        self.k = self.bandit(score)
        self.beta = self.colors[self.k] if self.method != 'cont' else self.k
        if beta != self.beta:
            self.set_beta()
        self.rollout = []

        # Monitoring
        self.logger.record('rollout score', score)
        self.logger.record('color', beta)
        if self.bandit.diff_rs:
            self.logger.record('differential reward', self.bandit.diff_rs[-1])
        if hasattr(self.bandit, 'K') and self.bandit.K > 1:
            for l in range(self.bandit.K):
                self.logger.record(f'mean for beta = {self.colors[l]}', self.bandit.means[-1][l])
                self.logger.record(f'std for beta = {self.colors[l]}', self.bandit.stds[-1][l])


class MNBandit:
    """MLE Normalized Reward Non-Stationary Bayesian Bandit"""
    def __init__(self, K, method='ts', initial=3, window=1000, sigma=1, q_acc=0, dx=None, ls=None, rng=None):
        """method: ts = Thompson sampling, bucb = Bayes-UCB, a number c = UCB with q = mean + c*std"""
        self.K = K                              # Number of arms
        self.method = method                    # Which bandit algorithm to use (ts, bayes-ucb or simple ucb)
        self.initial = initial                  # Initial exploration
        self.window = window                    # Number of rewards to keep (for non-stationarity)
        self.played = np.empty(window, int)     # Selected arms for each iteration
        self.rewards = np.empty(window)         # Bandit rewards (normalized/differential)
        self.ext_rs = np.empty(window)          # Reward buffer to store previous rewards for outlier detection
        self.previous_rs = np.zeros(K)          # Previous external rewards
        self.pr_stds = np.empty(window)         # Standard deviations (over arm) of previous external rewards
        self.i = 0                              # Iteration counter
        self.rng = rng                          # Random number generator
        self.q_acc = q_acc                      # Acceptance quantile for outlier detection in external reward
        self.sigma = sigma                      # Likelihood std
        self.covs = dx and ls                   # ls: Lengthscale of RBF kernel, dx: Distance between two  arms

        # Initialize priors (either with RBF kernel or independent arms)
        self.m0 = 0
        self.s0 = 1
        if self.covs:
            arms = np.linspace(0, self.K*dx, self.K)[:, None]
            self.m0 = np.ones(K)[:, None] * self.m0
            self.C0 = self.s0**2 * np.exp(-cdist(arms, arms)**2 / (2 * ls**2)) + np.eye(len(arms))*1e-4
        else:
            self.p0 = self.s0**-2               # Prior precision
            self.pm0 = self.s0**-2 * self.m0    # Prior precision adjusted mean
            self.p = self.p0 * np.ones(K)       # Precisions
            self.pm = self.pm0 * np.ones(K)     # Precision adjusted means

        # Monitoring
        self.played_all = []    # Played arms
        self.rs = []            # External rewards
        self.diff_rs = []       # Internal (differential/normalized/bandit rewards)
        self.means = []         # Estimations for mean differential rewards per arm
        self.stds = []          # Estimations for std differential rewards per arm
        self.monitor = {
            'played': self.played_all, 'rs': self.rs, 'diff_rs': self.diff_rs, 'means': self.means, 'stds': self.stds
        }

    def __call__(self, r=None):
        if self.K == 1:
            self.rs.append(r)
            self.played_all.append(0)
            return 0

        status = min(self.window, self.i + 1)    # Window-adjusted number of observations
        t = self.i % self.window                 # Current index in window-sized arrays

        if r is not None:
            # We assume our recommendation was followed and store arm and reward
            self.played[t] = self.k
            self.ext_rs[t] = r

            # Calculate scaling and offset parameters and normalize the reward
            rs = self.ext_rs[:status]
            b = rs.mean()
            a = np.sqrt(np.mean((rs - b)**2 / (self.sigma**2 + self.s0**2))) or 1
            dr = (r - b) / a
            self.rewards[t] = dr

            # Do Bayesian update
            if self.covs:
                y = self.rewards[:status, None]
                # mask = ~np.isnan(y).ravel()
                # y = y[mask]
                # A = self.played[:status, None][mask] == np.arange(self.K)
                A = self.played[:status, None] == np.arange(self.K)

                # Fix overconfidence at the beginning
                y_start = np.zeros(self.window - status)[:, None]
                A_start = np.arange(self.window - status)[:, None] % self.K == np.arange(self.K)
                y = np.r_[y_start, y]
                A = np.r_[A_start, A]

                G = A@self.C0@A.T + (np.eye(len(y)) * self.sigma**2)
                gain = solve(G.T, A @ self.C0.T, assume_a='pos').T
                ms = (self.m0 + gain@(y - A@self.m0)).ravel()
                ss = np.sqrt(np.diag(self.C0 - gain@A@self.C0))
            else:
                for l in range(self.K):
                    rs = self.rewards[:status][self.played[:status] == l]
                    # rs = rs[~np.isnan(rs)]
                    self.p[l] = self.p0 + (len(rs) * self.sigma**-2)
                    self.pm[l] = self.pm0 + np.sum(self.sigma**-2 * rs)

                    # Fix overconfidence at the beginning
                    self.p[l] += (self.window - status) * self.sigma**-2

                # Update means and standard deviations
                vs = self.p ** -1
                ms = vs * self.pm
                ss = np.sqrt(vs)

            # Register observation
            self.i += 1

            # Monitoring
            self.played_all.append(self.k)
            self.rs.append(r)
            self.diff_rs.append(self.rewards[t])
            self.means.append(ms)
            self.stds.append(ss)

        # Recommend next arm
        if self.i < self.initial * self.K:
            self.k = self.i % self.K                                       # Play each arm a few times at the start
        elif self.method == 'ts':
            self.k = np.argmax(norm.rvs(ms, ss, random_state=self.rng))    # Thompson sampling
        elif self.method == 'bucb':
            self.k = np.argmax(norm.ppf(1 - 1/status, ms, ss))             # Bayes-UCB
        else:
            self.k = np.argmax(ms + self.method*ss)                        # Simple UCB

        return self.k


class KMNBandit:
    """Kernel MLE Normalizing Thompson Sampling"""
    def __init__(self, a, b, kernel='rbf', method='ts', sigma=1, ls=1, window=1000, N=1000, rng=None):
        """method: ts = Thompson sampling, bucb = Bayes-UCB, a number c = UCB with q = mean + c*std"""
        self.a = a                         # Starting point
        self.b = b                         # End point
        self.method = method               # Which bandit algorithm to use (ts, bayes-ucb or simple ucb)
        self.window = window               # Number of rewards to keep (for non-stationarity)
        self.played = np.empty(window)     # Selected arms for each iteration
        self.rewards = np.empty(window)    # Bandit rewards (normalized/differential)
        self.ext_rs = np.empty(window)     # Reward buffer to store previous rewards for outlier detection
        self.i = 0                         # Iteration counter
        self.rng = rng                     # Random number generator
        self.sigma = sigma                 # Likelihood std
        self.N = N                         # Resolution of interval [a, b]. N = (b - a) / dx

        # Initialize prior mean and covariance functions (ls: Lengthscale of RBF kernel)
        self.s0 = 1
        self.m0 = lambda x: np.zeros_like(x)
        rbf = lambda x, y=None: self.s0**2 * np.exp(-cdist(x, x if y is None else y)**2 / (2 * ls**2))    # noqa: E731
        if kernel == 'rbf':
            self.k0 = rbf
        elif kernel == 'explore':
            phi = lambda x: 1 / (1 + np.exp(-2 * x))        # noqa: E731
            self.k0 = lambda x, y=None: rbf(phi(x), phi(y) if y is not None else None)
        self.m = self.m0
        self.cov = self.k0
        self.x = np.linspace(a, b, N)[:, None]    # Sampling space (for argmax)
        self.x_save = np.linspace(a, b, 100)[:, None]

        # Monitoring
        self.played_all = []    # Played arms
        self.rs = []            # External rewards
        self.diff_rs = []       # Internal (differential/normalized/bandit rewards)
        self.means = [self.m(self.x_save)]         # Estimations for mean differential rewards per arm
        self.stds = [np.diag(self.cov(self.x_save))]          # Estimations for std differential rewards per arm
        self.monitor = {
            'played': self.played_all, 'rs': self.rs, 'diff_rs': self.diff_rs, 'means': self.means, 'stds': self.stds
        }

    def __call__(self, r=None):
        if self.a == self.b:
            self.rs.append(r)
            self.played_all.append(self.a)
            return self.a

        status = min(self.window, self.i + 1)    # Window-adjusted number of observations
        t = self.i % self.window                 # Current index in window-sized arrays

        if r is not None:
            # We assume our recommendation was followed and store arm and reward
            self.played[t] = self.k
            self.ext_rs[t] = r

            # Calculate scaling and offset parameters and normalize reward
            rs = self.ext_rs[:status]
            b = rs.mean()
            a = np.sqrt(np.mean((rs - b)**2 / (self.sigma**2 + self.s0**2))) or 1
            dr = (r - b) / a
            self.rewards[t] = dr

            # - Do Bayesian update -
            rs = self.rewards[:status, None]
            A = self.played[:status, None]

            # Fix overconfidence at the beginning
            y_start = np.zeros(self.window - status)[:, None]
            A_start = np.linspace(self.a, self.b, self.window - status)[:, None]
            rs = np.r_[y_start, rs]
            A = np.r_[A_start, A]

            G = self.k0(A) + (np.eye(len(rs)) * self.sigma**2)
            gain = lambda x: solve(G.T, self.k0(x, A).T, assume_a='pos').T    # noqa: E731
            self.m = lambda x: self.m0(x) + gain(x)@(rs - self.m0(A))
            self.cov = lambda x, y=None: self.k0(x, y) - gain(x)@self.k0(A, x if y is None else y)
            # - End -

            # Register observation
            self.i += 1

            # Monitoring
            self.played_all.append(self.k)
            self.rs.append(r)
            self.diff_rs.append(self.rewards[t])
            self.means.append(self.m(self.x_save))
            self.stds.append(np.diag(self.cov(self.x_save)))

        # Recommend next arm
        if self.method == 'ts':
            # Thompson sampling
            self.k = self.x.ravel()[np.argmax(multivariate_normal.rvs(
                self.m(self.x).ravel(), self.cov(self.x) + np.eye(len(self.x))*1e-6, random_state=self.rng))]
        else:
            raise NotImplementedError("Only Thompson sampling is implemented")

        return self.k


class OracleColoredNoise(MonitorCallback):
    """Colored noise with beta (color coefficient) chosen by oracle (=testing each beta and taking best).
    Previously called ICN (Iterated Best Colored Noise). Very inefficient.
    """
    def __init__(self, gen_env, save=False, colors=None, noise_scale=0.3, len_rollout=10_000, len_test=10_000,
                 use_max=False, q_acc=0, rng=None, **monitor_kwargs):
        super().__init__(gen_env, save, **monitor_kwargs)
        self.colors = colors
        self.noise_scale = noise_scale    # Colored noise process std

        # Number of interactions per (training) rollout: steps between testing
        if isinstance(len_rollout, str) and len_rollout.startswith('ep_'):
            self.len_rollout = self.env._max_episode_steps * int(len_rollout[3:])
        else:
            self.len_rollout = len_rollout

        # Number of interactions for each color in each testing iteration
        if isinstance(len_test, str) and len_test.startswith('ep_'):
            self.len_test = self.env._max_episode_steps * int(len_test[3:])
        else:
            self.len_test = len_test

        self.i = 0                        # Iteration counter
        self.use_max = use_max            # Use max instead of mean of test episode returns for beta selection
        self.q_acc = q_acc                # Only accept returns which lie above acceptance quantile (default 0 = all)
        self.rng = rng                    # Random number generator

        # Monitoring
        self.color_scores = []
        self.monitor['color_scores'] = self.color_scores

    def _on_training_start(self):
        super()._on_training_start()

        # Initialize agent noise
        self.beta = self.update_beta()

    def set_beta(self, beta, seq_len):
        if isinstance(self.model, SAC):
            # State-dependent action noise is a bit more difficult
            self.model.actor.action_dist = SquashedDiagCNDistribution(
                self.n_actions, beta*np.ones(self.n_actions), seq_len, rng=self.rng)
        else:
            # For TD3 or other non-state dependent action noise agents
            self.model.action_noise = ColoredActionNoise(
                beta*np.ones(self.n_actions), self.noise_scale*np.ones(self.n_actions),
                seq_len, rng=self.rng)

    def sample_action(self, obs):
        # Sample action as per OffPolicyAlgorithm._sample_action
        action, _ = self.model.predict(obs, deterministic=False)
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = self.model.policy.scale_action(action)

            if self.model.action_noise is not None:
                action = np.clip(action + self.model.action_noise(), -1, 1)
            action = self.model.policy.unscale_action(action)
        return action

    def update_beta(self):
        """Run test rollouts and return and apply best-performing beta"""
        scores = np.zeros_like(self.colors)
        for j, beta in enumerate(self.colors):
            self.set_beta(beta, self.len_test)

            # Run rollouts
            returns = []
            cur_ret = 0
            obs = self.env.reset()
            for _ in range(self.len_test):
                action = self.sample_action(obs)
                obs, reward, done, _ = self.env.step(action)
                cur_ret += reward
                if done:
                    returns.append(cur_ret)
                    cur_ret = 0
                    obs = self.env.reset()

            # Score rollouts with current beta
            if self.use_max:
                scores[j] = max(returns)
            elif self.q_acc:
                rq = np.quantile(returns, self.q_acc)
                scores[j] = np.mean([r for r in returns if r >= rq])
            else:
                scores[j] = np.mean(returns)

        # Determine and apply best beta
        best_beta = self.colors[np.argmax(scores)]
        self.set_beta(best_beta, self.len_rollout)

        # Monitoring
        self.color_scores.append(scores)
        self.logger.record('best color', best_beta)

        return best_beta

    def _on_step(self):
        super()._on_step()

        # Check if rollout is finished
        i = self.num_timesteps // self.len_rollout
        if i == self.i:
            return
        self.i = i

        # Select new beta and change noise process
        self.beta = self.update_beta()


class RandomColoredNoise(MonitorCallback):
    def __init__(self, gen_env, save=False, colors=None, noise_scale=0.3, len_rollout=10_000,
                 method='explore3', rng=None, **monitor_kwargs):
        super().__init__(gen_env, save, **monitor_kwargs)
        self.colors = colors
        self.noise_scale = noise_scale    # Colored noise process std
        self.method = method              # 'list', 'uniform' or 'explore'
        self.len_rollout = len_rollout    # Number of interactions per rollout
        self.i = 0                        # Iteration counter
        self.rng = rng                    # Random number generator

        # Monitoring
        self.chosen_colors = []
        self.sigmas = []
        self.monitor['colors'] = self.chosen_colors
        self.monitor['sigmas'] = self.sigmas

    def _on_training_start(self):
        super()._on_training_start()

        # Initialize agent noise
        self.beta = self.update_beta()

    def set_beta(self, beta, sigma, seq_len):
        if isinstance(self.model, SAC):
            # State-dependent action noise is a bit more difficult
            self.model.actor.action_dist = SquashedDiagCNDistribution(
                self.n_actions, beta, seq_len, rng=self.rng)
        else:
            # For TD3 or other non-state dependent action noise agents
            self.model.action_noise = ColoredActionNoise(
                beta, sigma, seq_len, rng=self.rng)

    def update_beta(self):
        """Select new beta parameter (randomly)"""
        # Sample and apply beta
        if self.method == 'list':
            beta = self.rng.choice(self.colors)*np.ones(self.n_actions)
            sigma = self.noise_scale*np.ones(self.n_actions)
        if self.method == 'list2':    # independent beta
            beta = self.rng.choice(self.colors, size=self.n_actions)
            sigma = self.noise_scale*np.ones(self.n_actions)
        if self.method == 'list3':    # independent beta,sigma
            beta = self.rng.choice(self.colors, size=self.n_actions)
            sigma = self.rng.choice(self.noise_scale, size=self.n_actions)
        elif self.method == 'uniform':
            beta = self.rng.uniform(self.colors[0], self.colors[-1])*np.ones(self.n_actions)
            sigma = self.noise_scale*np.ones(self.n_actions)
        elif self.method == 'explore':
            u = self.rng.uniform()*np.ones(self.n_actions)
            beta = np.clip(np.arctanh(u), self.colors[0], self.colors[-1])
            sigma = self.noise_scale*np.ones(self.n_actions)
        elif self.method == 'cont':     # explore2
            u = self.rng.uniform()*np.ones(self.n_actions)
            beta = np.arctanh(u)
            sigma = self.noise_scale*np.ones(self.n_actions)
        elif self.method == 'cont2':    # explore3
            u = self.rng.uniform(size=self.n_actions)
            beta = np.arctanh(u)
            sigma = self.noise_scale*np.ones(self.n_actions)
        elif self.method == 'explore4':
            # beta like explore3, sigma similarly from exponential distribution
            u = self.rng.uniform(size=self.n_actions)
            beta = np.arctanh(u)
            sigma = -np.log10(self.rng.uniform(size=self.n_actions))
        self.set_beta(beta, sigma, self.len_rollout)

        # Monitoring
        self.logger.record('sampled color', beta)
        self.chosen_colors.append(beta)
        self.sigmas.append(sigma)

        return beta

    def _on_step(self):
        super()._on_step()

        # Check if rollout is finished
        i = self.num_timesteps // self.len_rollout
        if i == self.i:
            return
        self.i = i

        # Select new beta and change noise process
        self.beta = self.update_beta()


class ConstantColoredNoise(MonitorCallback):
    def __init__(self, gen_env, save=False, beta=None, noise_scale=0.3,
                 len_rollout=10_000, rng=None, **monitor_kwargs):
        super().__init__(gen_env, save, **monitor_kwargs)
        self.beta = beta
        self.noise_scale = noise_scale    # Colored noise process std
        self.len_rollout = len_rollout    # Number of interactions per rollout
        self.rng = rng                    # Random number generator

    def _on_training_start(self):
        super()._on_training_start()

        # Initialize agent noise
        beta = self.beta * np.ones(self.n_actions)
        sigma = self.noise_scale * np.ones(self.n_actions)
        self.set_beta(beta, sigma, self.len_rollout)

    def set_beta(self, beta, sigma, seq_len):
        if isinstance(self.model, SAC):
            # State-dependent action noise is a bit more difficult
            self.model.actor.action_dist = SquashedDiagCNDistribution(
                self.n_actions, beta, seq_len, rng=self.rng)
        else:
            # For TD3 or other non-state dependent action noise agents
            self.model.action_noise = ColoredActionNoise(
                beta, sigma, seq_len, rng=self.rng)


class ScheduledColoredNoise(MonitorCallback):
    """Linear scheduling from colors[0] to colors[1]"""
    def __init__(self, gen_env, save=False, colors=(2, 0), method='linear', noise_scale=0.3, len_rollout=10_000,
                 rng=None, **monitor_kwargs):
        super().__init__(gen_env, save, **monitor_kwargs)
        self.colors = colors
        self.method = method              # 'linear' or 'atanh' scheduling
        self.noise_scale = noise_scale    # Colored noise process std
        self.len_rollout = len_rollout    # Number of interactions per rollout
        self.i = 0                        # Iteration counter
        self.rng = rng                    # Random number generator

        # Monitoring
        self.chosen_colors = []
        self.monitor['colors'] = self.chosen_colors

    def _on_training_start(self):
        super()._on_training_start()

        # Initialize agent noise
        self.beta = self.update_beta()

    def set_beta(self, beta, seq_len):
        if isinstance(self.model, SAC):
            # State-dependent action noise is a bit more difficult
            self.model.actor.action_dist = SquashedDiagCNDistribution(
                self.n_actions, beta, seq_len, rng=self.rng)
        else:
            # For TD3 or other non-state dependent action noise agents
            self.model.action_noise = ColoredActionNoise(
                beta, self.noise_scale * np.ones(self.n_actions), seq_len, rng=self.rng)

    def update_beta(self):
        """Select new beta parameter (randomly)"""
        # Calculate and apply beta
        x = self.num_timesteps / self.locals['total_timesteps']
        if self.method == 'linear':
            beta = (1 - x)*self.colors[0] + x*self.colors[1]
            beta *= np.ones(self.n_actions)
        elif self.method == 'atanh':
            beta = np.clip(np.arctanh(min(1 - x, 1 - 1e-6)), 0, 4)
            beta *= np.ones(self.n_actions)
        self.set_beta(beta, self.len_rollout)

        # Monitoring
        self.logger.record('scheduler color', beta)
        self.chosen_colors.append(beta)

        return beta

    def _on_step(self):
        super()._on_step()

        # Check if rollout is finished
        i = self.num_timesteps // self.len_rollout
        if i == self.i:
            return
        self.i = i

        # Select new beta and change noise process
        self.beta = self.update_beta()


class CNTrainer(Trainer):
    def hook(self, ret):
        """Called when an episode is finished"""
        raise NotImplementedError()

    def run(self):
        '''Main loop copied from tonic.Trainer, added hook calling.'''

        start_time = last_epoch_time = time.time()

        # Start the environments.
        observations = self.environment.start()

        num_workers = len(observations)
        scores = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0

        while True:
            # Select actions.
            actions = self.agent.step(observations, self.steps)
            assert not np.isnan(actions.sum())
            logger.store('train/action', actions, stats=True)

            # Take a step in the environments.
            observations, infos = self.environment.step(actions)
            self.agent.update(**infos, steps=self.steps)

            scores += infos['rewards']
            lengths += 1
            self.steps += num_workers
            epoch_steps += num_workers
            steps_since_save += num_workers

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(
                    self.steps, self.epoch_steps, self.max_steps)

            # Check the finished episodes.
            for i in range(num_workers):
                if infos['resets'][i]:
                    logger.store('train/episode_score', scores[i], stats=True)
                    logger.store(
                        'train/episode_length', lengths[i], stats=True)

                    self.hook(ret=scores[i])

                    scores[i] = 0
                    lengths[i] = 0
                    episodes += 1

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment:
                    self._test()

                # Log the data.
                epochs += 1
                current_time = time.time()
                epoch_time = current_time - last_epoch_time
                sps = epoch_steps / epoch_time
                logger.store('train/episodes', episodes)
                logger.store('train/epochs', epochs)
                logger.store('train/seconds', current_time - start_time)
                logger.store('train/epoch_seconds', epoch_time)
                logger.store('train/epoch_steps', epoch_steps)
                logger.store('train/steps', self.steps)
                logger.store('train/worker_steps', self.steps // num_workers)
                logger.store('train/steps_per_second', sps)
                logger.dump()
                last_epoch_time = time.time()
                epoch_steps = 0

            # End of training.
            stop_training = self.steps >= self.max_steps

            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(logger.get_path(), 'checkpoints')
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith('step_'):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f'step_{self.steps}'
                save_path = os.path.join(path, checkpoint_name)
                self.agent.save(save_path)
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                break


class ScheduledCNTrainer(CNTrainer):
    def __init__(self, method, trainer_kwargs):
        super().__init__(**trainer_kwargs)
        self.method = method

    def hook(self, ret):
        # Update CN processes
        x = self.steps / self.max_steps

        if self.method == 'linear':
            beta = (1 - x)*2 + x*0
        elif self.method == 'atanh':
            beta = np.clip(np.arctanh(min(1 - x, 1 - 1e-6)), 0, 4)

        self.agent.set_beta(beta)


class RandomCNTrainer(CNTrainer):
    def __init__(self, method, colors, rng, trainer_kwargs):
        super().__init__(**trainer_kwargs)
        self.method = method
        self.colors = colors
        self.rng = rng

    def hook(self, ret):
        """Select new beta parameter (randomly)"""
        ndims = self.agent.action_space.shape[0]

        if self.method == 'list':       # all from list
            beta = self.rng.choice(self.colors)*np.ones(ndims)
        if self.method == 'list2':      # independent dimensions from list
            beta = self.rng.choice(self.colors, size=ndims)
        elif self.method == 'cont':     # explore2; all from continuous distribution
            u = self.rng.uniform()*np.ones(ndims)
            beta = np.arctanh(u)
        elif self.method == 'cont2':    # explore3; independent dimensions from continuous distribution
            u = self.rng.uniform(size=ndims)
            beta = np.arctanh(u)

        self.agent.set_beta(beta)


class BanditCNTrainer(CNTrainer):
    def __init__(self, method, colors, score, bandit_kwargs, logger, trainer_kwargs):
        super().__init__(**trainer_kwargs)
        self.method = method
        self.colors = colors
        self.score = score
        self.i = 0

        self.bandit = (
            KMNBandit(self.colors[0], self.colors[-1], **bandit_kwargs) if method == 'cont'
            else MNBandit(len(self.colors), **bandit_kwargs))

        self.window = self.bandit.window
        self.returns = np.empty(self.window)    # Past returns to calculate info-score

        logger.monitor['bandit'] = self.bandit.monitor    # (fingers crossed...)

        print("Bandit Configuration (MPO)")
        print(self.bandit, self.bandit.__dict__)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        self.k = self.bandit()
        self.beta = self.colors[self.k] if self.method != 'cont' else self.k
        self.agent.set_beta(self.beta)

    def score_rollout(self, ret):
        # Calculate rollout score
        if self.score == 'return':
            score = ret
        elif self.score == 'info':
            N = min(self.window, self.i - 1)
            if N == 0:
                score = 1    # = probability of 1/e
            else:
                # KDE, bandwidth from Murphy: PML1, Sec. 16.3.3
                rs = self.returns[:N]
                mad = np.median(abs(rs - np.median(rs)))
                mad = max(mad, 1e-2)
                h = 1.4826 * mad * (4 / (3 * N))**(1/5)
                score = np.log(N * np.sqrt(2 * np.pi) * h) - logsumexp(-(ret - rs)**2/(2 * h**2))

        return score

    def hook(self, ret):
        """Select new beta with bandit"""
        self.i += 1
        score = self.score_rollout(ret)

        # Store (average) return for info-score
        t = (self.i - 1) % self.window
        self.returns[t] = ret

        # Select new beta and change noise process
        self.k = self.bandit(score)
        self.beta = self.colors[self.k] if self.method != 'cont' else self.k
        self.agent.set_beta(self.beta)
