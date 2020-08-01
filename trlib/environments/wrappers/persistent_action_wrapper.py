from gym.wrappers import TimeLimit

class PersistentActionWrapper(TimeLimit):
    def __init__(self, env, persistence=1, gamma=1, max_episode_steps=None):

        super(PersistentActionWrapper, self).__init__(env, max_episode_steps)
        self.persistence = max(1, persistence)
        self.gamma = gamma ** persistence
        self.gamma_single_step = gamma
        self._max_episode_single_steps = max_episode_steps
        self.env._max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps // self.persistence + (max_episode_steps % self.persistence > 0)

    def step(self, action):
        cum_rew = 0.

        remaining_single_steps = self._max_episode_single_steps - self._elapsed_single_steps
        for i in range(min(self.persistence, remaining_single_steps)):
            state, reward, done, info = self.env.step(action)
            cum_rew += reward * self.gamma_single_step ** i
            self._elapsed_single_steps += 1

            if done:
                break

        # done is set to True by TimeLimit when the episode terminate due to the end of the horizon.
        # We want done being set to True only when the environment set done to True.
        if done and self._elapsed_single_steps == self._max_episode_single_steps:
            done = False

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['PersistentActionWrapper.truncated'] = not done

        return state, cum_rew, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self._elapsed_single_steps = 0
        return self.env.reset(**kwargs)

    def render(self, mode='human'):
        remaining_single_steps = self._max_episode_single_steps - self._elapsed_single_steps
        for i in range(min(self.persistence, remaining_single_steps)):
            super(PersistentActionWrapper, self).render(mode)
