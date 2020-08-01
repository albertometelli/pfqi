from gym.wrappers import TimeLimit

class PersistentActionWrapperSharedSamples(TimeLimit):
    def __init__(self, env, sampling_persistence=1, gamma=1, max_episode_steps=100):

        super(PersistentActionWrapperSharedSamples, self).__init__(env, max_episode_steps)
        self.sampling_persistence = max(1, sampling_persistence)
        self.gamma = gamma ** sampling_persistence
        self._max_episode_steps = max_episode_steps
        self.env._max_episode_steps = max_episode_steps

    def step(self, action):

        new_action = False
        if (self._elapsed_persistence_steps == 0):
            self._previous_action = action
            actual_action = action
            new_action = True
        else:
            actual_action = self._previous_action

        self._elapsed_persistence_steps += 1
        if (self._elapsed_persistence_steps >= self.sampling_persistence):
            self._elapsed_persistence_steps = 0

        state, reward, done, info = self.env.step(actual_action)
        self._elapsed_steps += 1

        # done is set to True by TimeLimit when the episode terminate due to the end of the horizon.
        # We want done being set to True only when the environment set done to True.
        if done and self._elapsed_steps == self._max_episode_steps:
            done = False

        info['new_action'] = new_action
        return state, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self._elapsed_persistence_steps = 0
        return self.env.reset(**kwargs)

    def render(self, mode='human'):
        remaining_single_steps = self._max_episode_single_steps - self._elapsed_single_steps
        for i in range(min(self.sampling_persistence, remaining_single_steps)):
            super(PersistentActionWrapperSharedSamples, self).render(mode)
