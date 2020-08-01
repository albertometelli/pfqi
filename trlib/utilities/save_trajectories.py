from trlib.utilities.interaction import split_data
import json


def save_trajectories(dataset, state_dim, action_dim):

    persistences_list = []

    for i in range(len(dataset)):
        t, s, _, _, _, _, _ = split_data(dataset[i], state_dim, action_dim)
        episode_list = []
        current_episode = []
        idx = 0
        for timestep in t:
            if timestep == 0 and current_episode != []:
                episode_list.append(current_episode)
                current_episode = []
            current_episode.append(s[idx].tolist())
            idx += 1
        if current_episode != []:
            episode_list.append(current_episode)
        persistences_list.append(episode_list)

    with open('Trajectories.json', 'w') as f:
        f.write(json.dumps(persistences_list))