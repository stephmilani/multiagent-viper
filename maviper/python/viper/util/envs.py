import gym

def get_input_shape(env):
    return env.observation_space.shape

def get_input_shape(env, i):
    try:
        return env.env.observation_space[i].shape
        #return env.get_obs_space()
    except AttributeError:
        print("Not the correct obs space.")
    else:
        print('env shape is', env.env.observation_space[i].shape)
        return env.get_obs_shape() #observation_space[i].shape
    
def get_actions(env, i):
    try:
        return env.get_action_shape()
    except AttributeError:
        print("Not the correct action space.")
    else:
        return env.action_space[i].n
