import numpy as np
import gym
from gym import spaces

class CybersecScenario(gym.Env):
    def __init__(self, n_agents=2, n_adversaries=1, n_edges=5,
                 n_vertices=4, n_bins=3, eta=1.0, 
                 adversary_types=[], poss_adv_types=['w', 'p'], graph=None,
                 p_obs=0.8, k=1, def_budget=5):
        self.re_init(
            n_agents, n_adversaries, n_edges, n_vertices, n_bins, eta,
            adversary_types, poss_adv_types, graph, p_obs, k, def_budget
        )

    def re_init(self, n_agents, n_adversaries, n_edges, n_vertices, n_bins, eta,
                adversary_types, poss_adv_types, graph, p_obs, k, def_budget):
        self.num_agents = n_agents
        self.num_adversaries = n_adversaries
        self.num_edges = n_edges
        self.num_vertices = n_vertices
        self.poss_adv_types = poss_adv_types
        self.eta = eta
        self.p_obs = p_obs
        self.k = k
        self.n = n_agents
        self.def_budget = def_budget
        self.remaining_budget = def_budget

        # we can either pass in the adv probs or generate randomly 
        if not adversary_types:
            self.adversary_types = np.random.choice(
                np.asarray(poss_adv_types),
                size=n_adversaries,
                p=np.asarray([eta, 1-eta])
            )
        else:
            self.adversary_types = adversary_types
        self.num_bins = n_bins # number of bins for discretizing target values
        self.action_space = self.init_action_space()
        self.observation_space = self.init_obs_space()
        self.obs_shapes = [
            self.observation_space[i].shape for i in range(self.num_agents)
        ]
        self.graph, self.vals = self.init_graph(graph)
        import time
        print(self.graph)
        print(self.adversary_types)
        time.sleep(10)
        self.entry_pts = self.get_entry_points()
        self.reset()
        # TODO: save the graph for when we do evaluation 

    def init_graph(self, graph):
        '''
        initialize attack graph as a dictionary 
        {
            'vertex' : [
                           [node_idx_j, success_rate_j, edge_idx_j, node_val_j],
                           [node_idx_k, success_rate_k, edge_idx_k, node_val_k]
                       ]
        }
        '''
        threshold = 0.2
        vals = [0 for _ in range(self.num_vertices)]
        if not graph:
            g = {v : [] for v in range(self.num_vertices)}
            added = 0
            while added < self.num_edges:
                for i in range(self.num_vertices):
                    for j in range(self.num_vertices):
                        # ? `p` used to control the density of edges
                        p = np.random.random()
                        if p < threshold:
                            t_prob = np.random.random()
                            bin_n = np.random.choice(
                                np.arange(self.num_bins)
                            )
                            
                            # ? Unidirectional guarantee
                            already_added = False
                            for v in g[i]:
                                if v[0] == j:
                                    already_added = True
                                    
                            if not already_added:
                                # ? Update graph adjacent list:
                                # ? [node_index, success_rate, edge_index, node_value]
                                g[i].append([j, t_prob, added, bin_n])
                                # ? Update graph node values
                                vals[j] = bin_n
                                added += 1
                    if added >= self.num_edges:
                        return g, vals
            return g, vals
        else:
            return graph, vals

    def get_entry_points(self):
        entry_pts = [False for _ in range(self.num_vertices)]
        for v, info in self.graph.items():
            for w in info:
                entry_pts[w[0]] = True
        return np.argwhere(np.array(entry_pts) == True).flatten()

    def init_obs_space(self):
        observation_space_n = []
        for i in range(self.num_agents):
            # attacker loc in vertices, bins of targets
            lb = np.zeros(self.num_vertices * 2)
            ub = np.concatenate((
                np.ones(self.num_vertices), np.full(self.num_vertices, self.num_bins)
            ))
            # if defender, also see set values of targets and remaining budget
            # attacker loc, bins of targets, set values of targets, budget
            #    num_v,         num_v,            num_v,              1
            #if i < self.num_adversaries:
            #    lb = np.concatenate((
            #        lb, np.zeros(self.num_vertices * self.history_len)
            #    ))
            #    ub = np.concatenate((
            #        ub, self.full(num_vertices, self.num_bins)
            #    ))
            # if defender, also see set values of targets
            # attacker loc, bins of targets, set values of targets 
            #else:
            if i >= self.num_adversaries:
                lb = np.concatenate((
                    lb, np.zeros(self.num_vertices) # + 1)
                ))
                ub = np.concatenate((
                    ub, np.full(self.num_vertices, self.num_bins)
                ))
                #ub = np.concatenate((
                #    ub, np.full(1, self.def_budget)
                #))
            observation_space_n.append(spaces.Box(lb, ub))
        return observation_space_n

    def init_action_space(self):
        action_space_n = []
        for i in range(self.num_agents):
            # if adversary
            if i < self.num_adversaries:
                action_space_n.append(spaces.Discrete(self.num_edges + 1))

            # if defender
            else:
                #target_acts = [self.num_bins for _ in range(self.num_vertices)]
                #edge_acts = [self.num_edges]
                # choose one vertex and value (or noop),
                # and one edge at each timestep (or noop)
                target_acts = self.num_bins * self.num_vertices + 1
                edge_acts = self.num_edges + 1
                total_acts = target_acts * edge_acts
                action_space_n.append(spaces.Discrete(total_acts))
                
        return action_space_n
            
    def get_adversary_action_outcome(self, i, action): 
        # get current loc
        states = self._get_obs()
        current_loc = np.argwhere(states[i][:self.num_vertices] == 1).flatten()[0]
        new_loc = np.zeros(self.num_vertices)
        loc = current_loc
        reward, done = 0, False
        #print(np.array(action).shape)
        #print('action is', action)
        action_edge = np.argmax(action)
        print('edge is', action_edge)
        # if the attacker wants to leave the system
        if action_edge > self.num_edges:
            new_state = np.concatenate((
                np.array(new_loc), np.zeros(np.array(self.vals).shape)))
            return new_state, reward, True, {}
        
        # check if valid action and stochastically transition to next state 
        for j, poss in enumerate(self.graph[current_loc]):
            if action_edge == poss[2]:
                if np.random.random() < self.graph[current_loc][j][1]:
                    new_vertex = poss[0]
                    new_loc[new_vertex] = 1
                    loc = new_vertex
                    reward = self.graph[current_loc][j][3]
                    done = False
            
        # TODO: instead of staying at same node, transition to next node randomly 
        # if agent stays at the same node
        if loc == current_loc:
            #new_loc[current_loc] = 1.
            # uniformly pick the next state 
            next_location = np.random.choice(len(self.graph[current_loc]))
            #for j, poss in enumerate(self.graph[current_loc]):
            new_vertex = self.graph[current_loc][next_location][0]
            new_loc[new_vertex] = 1.
            loc = new_vertex
            reward = self.graph[current_loc][next_location][3]
            done = False
        
        # depending on adversary, update the obs
        if self.adversary_types[i] == 'w':
            neighbors = self.graph[loc]
            perceived_target_vals = states[-1][
                self.num_vertices+len(self.vals):self.num_vertices+len(self.vals)*2]
            v_obs = [
                n if j in neighbors else 0 for j, n in enumerate(perceived_target_vals)
            ]
        else:
            v_obs = self.vals

        new_state = np.concatenate((np.array(new_loc), np.array(v_obs))).flatten()
        return new_state, reward, done, {}

    def get_defender_action_mapping(self, action):
        # decompose discrete actions into:
        num_vertex_actions = self.num_vertices * self.num_bins + 1
        num_edge_actions = self.num_edges + 1
        edge_action = np.argmax(action) // num_vertex_actions
        vertex_action = np.argmax(action) % num_vertex_actions
        return (vertex_action, edge_action) 

    def get_target_mapping(self, target_act):
        target_num = target_act // self.num_bins
        target_bin = target_act // self.num_vertices
        return target_num, target_bin
    
    def get_defender_action_outcome(self, i, action):
        states = self._get_obs()
        reward, done = 0, False
        #print('def i is', i)
        # get action mapping from chosen action
        target_act, edge_act = self.get_defender_action_mapping(action)
        # get the previous target values
        #print('states are', states)
        #print('len states', len(states))
        prev_perceived_target_vals = states[i][self.num_vertices*2:self.num_vertices*3]

        # TODO: double check if should be <=
        if target_act < self.num_vertices * self.num_bins: 
            target_vertex, target_bin = self.get_target_mapping(target_act)
            states[i][self.num_vertices*2:self.num_vertices*3][target_vertex] = target_bin
        # get reward based on where adversaries have moved to
        for j, s in enumerate(states):
            if j < self.num_adversaries:
                adv_loc = np.argwhere(s[:self.num_vertices] == 1).flatten()[0]
                reward -= self.vals[adv_loc]
        return states[i], reward, done, {}

    def _set_acts(self, action_n):
        self.acts = []
        for act in action_n:
            if isinstance(act, np.ndarray):
                act = act.tolist()
            self.acts.append(act)

    def step(self, action_n):
        action_n = action_n[0]
        #print('action_n is', action_n)
        self._set_acts(action_n)
        done_n = [False for _ in range(self.num_agents)]
        reward_n = [0 for _ in range(self.num_agents)]
        new_s = []
        a_act = [-1 for _ in range(self.num_adversaries)]
        
        # get each agents action
        for i, action in enumerate(action_n):
            # if adversary
            if i < self.num_adversaries:
                a_act[i] = np.argwhere(
                    self._get_obs()[i][:self.num_vertices] == 1).flatten()[0]
                new_state, reward, done, info = self.get_adversary_action_outcome(
                    i, action)
            # if defender
            else:
                if isinstance(action, str):
                    action = ast.literal_eval(action)
                new_state, reward, done, info = self.get_defender_action_outcome(
                        i, action)
                def_edge_act = self.get_defender_action_mapping(action)[0]
                # if the defender does not choose noop
                if def_edge_act < self.num_edges: 
                    # check if the adversary is caught
                    for j, act in enumerate(a_act):
                        if def_edge_act == act:
                            done_n[j] = True
                            reward_n[j] = -10  # penalty for being caught
                            reward_n[i] += 5   # reward for catching attacker
                            reward_n[i] += self.vals[act] # offset prev penalty 
            new_s.append(new_state)
            reward_n[i] = reward
            done_n[i] = done

        # the defender may be able to observe the attacker
        for i, state in enumerate(new_s):
            if i < self.num_adversaries:
                if np.random.random() <= self.p_obs:
                    new_s[-1][:self.num_vertices] = state[:self.num_vertices]

        self.states = new_s
        done_n = np.array(done_n, dtype=bool)
        done_n = done_n.reshape(1, done_n.shape[0])
        return np.array([self._get_obs()]), np.array([reward_n]), done_n, {}

    def _get_obs(self):
        return [np.array(s) for s in self.states]

    def reset(self):
        self.acts = None
        states = []
        for i in range(self.num_agents):
            states.append(np.zeros(self.obs_shapes[i]))
            # if adversary
            if i < self.num_adversaries:
                loc = np.random.choice(np.array(self.entry_pts))
                states[i][loc] = 1.
            # if defender
            else:
                states[i][self.num_vertices:self.num_vertices+len(self.vals)] = self.vals
                states[i][self.num_vertices+len(self.vals):self.num_vertices*3] = self.vals # perceived values of target
        self.states = states
        return self._get_obs()

    # TODO: reset render, render 
            
