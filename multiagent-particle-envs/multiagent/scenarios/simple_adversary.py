import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = num_agents - 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            return adv_rew

    def get_diff(self, d, t):
        if d < -t:
            return [1., 0.]
        elif d > t:
            return [0., 1.]
        else:
            return [0., 0.]

    def observation(self, agent, world):
        #obs_type = 'ignore_enemies' #'ignore_agents' #'bin'#'ignore_agents' #'bin' #'ignore_agents' #'ignore_adv'
        obs_type = 'other' #'bin'
        t = 0.08 
        # get positions of all entities in this agent's reference frame
        entity_pos, entity_inds = [], []
        for entity in world.landmarks:
            diff = entity.state.p_pos - agent.state.p_pos 
            entity_pos.append(diff)
            for d in diff:
                if d < -t:
                    entity_inds.extend([1., 0.])
                elif d > t:
                    entity_inds.extend([0., 1.])
                else:
                    entity_inds.extend([0., 0.])
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos, other_inds = [], []
        for other in world.agents:
            if other is agent: continue
            diff = other.state.p_pos - agent.state.p_pos
            other_pos.append(diff)
            for d in diff:
                if d < -t:
                    other_inds.extend([1., 0.])
                elif d > t:
                    other_inds.extend([0., 1.])
                else:
                    other_inds.extend([0., 0.])
        # goals of good agent
        agent_goal_inds = []
        if not agent.adversary:
            goal_diff = agent.goal_a.state.p_pos - agent.state.p_pos
            for g_d in goal_diff:
                if d < -t:
                    agent_goal_inds.extend([1., 0.])
                elif d > t:
                    agent_goal_inds.extend([0., 1.])
                else:
                    agent_goal_inds.extend([0., 0.])
        # construct observation based on agent, observation type
        # debugging obs space: to test if continuous representation is challenging
        # for constructing dt policies and to introduce manually-specified experts
        if obs_type == 'bin':
            if not agent.adversary:
                ob = np.concatenate(
                    (np.asarray(agent_goal_inds), np.concatenate(
                        (np.asarray(entity_inds), np.asarray(other_inds))
                    )))
            else:
                ob = np.concatenate(
                    (np.asarray(entity_inds), np.asarray(other_inds)))
        # debugging obs space: to test if there is causal confusion for defs
        # when they can see the adversary location 
        elif obs_type == 'ignore_adv':
            num_entries = 4
            if not agent.adversary:
                # TODO: confirm sequencing -- check agent.adv when storing info
                other_agents_ob = other_inds[-num_entries:]
                
                #print(other_agents_ob)
                ob = np.concatenate(
                    (np.asarray(agent_goal_inds), np.concatenate(
                        (np.asarray(entity_inds), np.asarray(other_agents_ob))
                     )))
            else:
                ob = np.concatenate(
                    (np.asarray(entity_inds), np.asarray(other_inds)))
        elif obs_type == 'bin_more_info':
            other_agent_ob = []
            for o_i, other_agent in enumerate(world.agents):
                if other_agent == agent: continue 
                diff_to_agent = other_agent.state.p_pos - agent.state.p_pos
                # get the distances to other agents, targets 
                for d in diff_to_agent:
                    other_agent_ob.extend(self.get_diff(d, t))
                for t, target in enumerate(world.landmarks):
                    diff_to_target = other_agent.state.p_pos - target.state.p_pos
                    for d in diff_to_target:
                        other_agent_ob.extend(self.get_diff(d, t))
            if not agent.adversary:
                ob = np.concatenate(
                    (np.asarray(agent_goal_inds), np.concatenate(
                        (np.asarray(entity_inds), np.asarray(other_agent_ob))
                    )))
            else:
                ob = np.concatenate(
                    (np.asarray(entity_inds), np.asarray(other_agent_ob)))
                
                        
        # debugging obs space: to test if ignoring enemies changes the policies
        # agents only get to see each other and the targets
        # adversary only gets to see the targets 
        elif obs_type == 'ignore_enemies':
            num_entries = 4
            # TODO: may also want to include the directions of the other agents
            #       to the targets -- this seems like an important thing that
            #       is missing in the representation
            #       make this change and do a new bin -- more info
            if not agent.adversary:
                other_agent_ob = []
                for o_i, other_agent in enumerate(world.agents):
                    if other_agent is agent or other_agent.adversary: continue 
                    diff = other_agent.state.p_pos - agent.state.p_pos
                    for d in diff:
                        if d < -t:
                            other_agent_ob.extend([1., 0.])
                        elif d > t:
                            other_agent_ob.extend([0., 1.])
                        else:
                            other_agent_ob.extend([0., 0.])
                ob = np.concatenate(
                    (np.asarray(agent_goal_inds), np.concatenate(
                        (np.asarray(entity_inds), np.asarray(other_agent_ob))
                    )))
            else:
                ob = np.asarray(entity_inds)
        elif obs_type == 'ignore_agents':
            if not agent.adversary:
                ob = np.concatenate(
                    (np.asarray(agent_goal_inds), np.concatenate(
                        (np.asarray(entity_inds), np.asarray(other_inds))
                )))
            else:
                ob = np.asarray(entity_inds)
        # debugging obs space: to test whether just looking at the closest landmark
        # is reasonable
        elif obs_type == 'simple':
            ob = []
            dists = [np.linalg.norm(agent.state.p_pos - lm.state.p_pos) for lm in world.landmarks]
            closest_landmark_idx = np.argmin(np.array(dists))
            # TODO: make sure that this is being constructed properly
            closest_landmark_diff = entity_pos[closest_landmark_idx]
            for d in closest_landmark_diff:
                if d < -t:
                    ob.extend([1., 0.])
                elif d > t:
                    ob.extend([0., 1.])
                else:
                    ob.extend([0., 0.])
            return np.array(ob)
        # default obs space: continuous 
        else:
            if not agent.adversary:
                ob = np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
            else:
                ob = np.concatenate(entity_pos + other_pos)
        return ob 
