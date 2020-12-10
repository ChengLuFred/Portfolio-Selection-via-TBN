from module.packages import *

class PolicyGradient:
    """
    ReinforceAgent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma, num_of_tilings=2, max_size=2048):
        # variable initialization
        self.alpha = alpha / num_of_tilings
        self.gamma = gamma
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings

        # a map from state to its index
        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)

        # position needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (0.05 - (-0.05))

        # variable declaration
        self.states = []
        self.rewards = []
        self.actions = []

        # action space
        self.ACTIONS = np.arange(0, 1, step = 0.01)


    # get indices of active tiles for given state and action
    def get_active_tiles(self, position, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tiles(self.hash_table, 
                             self.num_of_tilings,
                            [self.position_scale * position],
                            [action])
        return active_tiles # tiles index in each tilings

    # estimate the preference of given state and action
    def get_h(self, position, action):
        active_tiles = self.get_active_tiles(position, action)
        return np.sum(self.weights[active_tiles])

    def get_pi(self, position):
        """
        policy part: return the probability mass function(pmf) 
        """
        # preference vector for three actions
        h = []
        for action in self.ACTIONS:
            h.append(self.get_h(position, action))

        # soft-max in action preference
        t = np.exp(h - np.max(h))
        pmf = t / np.sum(t)       

        return pmf

    def choose_action(self, position, reward):
        """
        return the action according to the policy
        """
                # recording the reward, state for update using
        if reward is not None:
            self.rewards.append(reward)
        if position is not None:
            self.states.append(position)

        # choose the action accordingly
        pmf = self.get_pi(position)
        action = np.random.choice(3, p=pmf) - 1      
        self.actions.append(action)

        return action

class ReinforceBaselineAgent(PolicyGradient):
    def __init__(self, alpha, alpha_w, gamma, num_of_tilings = 2, max_size = 2048):
        PolicyGradient.__init__(self, alpha, gamma, num_of_tilings, max_size)
        self.alpha_w = alpha_w / num_of_tilings

        # a new hash table for baseline
        self.hash_table_baseline = IHT(max_size)

        # new weights vector for baseline
        self.weights_baseline = np.zeros(max_size)

    def get_active_tiles_baseline(self, position):
        """
        get indices of active tiles for given state
        """
        active_tiles = tiles(self.hash_table_baseline, 
                             self.num_of_tilings,
                            [self.position_scale * position])
        #print("position is ", self.position_scale * position)
        return active_tiles # tiles index in each tilings

    def episode_end(self):
        """
        clear all memory
        """
        gamma_pow = 1

        # compute total return from the end to start
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]
        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]        
        
        # learn theta
        for i in range(len(G)):
            pmf = self.get_pi(self.states[i])
            active_tiles = self.get_active_tiles_baseline(self.states[i])
            baseline = np.sum(self.weights_baseline[active_tiles])
            delta = G[i] - baseline

            # update baseline
            update_baseline = self.alpha_w * delta
            for active_tile in active_tiles:
                self.weights_baseline[active_tile] += update_baseline

            # update agent
            for action in self.ACTIONS:
                if action == self.actions[i]:
                    grad_ln_pi = 1 - pmf[action + 1]
                else:
                    grad_ln_pi = -pmf[action + 1]
                update = self.alpha * gamma_pow * baseline * grad_ln_pi
                active_tile = self.get_active_tiles(self.states[i], action)
                self.weights[active_tile] += update

            gamma_pow *= self.gamma

        # clear the record at the end waiting for another iteration
        self.rewards = []
        self.actions = []
        self.states = [0]