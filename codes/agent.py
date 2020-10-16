from packages import *

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

    def choose_action(self, position, reward):
        """
        return the action according to the policy
        """