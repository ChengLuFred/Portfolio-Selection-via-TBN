from environment import *
from agent import *
from module.packages import *
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def main():
    """train and test the agent in the environment
    Args:
        "num_iteration": controls the number of iterations.
        seed: set a seed for repilcation
        TEST_NAME: give each experiment a name
    Returns:
        None
    """
    # initialize global variable
    seed = 123
    TEST_NAME = "REINFORCE" + '_' + str(seed)
    num_iteration = 200

    # Initialize the agent and environment
    agent_type = ["REINFORCE", "Baseline", "ActorCritic"][2]
    train_time_start = datetime(1996, 1, 1)
    train_time_end = datetime(2014, 12, 31)
    env_train = market_envrionment([''], train_time_start, train_time_end)

    # train the agent and output training graph
    rewards, trained_agent = trial(environment = env_train, 
                                   env_type = "train", 
                                   agent_generator = None, 
                                   agent_type = agent_type, 
                                   num_iteration = num_iteration, 
                                   test_name = None)
    training_figure(num_iteration, rewards, TEST_NAME)

    # test well trained agent on following years
    test_time_start = datetime(2015, 1, 1)
    test_time_end = datetime(2017, 12, 31)
    env_test = market_envrionment([''], test_time_start, test_time_end)
    trial(environment = env_test, env_type = "test", agent_generator = trained_agent, agent_type = agent_type, num_iteration = 1, test_name = TEST_NAME)


def trial(environment, env_type, agent_generator, agent_type, num_iteration, test_name):
    '''agent interact with environment
    Args:
        environment:
        env_type: whether it's for train or test
        agent_generator:
        agent_type:
        num_iteration:
        test_name:
    
    Returns:
        rewars: total rewards for each iteration
        agent: a well trained agent
    '''
    # initialize agent and environment
    env = environment
    if agent_generator is not None:
        agent = agent_generator
    else:
        agent = ReinforceBaselineAgent(alpha=5e-4, gamma=1, alpha_w=5e-2)

    # record the output
    if env_type == "test":
        num_iteration = 1
    rewards = np.zeros(num_iteration)

    # training iteration
    for iteration_idx in tqdm(range(num_iteration)):
        # initialize data recoder and enviroment
        rewards_sum = 0
        reward_each = 0
        state = 0
        env.reset()

        while True:
            # agent takes action            
            action = agent.choose_action(state, reward_each)

            # environment response
            reward_each, state = env.step(action)

            # record reward for each step            
            if not math.isnan(reward_each):
                rewards_sum += reward_each

            # update the agent at the end of each episode
            if env.episode_end():
                if env_type == "test":
                    env.data_graph(test_name)
                    env.output_to_file()
                else:
                	agent.episode_end()
                break

        # record total reward and return
        rewards[iteration_idx] = rewards_sum

    return rewards, agent

def training_figure(num_iteration, rewards, test_name):
    '''
    TO Do
    '''

if __name__ == '__main__':
    main()
