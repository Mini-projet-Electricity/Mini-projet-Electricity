import pypownet.environment

"""
Concerning the preprocessing, we want to have some good data set, so we did a first draw on the raw data :
  we manually removed the data that we talked about in the proposal (the ones which repeated themselves every week day and weekend day).
  Hence we only kept 2/7th of the data. In order to know which lines in the table to remove, we searched for patterns.
  In the _N_loads_p file in the public_data directory, we took a look at the first column, which contains the loads numbers corresponding to
  the biggest curve in the Load Time Series graph on the notebook. As we can see on that graph, there is the same pattern every week day,
  but the values slightly vary. And indeed in the data file, we couldn't find twice the same exact values. We knew to look at 288 lines sets,
  because every line corresponds to 5 minutes, and there is 288 times 5 minutes in a day. Hence we managed to keep only part of the original
  data set for our agent. However, the manual selection has to be repeated manually, and the new data files are only found on our computer,
  we haven't found an algorithm that works yet. So we are not able to completely give you what you asked for.

"""


class TreeSearchLineServiceStatus(Agent):
    """ Exhaustive tree search of depth 1 limited to no action + 1 line switch activation
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.verbose = True

        self.ioman = ActIOnManager(destination_path='saved_actions_TreeSearchLineServiceStatus.csv')

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        number_of_lines = self.environment.action_space.lines_status_subaction_length
        # Simulate the line status switch of every line, independently, and save rewards for each simulation (also store
        # the actions for best-picking strat)
        simulated_rewards = []
        simulated_actions = []
        for l in range(number_of_lines):
            if self.verbose:
                print('    Simulating switch activation line %d' % l, end='')
            # Construct the action where only line status of line l is switched
            action = action_space.get_do_nothing_action()
            action_space.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)
            simulated_reward = self.environment.simulate(action=action)

            # Store ROI values
            simulated_rewards.append(simulated_reward)
            simulated_actions.append(action)
            if self.verbose:
                print('; expected reward %.5f' % simulated_reward)

        # Also simulate the do nothing action
        if self.verbose:
            print('    Simulating switch activation line %d' % l, end='')
        donothing_action = self.environment.action_space.get_do_nothing_action()
        donothing_simulated_reward = self.environment.simulate(action=donothing_action)
        simulated_rewards.append(donothing_simulated_reward)
        simulated_actions.append(donothing_action)

        # Seek for the action that maximizes the reward
        best_simulated_reward = np.max(simulated_rewards)
        best_action = simulated_actions[simulated_rewards.index(best_simulated_reward)]

        # Dump best action into stored actions file
        self.ioman.dump(best_action)

        if self.verbose:
            print('  Best simulated action: disconnect line %d; expected reward: %.5f' % (
                simulated_rewards.index(best_simulated_reward), best_simulated_reward))

        return best_action
