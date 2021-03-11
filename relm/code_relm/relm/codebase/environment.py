import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from .utils import objectives
import copy
import time

class TheEnvironment:

    def __init__(self, reward_measure, cluster=1, sample_size=0.5, thresh=0.5, tol=0.1, random_state=None):

        self.reward_measure = reward_measure
        self.thresh = thresh
        self.tol = tol
        self.sample_size = sample_size
        self.random_state = random_state
        self.cluster = cluster

        self.shuffler = StratifiedShuffleSplit(n_splits=self.cluster,
                                               test_size=self.sample_size,
                                               random_state=self.random_state)

        self.X_new = self.y_new = self.X_old = self.y_old = None
        self.state = None

    def __initiate_environment(self, X_new, y_new, X_old, y_old):
        self.X_new = X_new
        self.y_new = y_new
        self.X_old, self.y_old = self.sample_data(X_old, y_old)

    def __sample_data(self, X_old, y_old):
        # space where sampling would be done
        for splits in self.shuffler.split(X_old, y_old):
            split = splits[1]
        self.X_old = X_old[split]
        self.y_old = y_old[split]

    def new_data(self, X_new, y_new):
        if self.X_new is not None:
            if self.X_old is not None:
                X_old = self.X_old
                y_old = self.y_old
                self.__sample_data(self.X_new, self.y_new)
                self.X_old = np.append(X_old, self.X_old, axis=0)
                self.y_old = np.append(y_old, self.y_old, axis=0)
                self.X_new = X_new
                self.y_new = y_new

            else:
                self.__sample_data(self.X_new, self.y_new)
                self.X_new = X_new
                self.y_new = y_new
            self.state += 1
        else:
            if self.state is None:
                self.state = 1
            self.X_new = X_new
            self.y_new = y_new

    def action_space(self, test_old=False, return_y=False):

        if self.X_new is not None and not test_old:
            if return_y:
                return self.X_new, self.y_new
            return self.X_new

        elif self.X_new is not None and test_old:
            if self.X_old is not None:
                if return_y:
                    return np.append(self.X_new, self.X_old, axis=0), np.append(self.y_new, self.y_old, axis=0)
                return np.append(self.X_new, self.X_old, axis=0)

            else:
                print('No previous data sapce available with Environment! Returning latest data space!')
                if return_y:
                    return self.X_new, self.y_new
                return self.X_new

        elif self.X_new is None:
            if self.X_old is not None:
                if return_y:
                    return self.X_old, self.y_old
                return self.X_old
            else:
                print('No data available with Environment! Returning None!')
                if return_y:
                    return None, None
                return None

    def step(self, y_pred, y_true=None, test_old=False, rewards_params={}):
        if y_true is None:
            if self.y_new is not None and not test_old:
                y_true = self.y_new
            elif self.y_new is not None and test_old:
                if self.X_old is not None:
                    y_true = np.append(self.y_new, self.y_old, axis=0)
                else:
                    print('No previous data sapce available with Environment! Returning latest data space!')
                    y_true = self.y_new
            elif self.y_new is None:
                if self.y_old is not None:
                    y_true = self.y_old
                else:
                    print('No data available with Environment! Returning None!')
                    y_true = None

        # palce where reward strategy can be changed
        reward_value = objectives[self.reward_measure](y_true, y_pred, **rewards_params)
        #         reward = 1 if reward_value >= (self.thresh -(self.thresh*self.tol)) else 0
        return reward_value


# the relm_code game
class RELMEvaluator(object):

    def __init__(self, num_episodes=10, select_k=5, threshold=12, solved_state=5,
                 sample_size=0.99, num_iterations=20, is_warm=True):

        self.num_episodes = num_episodes
        self.select_k =select_k
        self.threshold =threshold
        self.sample_size =sample_size
        self.num_iterations = num_iterations
        self.is_warm = is_warm
        self.population = None
        self.solved_state = solved_state

    def learn(self, agent, environment):
        # experiment rewards list and new_learnings
        rewards_list = []
        new_learnings = []

        # wether agent has been performing consitent
        solved = 0

        # warm start: wether any previous population to be used or not
        # pop_model_param = population.model_param

        # model from the population
        # model = population.select_best()

        #timer
        t = time.time()

        # doing experimentation
        for i in range(self.num_episodes):

            # Reset environment and get newly added data observation
            Q, Q_y = environment.action_space(return_y=True)

            # Shuffling data to avoid similar search space
            # np.random.shuffle(Q)

            # Iterting for rewards accors different sample sets
            #     reward = np.zeros(num_iterations)
            rewards = []

            # mask for random sampling
            mask = np.random.choice([True, False], len(Q), p=[self.sample_size, 1 - self.sample_size])

            # backing present agent's learning
            agent_ = agent.who_am_i()

            #doing test iterations
            print('\nRunning Episode: {} Action Space: {}'.format(i + 1, Q.shape))
            for j in range(self.num_iterations):
                mask_it = mask
                Q_ = Q[mask_it]
                Q_y_ = Q_y[mask_it]
                y_pred, y_pred_bin = agent.evaluate(Q_)
                reward = environment.step(y_pred=y_pred_bin, y_true=Q_y_)
                #         reward = f1_score(y_true,y_pred_bin)
                #         print('For Iteration: {} Sample Size :{} Reward: {}'.format(i,len(mask_it),reward))
                #         reward[i] = reward
                rewards.append(reward)

            rewards_list.append(rewards)
            print('Rewards for Episode are: {} Individual values: {}'.format(np.sum(rewards), rewards))

            # Checking reward threshold for new traing on data
            if np.sum(rewards) < float(self.threshold):
                # resetting agents history for new learning
                agent.reset_history()
                # changing agents name to inform about environment and game episode
                agent.name = '{}_{}_{}'.format(agent_['name'], environment.state, i)

                Q_sampled, y_sampled = environment.action_space(test_old=True, return_y=True)

                print('\nMaking agent learn form the new data!')
                print('Data Size: {} True Positives: {}'.format(Q_sampled.shape, sum(y_sampled)))

                # getting population as prior for next iteration
                if self.is_warm:
                    pop_solutions, pop_fitness = agent.best_population.get_population(self.select_k)
                    _ = agent.solver.ask()
                    agent.solver.solutions = pop_solutions
                    agent.solver.tell(pop_fitness)
                    # adaptive k to use better population in nest iterations
                    self.select_k += 2

                print('Evoluting for newer generation!')
                agent.execute(Q_sampled, y_sampled)

                print('Taking last best population as to measure samples!')
                # population = new_learning[1]
                population = agent.best_population
                # model = population.select_best()
                new_learnings.append(agent.score_list)

                print('Population of {} has evolved with fitness:\n'.format(len(population.genes)),
                      population.fitness[population.fitness_measure])

                agent.accumulate_history(agent_)
            else:
                solved += 1
                new_learnings.append([])
                print('No new learning for this iteration! Good Luck for next episode.. ;\)')

            # resetting agent's name to original
            agent.who_was_me(agent_)

            if solved >= self.solved_state:
                break

        print('Total Time for relm_code Evaluation: {:>5.1f}'.format(time.time()-t))
        self.population = population
        self.learnings = new_learnings

        return agent


def reinforced_learning(envr, population, learning_iterations, problem='binaryclass',
                        name=None, select_k=5, threshold=12, num_episodes=10, sample_size=0.99,
                        num_iterations=20, is_warm=True):
    # experiment rewards list and new_learnings
    rewards_list = []
    new_learnings = []

    # wether agent has been performing consitent
    solved = 0

    # warm start: wether any previous population to be used or not
    pop_model_param = population.model_param

    # model from the population
    model = population.select_best()

    # doing experimentation
    for i in range(num_episodes):

        # Reset environment and get newly added data observation
        Q, Q_y = envr.action_space(return_y=True)

        # Shuffling data to avoid similar search space
        # np.random.shuffle(Q)

        # Iterting for rewards accors different sample sets
        #     reward = np.zeros(num_iterations)
        rewards = []

        # mask for random sampling
        mask = np.random.choice([True, False], len(Q), p=[sample_size, 1 - sample_size])

        print('\nRunning Episode: {} Action Space: {}'.format(i + 1, Q.shape))
        for j in range(num_iterations):
            mask_it = mask
            Q_ = Q[mask_it]
            Q_y_ = Q_y[mask_it]
            y_true, y_pred, y_pred_bin = evaluate(Q_, Q_y_, model, CUTOFF)
            reward = envr.step(y_pred=y_pred_bin, y_true=y_true)
            #         reward = f1_score(y_true,y_pred_bin)
            #         print('For Iteration: {} Sample Size :{} Reward: {}'.format(i,len(mask_it),reward))
            #         reward[i] = reward
            rewards.append(reward)

        rewards_list.append(rewards)
        print('Rewards for Episode are: {} Individual values: {}'.format(np.sum(rewards), rewards))

        # Checking reward threshold for new traing on data
        if np.sum(rewards) < threshold:
            Q_sampled, y_sampled = envr.action_space(test_old=True, return_y=True)

            print('\nMaking agent learn form the new data!')
            print('Data Size: {} True Positives: {}'.format(Q_sampled.shape, sum(y_sampled)))

            # getting population as prior for next iteration
            if is_warm:
                pop_solutions, pop_fitness = population.get_population(select_k)
                _ = solver.ask()
                solver.solutions = pop_solutions
                solver.tell(pop_fitness)

            print('Evoluting for newer generation!')
            new_learning = work(Q_sampled, y_sampled,
                                solver=solver,
                                obj_func=obj_func,
                                problem=problem,
                                n_iteration=learning_iterations,
                                model_params=pop_model_param,
                                name='{}_rl_{}'.format(name, i + 1),
                                location='/Users/jitins_lab/Documents/experiment/notebooks/')

            print('Taking last best population as to measure samples!')
            population = new_learning[1]
            model = population.select_best()
            new_learnings.append(new_learning)

            print('Population of {} has evolved with fitness:\n'.format(len(population.genes)),
                  population.fitness[population.fitness_measure])
        else:
            solved += 1
            new_learnings.append([])
            print('No new learning for this iteration! Good Luck for next episode.. ;\)')

        if solved >= 5:
            break

    return population, new_learnings

