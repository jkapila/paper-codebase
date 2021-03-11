from __future__ import print_function, division
import os
import sys
from numba import jit, njit, prange
import numpy as np
import pandas as pd

from .utils import objectives
from .model_population import Model, ModelPopulation
# from .esmethods.methods import SimpleGA, PEPG, CMAES, OpenES
from .utils import plot_learning, plot_ROC

import time


# main class of relm_agent
class RELMAgent(object):

    def __init__(self, model_params, solver, n_population=21, n_iteration=150, problem='binaryclass',
                 cutoff=0.9, scorers=None, name=None, location=None):

        self.model_params = model_params
        self.solver = solver
        if solver.popsize == n_population:
            self.n_population = n_population
        else:
            print('Polupation is different for solver. Replacing with solver\'s population!')
            self.n_population = solver.popsize

        self.n_iteration = n_iteration
        self.cutoff = cutoff
        self.problem = problem

        if problem == 'binaryclass':
            self.scorers = ['f1_score', 'accuracy', 'loss', 'cohen_kappa']
        elif problem == 'multiclass':
            self.scorers = ['f1_score_multi', 'accuracy_multi', 'loss_multi', 'cohen_kappa_multi']
        elif problem == 'regression':
            self.scorers = ['1_mape', 'mse', 'mi', 'kld']

        if scorers is not None:
            if scorers not in self.scorers:
                assert isinstance(scorers, str)
                self.scorers = self.scorers.append(scorers)

        self.name = 'model' if name is None else name
        self.location = os.path.join(os.getcwd(), 'modelstore') if location is None else location

        self.history = None
        self.score_list = None
        self.best_population = None
        self.last_population = None
        self.best_model = None
        self.best_population_path = None
        self.last_population_path = None

    def __evaluate(self, X, model):
        pred = model.get_action_once(X)
        cut_pred = np.vectorize(lambda x: 1 if x >= self.cutoff else 0)
        y_pred = pred
        y_pred_bin = cut_pred(y_pred)

        return y_pred, y_pred_bin

    def __fitness(self, X, y, solutions, model):
        model_ = model
        popsize = len(solutions)
        scores = np.zeros((popsize, len(self.scorers)))
        # this is what can be parallalized.. ;)
        for i in range(popsize):
            model_.set_model_params(solutions[i])
            y_true_tot = y
            y_pred_tot, y_pred_bin_tot = self.__evaluate(X, model_)

            # calculating losses
            for j, scorer in enumerate(self.scorers):
                if str(scorer).startswith('f1') or str(scorer).startswith('acc') or str(scorer).startswith('cohen'):
                    scores[i, j] = objectives[scorer](y_true_tot, y_pred_bin_tot)
                else:
                    scores[i, j] = objectives[scorer](y_true_tot, y_pred_tot)

        return scores.T

    def __solver(self):
        pass

    def __objective(self, scores):
        fitness_values = scores[0, :]
        position = np.argmax(fitness_values)

        return fitness_values, position

    def __worker(self, X, y):

        model = Model(self.model_params)

        history = []
        score_list = []
        best_solution = []
        best_model = None
        best_population = None
        n_test_iter = 10

        print('Scorers for the experiment are: {}'.format(self.scorers))

        for j in range(self.n_iteration):
            t = time.time()

            solutions = self.solver.ask()

            scores = self.__fitness(X, y, solutions, model)
            fitness_values, position = self.__objective(scores)

            self.solver.tell(fitness_values)
            result = self.solver.result()  # first element is the best solution, second element is the best fitness

            history.append(result[1])
            score_list.append(scores)

            s = 'Iteration:{:>3d} Time: {:>3.1f} secs has fitness: {:>5.6f}'.format(
                (j + 1), (time.time() - t), result[1])

            for i, scorer in enumerate(self.scorers):
                score = scores[i]
                s += ' {}: {:>5.6f}'.format(scorer, score[position])

            print(s)

            fitness_max = fitness_values[position]

            # todo : seperately after x iterations we will write/overwrite the matrix and save the model based best fitness
            if (best_model is None or fitness_max >= best_model):
                best_solution = solutions[position]
                best_model = fitness_max
                if (j + 1) >= n_test_iter or (j + 1) > 5:
                    print('New best solution found!')
                    model.set_model_params(best_solution)
                    path = '{}/{}_{}_{}_{}_{}.json'.format(self.location, self.name, X.shape[0], self.n_population,
                                                           self.n_iteration, (j + 1))
                    model.save_model(path)
                    self.best_model = model
                    path = '{}/{}_population_{}_{}_{}.json'.format(self.location, self.name, X.shape[0],
                                                                   self.n_population, self.n_iteration)
                    ModelPopulation().save_population(model=model,
                                                      solutions=solutions.tolist(),
                                                      fitness=fitness_values.tolist(),
                                                      scores=scores.tolist(),
                                                      scorers=self.scorers,
                                                      measure=position.tolist(),
                                                      filename=path)
                    self.best_population = ModelPopulation().load_population(path)
                    self.best_population_path = path

            if j == n_test_iter:
                n_test_iter += 5

        last_solution = solutions[position]
        last_solutions = solutions
        print('Saving last solution found!')
        model.set_model_params(best_solution)
        path = '{}/{}_last_{}_{}_{}_{}.json'.format(self.location, self.name, X.shape[0], self.n_population,
                                                    self.n_iteration, (j + 1))
        model.save_model(path)
        path = '{}/{}_population_last_{}_{}_{}.json'.format(self.location, self.name, X.shape[0], self.n_population,
                                                            self.n_iteration)
        self.last_population_path = path
        ModelPopulation().save_population(model=model,
                                          solutions=solutions.tolist(),
                                          fitness=fitness_values.tolist(),
                                          scores=scores.tolist(),
                                          scorers=self.scorers,
                                          measure=position.tolist(),
                                          filename=path)
        self.last_population = ModelPopulation().load_population(path)

        if self.history is None:
            self.history = history
            self.score_list = score_list
        else:
            self.history.append(history)
            self.score_list.append(score_list)

    def execute(self, X, y):
        t = time.time()
        self.__worker(X, y)
        print('Total time of learning: {:>5.1f}'.format(time.time() - t))

    def evaluate(self, X):
        if self.best_model is None:
            self.best_population = self.solver.ask()
            model_index = np.random.randint(self.n_population, size=1)
            model = Model(model_params=self.model_params)
            model.set_model_params(self.best_population[model_index][0])
            self.best_model = model

        return self.__evaluate(X, self.best_model)

    def plot_scoring(self, X_train, y_train, X_new, y_new):
        if self.best_model is not None:
            y_t = y_train
            y_p, y_p_b = self.__evaluate(X_train, self.best_model)
            df_pred_train = pd.DataFrame(
                {'y_true': y_t.flatten(), 'y_pred': y_p.flatten(), 'y_pred_binary': y_p_b.flatten()})

            y_t = y_new
            y_p, y_p_b = self.__evaluate(X_new, self.best_model)
            df_pred_test = pd.DataFrame(
                {'y_true': y_t.flatten(), 'y_pred': y_p.flatten(), 'y_pred_binary': y_p_b.flatten()})

            plot_ROC(df_pred_train.y_true, df_pred_train.y_pred, df_pred_test.y_true, df_pred_test.y_pred,
                     threshold=0.9, path=self.location, name=self.name)
        else:
            print('There is no model to evaluate for!')

    def plot_relm_learning(self, old_agent):

        old_tr = old_agent.score_list
        # print(len(old_tr))
        # print(len(rl_history))
        old_tr.append(self.score_list)
        old_tr = np.array(old_tr)
        print('Total history of learning: {}'.format(old_tr.shape))
        plot_learning(old_tr, path=self.location, name='{}_relm'.format(self.name))

    def plot_history(self):
        plt_data = np.array(self.score_list)
        print('Learning of agent :{}'.format(plt_data.shape))
        plot_learning(plt_data, path=self.location, name=self.name)

    def load_last_populations(self):
        self.last_population = ModelPopulation().load_population(self.last_population_path)
        self.best_population = ModelPopulation().load_population(self.best_population_path)

    def load_population(self, path):
        self.best_population = ModelPopulation().load_population(path)
        self.best_model = self.best_population.select_best()

    def reset_history(self):
        self.history = None
        self.score_list = None

    def accumulate_history(self, old_agent):

        if isinstance(old_agent, dict):
            old_history = old_agent['my_history']
            old_scores = old_agent['my_scores']
        else:
            old_history = old_agent.history
            old_scores = old_agent.score_list

        if old_history is not None:
            self.history = old_history.append(self.history)
            self.score_list = old_scores.append(self.score_list)

    def who_am_i(self):
        features = {'name': self.name, 'location': self.location, 'scorers': self.scorers,
                    'my_history': self.history, 'my_scores': self.score_list, 'best_model': self.best_model,
                    'best_population': self.best_population, 'cutoff': self.cutoff,
                    'n_iteration': self.n_iteration, 'problem': self.problem}
        return features

    def who_was_me(self, features):
        self.name = features['name']
        self.location = features['location']
        self.scorers = features['scorers']
        self.cutoff = features['cutoff']
        self.n_iteration = features['n_iteration']
        self.problem = features['problem']

    def save_agent(self):
        pass

    def load_agent(self):
        pass


# Evaluation function

def eval_parallel(X, model, cutoff):
    pred = model.get_action_once(X)
    cut_pred = np.vectorize(lambda x: 1 if x >= cutoff else 0)
    y_pred = pred
    y_pred_bin = cut_pred(y_pred)

    return y_pred, y_pred_bin


# todo make this function more efficient and paralle
# fitness function for classification
# @jit(parallel = True, fastmath=True)
def fitness(X, y, solutions, model, scorers, cutoff, nthreads, parallel, method=1):
    model_ = model

    if method == 1:
        popsize = len(solutions)
        scores = np.zeros((popsize, len(scorers)))
        for i in range(popsize):

            model_.set_model_params(solutions[i])

            y_true_tot = None
            y_pred_tot = None
            y_pred_bin_tot = None

            # y_true_tot,y_pred_tot,y_pred_bin_tot = evaluate(X,y,model_,cutoff)
            y_true_tot = y
            y_pred_tot, y_pred_bin_tot = eval_parallel(X, model_, cutoff)

            # calculating losses
            for j, scorer in enumerate(scorers):
                if str(scorer).startswith('f1') or str(scorer).startswith('acc') or str(scorer).startswith('cohen'):
                    scores[i, j] = objectives[scorer](y_true_tot, y_pred_bin_tot)
                else:
                    scores[i, j] = objectives[scorer](y_true_tot, y_pred_tot)

    elif method == 2:
        pass

    return scores.T


# todo: make multi-objective
def objective(scores):
    fitness_values = scores[0, :]
    position = np.argmax(fitness_values)

    return fitness_values, position


# main fuction that integrates all parts
# have lot of todo in this
def work(X, y, solver, obj_func, n_iteration, model_params, problem='binaryclass', scorers=None, name=None, location='',
         parallel=False, method=1):
    model = Model(model_params)

    history = []
    score_list = []
    best_solution = []
    best_model = None
    n_test_iter = 10
    if name is None:
        name = 'model'

    if scorers is None:
        if problem == 'binaryclass':
            scorers = ['f1_score', 'accuracy', 'loss', 'cohen_kappa']
        elif problem == 'multiclass':
            scorers = ['f1_score_multi', 'accuracy_multi', 'loss_multi', 'cohen_kappa_multi']
        elif problem == 'regression':
            scorers = ['1_mape', 'mse', 'mi', 'kld']

    print('Scorers for the experiment are: {}'.format(scorers))

    for j in range(n_iteration):
        t = time.time()

        solutions = solver.ask()

        scores = fitness(X, y, solutions, model, scorers, cutoff=0.9, nthreads=10, parallel=parallel, method=method)
        fitness_values, position = obj_func(scores)

        solver.tell(fitness_values)
        result = solver.result()  # first element is the best solution, second element is the best fitness

        history.append(result[1])
        score_list.append(scores)

        s = 'Iteration: {:>3d} Time: {:>3.0f} secs has fitness: {:>8.4f}'.format((j + 1), (time.time() - t), result[1])

        for i, scorer in enumerate(scorers):
            score = scores[i]
            s += ' {}:{:>8.4f}'.format(scorer, score[position])

        print(s)

        fitness_max = fitness_values[position]
        # loss_ = loss_scores[np.argmax(f1_scores)]
        # acc_ = accuracy[np.argmax(f1_scores)]
        # f1_ = f1_scores[np.argmax(f1_scores)]

        # f1_list.append(f1_scores)
        # acc_list.append(accuracy)
        # loss_list.append(loss_scores)
        # print("At iteration {} Fitness: {:>5.4f} Best Accuracy: {:>5.4f}  F1 Score: {:>5.4f} Loss: {:>5.4f} Time {:>3.0f} secs ".format((j+1),
        #             result[1], acc_, f1_, loss_, (time.time()-s)))

        # todo : seperately after x iterations we will write/overwrite the matrix and save the model based best fitness
        if (best_model is None or fitness_max >= best_model):
            best_solution = solutions[position]
            best_model = fitness_max
            if (j + 1) >= n_test_iter or (j + 1) > 5:
                print('New best solution found!')
                model.set_model_params(best_solution)
                path = '{}/{}_{}_{}_{}_{}.json'.format(location, name, X.shape[0], solver.popsize, n_iteration, (j + 1))
                model.save_model(path)
                path = '{}/{}_population_{}_{}_{}.json'.format(location, name, X.shape[0], solver.popsize, n_iteration)
                ModelPopulation().save_population(model=model,
                                                  solutions=solutions.tolist(),
                                                  fitness=fitness_values.tolist(),
                                                  scores=scores.tolist(),
                                                  scorers=scorers,
                                                  measure=position.tolist(),
                                                  filename=path)
                best_population = ModelPopulation().load_population(path)

        if j == n_test_iter:
            n_test_iter += 5

    last_solution = solutions[position]
    last_solutions = solutions
    print('Saving last solution found!')
    model.set_model_params(best_solution)
    path = '{}/{}_last_{}_{}_{}_{}.json'.format(location, name, X.shape[0], solver.popsize, n_iteration, (j + 1))
    model.save_model(path)
    path = '{}/{}_population_last_{}_{}_{}.json'.format(location, name, X.shape[0], solver.popsize, n_iteration)
    ModelPopulation().save_population(model=model,
                                      solutions=solutions.tolist(),
                                      fitness=fitness_values.tolist(),
                                      scores=scores.tolist(),
                                      scorers=scorers,
                                      measure=position.tolist(),
                                      filename=path)
    last_population = ModelPopulation().load_population(path)

    return history, best_population, last_population, score_list
