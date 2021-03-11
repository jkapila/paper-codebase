import numpy as np
from collections import namedtuple
import json
import copy


# Defining the neural network model

ModelParam = namedtuple('ModelParam',
                        ['input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise'])

model_params = {}
model_test1 = ModelParam(
    input_size=9,
    output_size=1,
    layers=[45, 5],
    activation=['sigmoid'],
    noise_bias=0.0,
    output_noise=[False, False, True],
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def passthru(x):
    return x


# useful for discrete actions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# useful for discrete actions
def sample(p):
    return np.argmax(np.random.multinomial(1, p))


activate = {'sigmoid': sigmoid, 'relu': relu, 'tanh': np.tanh, 'softmax': softmax, 'passthru': passthru}


class Model(object):
    ''' simple feedforward model '''

    def __init__(self, model_params=None):

        if model_params is not None:
            # Requirement for mfa to initialize
            # self.output_noise = model_params.output_noise
            self.layers = model_params.layers
            self.input_size = model_params.input_size
            self.output_size = model_params.output_size
            self.activation = model_params.activation

            # secondary requirement
            self.rnn_mode = False  # in the future will be useful
            self.time_input = 0  # use extra sinusoid input
            self.sigma_bias = model_params.noise_bias  # bias in stdev of output
            self.noise_bias = model_params.noise_bias
            self.sigma_factor = 0.5  # multiplicative in stdev of output
            self.output_noise = model_params.output_noise

            self.shapes = []
            self.sample_output = False
            self.render_mode = False
            self.weight = []
            self.bias = []
            self.param_count = 0

            self.initialize()

    def initialize(self):

        # Setting shapes for weight matrix
        self.shapes = []
        for _layer in range(len(self.layers)):
            if _layer == 0:
                self.shapes = [(self.input_size, self.layers[_layer])]
            else:
                self.shapes.append((self.layers[_layer - 1], self.layers[_layer]))
        self.shapes.append((self.layers[_layer], self.output_size))

        # setting activations for the model
        if len(self.activation) > 1:
            self.activations = [activate[x] for x in self.activation]
        elif self.activation[0] == 'relu':
            self.activations = [relu, relu, passthru]
        elif self.activation[0] == 'sigmoid':
            self.activations = [np.tanh, np.tanh, sigmoid]
        elif self.activation[0] == 'softmax':
            self.activations = [np.tanh, np.tanh, softmax]
            self.sample_output = True
        elif self.activation[0] == 'passthru':
            self.activations = [np.tanh, np.tanh, passthru]
        else:
            self.activations = [np.tanh, np.tanh, np.tanh]

        self.weight = []
        self.bias = []
        # self.bias_log_std = []
        # self.bias_std = []
        self.param_count = 0

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])
            # if self.output_noise[idx]:
            #     self.param_count += shape[1]
            #     log_std = np.zeros(shape=shape[1])
            #     self.bias_log_std.append(log_std)
            #     out_std = np.exp(self.sigma_factor*log_std + self.sigma_bias)
            #     self.bias_std.append(out_std)
            #     idx += 1

    def get_action(self, X, mean_mode=False):
        # if mean_mode = True, ignore sampling.
        h = np.array(X).flatten()
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            # if (self.output_noise[i] and (not mean_mode)):
            #     out_size = self.shapes[i][1]
            #     out_std = self.bias_std[i]
            #     output_noise = np.random.randn(out_size)*out_std
            #     h += output_noise
            h = self.activations[i](h)

        if self.sample_output:
            h = sample(h)

        return h

    def get_action_once(self, X, mean_mode=False):
        # if mean_mode = True, ignore sampling.
        h = X
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.dot(h, w) + b
            # if (self.output_noise[i] and (not mean_mode)):
            #     out_size = self.shapes[i][1]
            #     out_std = self.bias_std[i]
            #     output_noise = np.random.randn(out_size)*out_std
            #     h += output_noise
            h = self.activations[i](h)

        if self.sample_output:
            h = sample(h)

        return h

    def set_model_params(self, gene):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(gene[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            # if self.output_noise[i]:
            #     s = b_shape
            #     self.bias_log_std[i] = np.array(gene[pointer:pointer+s])
            #     self.bias_std[i] = np.exp(self.sigma_factor*self.bias_log_std[i] + self.sigma_bias)
            if self.render_mode:
                print("bias_std, layer", i, self.bias_std[i])
            pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            datastore = json.load(f)
        model_param = ModelParam(
            input_size=datastore['input_size'],
            output_size=datastore['output_size'],
            layers=datastore['layers'],
            activation=datastore['activation'],
            noise_bias=datastore['noise_bias'],
            output_noise=datastore['output_noise'],
        )
        model_ = Model(model_param)
        model_.set_model_params(datastore['gene'])
        print('Loading model from file: {}'.format(filename))
        return model_

    def save_model(self, filename=None):
        modelstore = copy.deepcopy(self.__dict__)
        datastore = {}
        for key in modelstore.keys():
            if key in ['input_size', 'output_size', 'layers', 'activation',
                       'noise_bias', 'output_noise']:
                datastore[key] = modelstore[key]
        datastore['gene'] = self.get_gene()
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(datastore, f)
            print('Saving model to file: {}'.format(filename))
        else:
            print('Please define filename to store model object!')

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev

    def get_gene(self):

        gene = []
        for i in range(len(self.weight)):
            w = self.weight[i].reshape(-1).tolist()
            b = self.bias[i].reshape(-2).tolist()
            gene.extend(w)
            gene.extend(b)
        assert len(gene) == self.param_count
        return gene

    def __repr__(self):
        modelstore = copy.deepcopy(self.__dict__)
        s = 'Model Characteristics'
        for key in modelstore.keys():
            if key in ['input_size', 'output_size', 'layers', 'activation',
                       'noise_bias', 'output_noise']:
                s = '{} \n{}: {}'.format(s,key,modelstore[key])

        return s


class ModelPopulation(object):

    def __init__(self):
        self.model_param = None
        self.model = None
        self.genes = None
        self.fitness = None
        self.scores = None
        self.scorers = None
        # right now this is just index number passed to list
        # todo: make a dist of measures and use it to take vlue from it
        self.fitness_measure = None
        self.size = None

    def initialize(self):
        if self.model_param is not None:
            self.model = Model(self.model_param)

    def save_population(self, model, solutions, fitness, measure, scores, scorers, filename=''):
        modelstore = copy.deepcopy(model.__dict__)
        datastore = {}
        for key in modelstore.keys():
            if key in ['input_size', 'output_size', 'layers', 'activation',
                       'noise_bias', 'output_noise']:
                datastore[key] = modelstore[key]
        datastore['genes'] = solutions
        datastore['fitness'] = fitness
        datastore['scores'] = scores
        datastore['scorers'] = scorers
        datastore['fitness_measure'] = measure
        datastore['size'] = len(solutions)
        # for key in datastore.keys():
        #     print('{} has type {}'.format(key,type(datastore[key])))
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(datastore, f)
            print('Saving model population to file: {}'.format(filename))
        else:
            print('Please define filename to store model object!')

    def load_population(self, filename, silent=True):
        with open(filename) as f:
            datastore = json.load(f)
        model_param = ModelParam(
            input_size=datastore['input_size'],
            output_size=datastore['output_size'],
            layers=datastore['layers'],
            activation=datastore['activation'],
            noise_bias=datastore['noise_bias'],
            output_noise=datastore['output_noise'],
        )
        self.model_param = model_param
        self.model = Model(model_param)
        self.genes = datastore['genes']
        self.fitness = datastore['fitness']
        self.scores = datastore['scores']
        self.scorers = datastore['scorers']
        self.size = datastore['size']
        self.fitness_measure = datastore['fitness_measure']
        if not silent:
            print('Loading model population from file: {}'.format(filename))
            fitment = self.fitness
            print('Population Size: {} Fitness Measure: {} Best Fitness: {}'.format(len(self.genes),
                                                                                    self.scorers,
                                                                                    fitment[np.argmax(fitment)]))
        return self

    # todo: put logic in place
    def select_best(self, k=1, fill_population=False):
        if self.genes is not None:
            if k == 1:
                # todo: put logic for k right now returns the best in population
                fitment = self.fitness
                gene = self.genes[np.argmax(fitment)]
                model = self.model
                model.set_model_params(gene)
                return model
            else:
                if k < 1 and k > 0 and isinstance(k, float):
                    p = np.percentile(self.fitness, q=int((1 - k) * 100))
                    ind = np.where(self.fitness > k)
                else:
                    ind = np.argpartition(self.fitness, -k)[-k:]

                if fill_population:
                    ind = np.tile(ind, int(self.size / len(ind)) + 1)
                    ind = ind[:self.size]
                    assert ind.shape[0] == self.size
                    print('Population filled with {}'.format(ind))
                models = []
                for i in ind:
                    model = self.model
                    model.set_model_params(self.genes[np.argmax(i)])
                    models.append(model)
                return models
        else:
            model = self.model
            model.set_model_params(np.random.random(size=self.model.param_count))
            return model

    def put_population(self, genes, fitnesses):
        if len(genes) < self.size:
            ind = list(range(len(genes)))
            inds = np.random.choice(ind, size=self.size)
            gene = []
            fitness = []
            for i in inds:
                gene.append(genes[i])
                fitness.append(fitnesses[i])
            self.genes = gene
            self.fitness = fitness
        else:
            self.genes = genes
            self.fitness = fitnesses

    def get_population(self, k=1, fill=True):

        if k >= 1 and k <= self.size:
            ind = np.argpartition(self.fitness, -k)[-k:]
        elif k < 1 and k > 0 and isinstance(k, float):
            p = np.percentile(self.fitness, q=int((1 - k) * 100))
            ind = np.where(np.array(self.fitness) >= p)
        else:
            print('Either k in negative or greater than size of population! Returning the whole population!')
            return self.genes, self.fitness

        if fill:
            #             print(ind)
            ind = np.tile(ind, int(self.size / len(ind)) + 1)
            #             print(ind)
            ind = ind.flatten()[:(self.size)]
            #             print(ind)
            assert ind.shape[0] == self.size, '{} and {} dont match. Check for correctness!'.format(ind.shape[0],
                                                                                                    self.size)
            print('Population filled with {}'.format(ind))

        gene = []
        fitness = []
        for i in ind:
            gene.append(self.genes[i])
            fitness.append(self.fitness[i])
        return gene, fitness

    def evaluate_batch(self):
        pass
