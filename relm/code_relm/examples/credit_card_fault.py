from relm import Model,ModelParam,RELMAgent,RELMEvaluator,TheEnvironment, CMAES
import pandas as pd
import os



if __name__ == '__main__':
    # loading data
    dir = os.getcwd()
    os.chdir("..")
    dir = os.path.abspath(os.curdir)
    df = pd.read_csv(os.path.join(dir,'data/creditcard.csv'))
    round(df.describe(), 2)
    X = df[[x for x in df.columns if x.startswith('V')]].values
    y = df.Class.values
    print('Data has shape: {}'.format(X.shape))

    # setting agents model behaviour
    model_params = ModelParam(
        input_size=28,
        output_size=1,
        layers=[45, 15, 6],
        activation=['tanh', 'tanh', 'tanh', 'sigmoid'],
        noise_bias=0.0,
        output_noise=[False, False, False, True]
    )

    model = Model(model_params)
    print(model)

    # setting solver specification
    print('\nSolver specification:')
    cmaes = CMAES(model.param_count,
                  popsize=21,
                  weight_decay=0.0,
                  sigma_init=0.5
                  )
    print()
    # making agent
    agent = RELMAgent(model_params=model_params, solver=cmaes, n_iteration=100,
                      # name='test_mod_all',
                      # location='/Users/jitins_lab/Documents/experiment/notebooks/modelstore'
                      )

    # 1st set of data
    envr = TheEnvironment('f1_score', sample_size=0.8)

    # adding samples learnt by network initially
    envr.new_data(X[:70000], y[:70000])

    agent.execute(X[:70000,],y[:70000,])
    # evaluator = RELMEvaluator(num_episodes=15, threshold=14, is_warm=False)
    # agent = evaluator.learn(agent, envr)

    agent.plot_history()
    agent.plot_scoring(X[:70000, ], y[:70000], X[70000:, ], y[70000:, ])

    # 2nd set of data
    old_agent = agent
    agent.reset_history()
    envr.new_data(X[70000:150000], y[70000:150000])

    evaluator = RELMEvaluator(num_episodes=20, threshold=16, is_warm=True)
    agent = evaluator.learn(agent, envr)

    agent.plot_history()
    agent.plot_scoring(X[70000:150000], y[70000:150000], X[150000:, ], y[150000:, ])
    agent.plot_relm_learning(old_agent)

    # 3rd set of data
    agent.accumulate_history(old_agent)
    old_agent = agent
    agent.reset_history()
    envr.new_data(X[150000:], y[150000:])

    evaluator = RELMEvaluator(num_episodes=20, threshold=16, is_warm=True)
    agent = evaluator.learn(agent, envr)

    agent.plot_history()
    agent.plot_scoring(X[150000:], y[150000], X, y)
    agent.plot_relm_learning(old_agent)

