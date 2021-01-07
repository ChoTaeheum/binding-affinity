



bayes_optimizer = BayesianOptimization(
    f=train_and_validate,
    pbounds={
        'init_learning_rate_log': (-5, -1),    # FIXME
        'weight_decay_log': (-5, -1)            # FIXME
        'eps': 
    },
    random_state=0,
    verbose=2
)

bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei', xi=0.01)    # FIXME

for i, res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print('Final result: ', bayes_optimizer.max)