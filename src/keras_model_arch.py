import keras

SEED = 100
PARAMS_MODEL = {
    # layer 1
    'L1_units': 16,
    'L1_act': 'relu',
    'L1_dropout': 0.5,

    # optimizer
    'adam_lr': 1e-3,

    # bias initializer
    'use_bias_init_last_layer': True,
}
METRICS = 'auc'
metrics = METRICS

params = PARAMS_MODEL
output_bias = None
n_feats = 29 # amount and v1-v28 (time is not used, class is target)


def get_model():

    # num of layers
    n_layers = len([i for i in list(params.keys()) if i.endswith('_units')])

    # layers
    model = keras.Sequential(name='Sequential')

    # layer 1
    model.add(keras.layers.Dense(
        params['L1_units'],
        activation=params['L1_act'],
        input_shape=(n_feats,),
        name='Layer_1'
        ))
    model.add(keras.layers.Dropout(params['L1_dropout'],
                                seed=SEED,name='Dropout_1'))

    # middle layers
    for i in range(2,n_layers+1): # 2,3, etc
        model.add(keras.layers.Dense(
            params['L{}_units'.format(i)],
            activation=params['L{i}_act'.format(i)],
            name='Layer_{}'.format(i)),)
        model.add(keras.layers.Dropout(
            params['L{}_dropout'.format(i)],
            seed=SEED,
            name="Dropout_{}".format(i)),)

    # last layer is dense 1 with activation sigmoid
    model.add(keras.layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer=output_bias,
        name='Layer_{}'.format(n_layers+1)
        ))

    #=================================================== compile
    model.compile(
        optimizer=keras.optimizers.Adam(lr=params['adam_lr']),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

if __name__ == '__main__':
    # run the program
    model = get_model()
    print(model.summary())