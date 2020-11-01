import numpy as np
import pandas as pd

from sklearn import metrics
import config

def print_time_taken(time_taken):

    h,m = divmod(time_taken,60*60)
    m,s = divmod(m,60)
    time_taken = f"{h:.0f} h {m:.0f} min {s:.2f} sec" if h > 0 else f"{m:.0f} min {s:.2f} sec"
    time_taken = f"{m:.0f} min {s:.2f} sec" if m > 0 else f"{s:.2f} sec"

    print(f"\nTime Taken: {time_taken}")

def get_model(params,metrics,n_feats,output_bias=None):
    import keras

    # use initial bias for imbalanced data
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

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
                                seed=config.SEED,name='Dropout_1'))

    # middle layers
    for i in range(2,n_layers+1): # 2,3, etc
        model.add(keras.layers.Dense(
            params[f'L{i}_units'],
            activation=params[f'L{i}_act'],
            name=f'Layer_{i}'),)
        model.add(keras.layers.Dropout(
            params[f'L{i}_dropout'],
            seed=SEED,
            name=f"Dropout_{i}"))

    # last layer is dense 1 with activation sigmoid
    model.add(keras.layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer=output_bias,
        name=f'Layer_{n_layers+1}'
        ))

    #=================================================== compile
    model.compile(
        optimizer=keras.optimizers.Adam(lr=params['adam_lr']),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model
