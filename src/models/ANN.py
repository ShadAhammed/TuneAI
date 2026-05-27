"""Artificial Neural Network with automated hyperparameter tuning.

Uses Keras Tuner (Hyperband by default) to find the best architecture
across three hidden layers.  The search explores layer widths, activation
functions, dropout rates, and the Adam learning rate simultaneously.

Tested with Keras 3.x + TensorFlow 2.21.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt


class ANN:
    """Builds, tunes, and fits a binary-classification ANN.

    Parameters
    ----------
    X_train: feature matrix used for both tuning and final training.
    y_train: binary target labels corresponding to X_train.
    """

    QUICK_MODE = False

    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def _build_model(self, hp: kt.HyperParameters) -> keras.Model:
        """Construct and compile a model given a HyperParameters object.

        The architecture has three hidden layers whose widths, activations,
        and dropout rates are all subject to search.
        """
        model = Sequential()
        model.add(keras.Input(shape=(self.X_train.shape[1],)))

        layer_configs = [
            ('units_layer_1', 10, 30, 'activation_1'),
            ('units_layer_2',  6, 24, 'activation_2'),
            ('units_layer_3',  4, 16, 'activation_3'),
        ]
        dropout_keys = ['dropout_rate_1', 'dropout_rate_2', 'dropout_rate_3']

        for (units_key, lo, hi, act_key), drop_key in zip(layer_configs, dropout_keys):
            units = hp.Int(units_key, min_value=lo, max_value=hi, step=1)
            activation = hp.Choice(act_key, ['relu', 'sigmoid', 'tanh'])
            dropout_rate = hp.Float(drop_key, min_value=0.0, max_value=0.5, step=0.1)
            model.add(Dense(units=units, activation=activation))
            model.add(Dropout(rate=dropout_rate))

        model.add(Dense(1, activation='sigmoid'))

        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return model

    def _tune(self, tuner_id: int) -> kt.Tuner:
        """Create and run a Keras Tuner search, returning the configured tuner.

        tuner_id legend:
            1 — RandomSearch
            2 — Hyperband  (default, fastest)
            3 — BayesianOptimization
        """
        common_kwargs = dict(
            hypermodel=self._build_model,
            objective='val_accuracy',
            overwrite=True,
            project_name='tuneai_ann',
        )

        max_trials = 3 if self.QUICK_MODE else 10
        search_epochs = 30 if self.QUICK_MODE else 80

        if tuner_id == 1:
            tuner = kt.RandomSearch(**common_kwargs, max_trials=max_trials)
        elif tuner_id == 3:
            tuner = kt.BayesianOptimization(**common_kwargs, max_trials=max_trials)
        else:
            # Hyperband is the default — good balance of speed and coverage
            max_epochs = 10 if self.QUICK_MODE else 20
            tuner = kt.Hyperband(**common_kwargs, max_epochs=max_epochs, factor=3)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        tuner.search(
            self.X_train, self.y_train,
            epochs=search_epochs,
            validation_split=0.25,
            callbacks=[early_stop],
            verbose=0,
        )
        return tuner

    def ANN_model(self, tuner_id: int = 2) -> keras.Model:
        """Return a trained ANN using the best hyperparameters found.

        The model is re-trained from scratch on the full training set for
        exactly the number of epochs that produced the best validation
        accuracy during the search.
        """
        tuner = self._tune(tuner_id)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f'\nBest ANN hyperparameters: {best_hp.values}\n')

        # Build directly from the bound method — avoids relying on
        # tuner.hypermodel.build(), which behaves differently depending on
        # whether hypermodel was a callable or a HyperModel subclass.
        fit_epochs = 30 if self.QUICK_MODE else 80
        model = self._build_model(best_hp)
        history = model.fit(
            self.X_train, self.y_train,
            epochs=fit_epochs,
            validation_split=0.2,
            verbose=0,
        )

        best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
        print(f'Best epoch identified: {best_epoch}\n')

        # Re-build a fresh model and train for exactly best_epoch epochs
        final_model = self._build_model(best_hp)
        final_model.fit(
            self.X_train, self.y_train,
            epochs=best_epoch,
            validation_split=0.2,
            verbose=0,
        )
        return final_model
