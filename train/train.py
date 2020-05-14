from sklearn.preprocessing import RobustScaler
from keras.layers import Dropout, Dense
import sagemaker_containers
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import pickle
import keras
import json
import sys
import os

def _get_train_data(training_dir):
    '''Retrieve training data'''
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"))

    X_train = train_data.drop(['Class'], axis=1)
    y_train = train_data['Class']

    return X_train, y_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=1000, metavar='B',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--learning-rate', type=float, default=.001, metavar='L',
                        help='leaning rate (default: 0.001)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    # Get training data
    print('Reading training data...', end='', flush=True)
    X_train, y_train = _get_train_data(args.data_dir)
    print('complete')

    # Train model
    print('Training model...')
    rs = RobustScaler()
    X_train_scaled = rs.fit_transform(X_train)

    model = keras.models.Sequential([
        Dense(X_train_scaled.shape[1], 
            input_dim=X_train_scaled.shape[1],
            activation='relu'),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        keras.optimizers.Adam(lr=args.learning_rate), 
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC()]
    )

    model.fit(X_train_sm, y_train_sm,
        validation_split=0.2,
        batch_size=args.batch_size,
        epochs=args.epochs, 
        verbose=1,
    )