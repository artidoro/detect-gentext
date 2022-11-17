from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import json
import numpy as np
import sklearn
import fire
import math
import os


from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

import data_loading

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def eer(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(f'Threshold: {thresh}')
    return eer

model_type_dict = {
    "roberta-large": 'roberta',
    "roberta-base": 'roberta',
    "bert-large-cased": 'bert',
    "bert-base-cased": 'bert',
    "google/electra-large-discriminator": 'electra',
    "google/electra-small-discriminator": 'electra',
    "albert-xxlarge-v2": 'albert',
    "albert-base-v2": 'albert',
}

def rec_remove(path):
    dirs = os.listdir(path)
    dirs = [os.path.join(path, dir) for dir in dirs]
    for dir in dirs:
        if 'best_model' in dir:
            continue
        elif '.bin' in dir or '.pt' in dir:
            os.remove(dir)
            print(f'removed {dir}')
        else:
            if os.path.isdir(dir):
                rec_remove(dir)

def main (
        source,
        checkpoint_dir='outputs',
        best_model_dir=None,
        data_dir='gentext_data',
        model_name='roberta-large',
        n_train=1000000,
        n_valid=1000000,
        n_test=1000000,
        n_epochs=10,
        learning_rate=1e-06,
        train_batch_size=64,
        eval_batch_size=64,
        max_seq_length=256,
        evaluate_during_training=True,
        evaluate_during_training_steps=-1, # -1 = once per epoch
        evaluate_during_training_verbose=True,
        reprocess_input=True,
        overwrite_output_dir=True,
        n_gpu=1
    ):
    if best_model_dir is None:
        best_model_dir = f'{checkpoint_dir}_best_model'

    train_df = data_loading.load_split(data_dir, source, 'train', n=n_train)
    valid_df = data_loading.load_split(data_dir, source, 'valid', n=n_valid)
    test_df = data_loading.load_split(data_dir, source, 'test', n=n_test)

    # evaluate_during_training_steps = math.floor(len(train_df) / train_batch_size / 4) # four times per epoch

    model_type = model_type_dict[model_name]
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        checkpoints = sorted([dir for dir in os.listdir(checkpoint_dir) if 'checkpoint' in dir], key=lambda x: int(x.split('-')[1]))
        model_name = os.path.join(checkpoint_dir, checkpoints[-1])
        for checkpoint in checkpoints[:-1]:
            rec_remove(os.path.join(checkpoint_dir, checkpoint))          
        print(f'WARNING: resuming training from {model_name} among')
        print(checkpoints)

    # Optional model configuration
    model_args = ClassificationArgs(
        num_train_epochs=n_epochs,
        evaluate_during_training=evaluate_during_training,
        evaluate_during_training_steps=evaluate_during_training_steps,
        evaluate_during_training_verbose=evaluate_during_training_verbose,
        best_model_dir=best_model_dir,
        manual_seed=0,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        overwrite_output_dir=overwrite_output_dir,
        n_gpu=n_gpu,
        output_dir=checkpoint_dir,
        reprocess_input_data=reprocess_input,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
    )

    # Create a ClassificationModel
    model = ClassificationModel(
        model_type=model_type,
        model_name=model_name,
        args=model_args,
        use_cuda=True
    )

    # Train the model
    model.train_model(
        train_df,
        eval_df=valid_df,
        f1=sklearn.metrics.f1_score,
        acc=sklearn.metrics.accuracy_score,
        eer=eer
        )

    # remove checkpoints only keeping best model:
    rec_remove(checkpoint_dir)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df,
        f1=sklearn.metrics.f1_score,
        acc=sklearn.metrics.accuracy_score,
        eer=eer
    )

if __name__ == '__main__':
    fire.Fire(main)