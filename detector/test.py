from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import json
import numpy as np
import sklearn
import fire

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

import data_loading
import json

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def eer(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(f'Threshold: {thresh}')
    return eer

# Which data sources to test on.
sources_test = [
    "generator=mega~dataset=p0.90",
    "generator=mega~dataset=p0.92",
    "generator=mega~dataset=p0.94",
    "generator=mega~dataset=p0.96",
    "generator=mega~dataset=p0.98",
    "generator=mega~dataset=p1.00",
    "xl-1542M",
    "xl-1542M-k40",
    "xl-1542M-nucleus",
    'gpt2-md-covid-random',
    'gpt2-md-covid-topk40',
    'gpt2-md-covid-topp96',
]

def main (
        outputs_base,
        source_train=None,
        data_dir='gentext_data',
        n_test=np.inf,
        reprocess_input=True,
        n_gpu=1,
        eval_batch_size=64,
        max_seq_length=256,
    ):
    for source_test in sources_test:
        print('trained on:', source_train, 'tested_on:', source_test)
        
        test_df = data_loading.load_split(data_dir, source_test, 'valid', n=n_test)

        # Optional model configuration
        model_args = ClassificationArgs(
            output_dir=f"outputs/cross_decoding/{source_train}_{source_test}",
            manual_seed=0,
            eval_batch_size=eval_batch_size,
            max_seq_length=max_seq_length,
            overwrite_output_dir=True,
            reprocess_input_data=reprocess_input,
            n_gpu=n_gpu,
            no_cache=True,
        )

        # Create a ClassificationModel
        model = ClassificationModel(
            "roberta",
            f"{outputs_base}/{source_train}_best_model",
            args=model_args,
            use_cuda=True
        )

        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(
            test_df,
            f1=sklearn.metrics.f1_score,
            acc=sklearn.metrics.accuracy_score,
            eer=eer
        )

if __name__ == '__main__':
    fire.Fire(main)
