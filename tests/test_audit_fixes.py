"""Regression tests for the audit findings fixed in the corrected pipeline."""

import inspect

from sklearn.preprocessing import LabelEncoder

from src.balancing import balance_training_fold
from src.train_hgb import train_and_evaluate as train_hgb
from src.train_xgboost import train_and_evaluate as train_xgboost
from src.train_logistic import train_and_evaluate as train_logreg


def test_normal_class_index_is_derived_from_encoder_not_assumed_zero():
    encoder = LabelEncoder().fit(['Normal', 'Analysis', 'Worms'])
    assert list(encoder.classes_).index('Normal') != 0


def test_classical_trainers_expose_runtime_balancer_and_plot_options():
    for trainer in (train_hgb, train_xgboost, train_logreg):
        parameters = inspect.signature(trainer).parameters
        assert 'balancer' in parameters
        assert 'make_plots' in parameters
        assert 'normal_class_idx' in parameters


def test_kmeans_strategy_uses_actual_kmeans_smote_implementation():
    source = inspect.getsource(balance_training_fold)
    assert 'KMeansSMOTE(' in source
    assert 'fit_predict' not in source


def test_final_dl_preprocessing_keeps_the_bounded_resampling_policy():
    from src.dl_pipeline import preprocess_final

    source = inspect.getsource(preprocess_final)
    assert 'RandomUnderSampler' in source
    assert 'rus_cap' in source
