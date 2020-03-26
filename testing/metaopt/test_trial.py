import datetime

from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.metaopt.trial import Trial, MAIN_SCORING_METRIC_NAME, TRIAL_STATUS, TRIAL_DATETIME_STR_FORMAT, Trials

EXPECTED_ERROR_TRACEBACK = 'NoneType: None\n'

EXPECTED_METRIC_RESULTS = {
    'main': {'train_values': [0.5, 0.7, 0.4], 'validation_values': [0.5, 0.7, 0.4], 'higher_score_is_better': False}}


def test_trial_should_create_new_split():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)

    with trial.new_validation_split() as trial_split:
        trial_split.set_success()

    assert isinstance(trial_split.start_time, datetime.datetime)
    assert isinstance(trial_split.end_time, datetime.datetime)
    assert trial_split.start_time < trial_split.end_time
    assert trial.validation_splits[0] == trial_split


def test_trial_split_is_new_best_score_should_return_true_with_one_score():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)

    with trial.new_validation_split() as trial_split:
        trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.5, higher_score_is_better=False)
        trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.5,
                                                  higher_score_is_better=False)

    assert trial_split.is_new_best_score()


def test_trial_split_is_new_best_score_should_return_false_with_not_a_new_best_score():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)

    with trial.new_validation_split() as trial_split:
        trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.5, higher_score_is_better=False)
        trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.5,
                                                  higher_score_is_better=False)

        trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.7, higher_score_is_better=False)
        trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.7,
                                                  higher_score_is_better=False)

    assert not trial_split.is_new_best_score()


def test_trial_split_is_new_best_score_should_return_true_with_a_new_best_score_after_multiple_scores():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)

    with trial.new_validation_split() as trial_split:
        trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.5, higher_score_is_better=False)
        trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.5,
                                                  higher_score_is_better=False)

        trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.7, higher_score_is_better=False)
        trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.7,
                                                  higher_score_is_better=False)

        trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.4, higher_score_is_better=False)
        trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.4,
                                                  higher_score_is_better=False)

    assert trial_split.is_new_best_score()


def test_success_trial_split_to_json():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)

    trial_split = given_success_trial_validation_split(trial)

    trial_json = trial_split.to_json()

    then_success_trial_split_json_is_valid(trial_json)


def then_success_trial_split_json_is_valid(trial_json):
    assert trial_json['status'] == TRIAL_STATUS.SUCCESS.value
    assert trial_json['error'] is None
    assert trial_json['error_traceback'] is None
    assert trial_json['metric_results'] == EXPECTED_METRIC_RESULTS
    start_time = datetime.datetime.strptime(trial_json['start_time'], TRIAL_DATETIME_STR_FORMAT)
    end_time = datetime.datetime.strptime(trial_json['end_time'], TRIAL_DATETIME_STR_FORMAT) + datetime.timedelta(
        hours=1)
    assert start_time < end_time

    return True


def test_success_trial_to_json():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)

    given_success_trial_validation_split(trial)

    trial_json = trial.to_json()

    assert trial_json['status'] == TRIAL_STATUS.SUCCESS.value
    assert trial_json['error'] is None
    assert trial_json['error_traceback'] is None
    assert then_success_trial_split_json_is_valid(trial_json['validation_splits'][0])

    start_time = datetime.datetime.strptime(trial_json['start_time'], TRIAL_DATETIME_STR_FORMAT)
    end_time = datetime.datetime.strptime(trial_json['end_time'], TRIAL_DATETIME_STR_FORMAT) + datetime.timedelta(
        hours=1)

    assert start_time < end_time


def given_success_trial_validation_split(trial):
    with trial:
        with trial.new_validation_split() as trial_split:
            trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.5, higher_score_is_better=False)
            trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.5,
                                                      higher_score_is_better=False)

            trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.7, higher_score_is_better=False)
            trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.7,
                                                      higher_score_is_better=False)

            trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.4, higher_score_is_better=False)
            trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.4,
                                                      higher_score_is_better=False)
            trial_split.set_success()
            trial.set_success()

    return trial_split


def test_failure_trial_split_to_json():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)
    trial_split = given_failed_trial_split(trial)

    trial_json = trial_split.to_json()

    then_failed_validation_split_json_is_valid(trial_json, trial_split)


def then_failed_validation_split_json_is_valid(trial_json, trial_split):
    assert trial_json['status'] == TRIAL_STATUS.FAILED.value
    assert trial_json['error'] == str(trial_split.error)
    assert trial_json['error_traceback'] == EXPECTED_ERROR_TRACEBACK
    assert trial_json['metric_results'] == EXPECTED_METRIC_RESULTS
    start_time = datetime.datetime.strptime(trial_json['start_time'], TRIAL_DATETIME_STR_FORMAT)
    end_time = datetime.datetime.strptime(trial_json['end_time'], TRIAL_DATETIME_STR_FORMAT) + datetime.timedelta(
        hours=1)
    assert start_time < end_time
    return True


def test_failure_trial_to_json():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)
    trial_split = given_failed_trial_split(trial)

    trial_json = trial.to_json()

    assert trial_json['status'] == TRIAL_STATUS.FAILED.value
    assert trial_json['error'] == str(trial_split.error)
    assert trial_json['error_traceback'] == EXPECTED_ERROR_TRACEBACK
    assert then_failed_validation_split_json_is_valid(trial_json['validation_splits'][0], trial_split=trial_split)

    start_time = datetime.datetime.strptime(trial_json['start_time'], TRIAL_DATETIME_STR_FORMAT)
    end_time = datetime.datetime.strptime(trial_json['end_time'], TRIAL_DATETIME_STR_FORMAT) + datetime.timedelta(
        hours=1)

    assert start_time < end_time


def given_failed_trial_split(trial):
    with trial:
        with trial.new_validation_split() as trial_split:
            trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.5, higher_score_is_better=False)
            trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.5,
                                                      higher_score_is_better=False)

            trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.7, higher_score_is_better=False)
            trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.7,
                                                      higher_score_is_better=False)

            trial_split.add_metric_results_train(name=MAIN_SCORING_METRIC_NAME, score=0.4, higher_score_is_better=False)
            trial_split.add_metric_results_validation(name=MAIN_SCORING_METRIC_NAME, score=0.4,
                                                      higher_score_is_better=False)
            error = IndexError('index error')
            trial_split.set_failed(error)
            trial.set_failed(error)
    return trial_split


def test_trials_get_best_hyperparams():
    hp = HyperparameterSamples({'a': 2})
    trial = Trial(hyperparams=hp)
    trial_split = given_success_trial_validation_split(trial)
    trials = Trials(trials=[trial])

    best_hyperparams = trials.get_best_hyperparams()

