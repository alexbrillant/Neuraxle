"""
Tree parzen estimator
====================================
Code for tree parzen estimator auto ml.
"""
import numpy as np
from typing import Optional
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.hyperparams.distributions import DistributionMixture
from .auto_ml import BaseHyperparameterOptimizer, RandomSearchHyperparameterOptimizer, TRIAL_STATUS


def _linear_forgetting_Weights(number_samples, number_recent_trial_at_full_weights):
    """This part of code has been taken from Hyperopt (https://github.com/hyperopt) code."""
    if number_samples == 0:
        return np.asarray([])

    if number_samples < number_recent_trial_at_full_weights:
        return np.ones(number_samples)

    weights_ramp = np.linspace(1.0 / number_samples, 1.0, number_samples - number_recent_trial_at_full_weights)
    weights_flat = np.ones(number_recent_trial_at_full_weights)
    weights = np.concatenate((weights_ramp, weights_flat), axis=0)

    return weights


class TreeParzenEstimatorHyperparameterOptimizer(BaseHyperparameterOptimizer):

    def __init__(self, number_of_initial_random_step: int = 40, quantile_threshold: float = 30,
                 number_possible_hyperparams_candidates=100,
                 prior_weight: float = 0.,
                 use_linear_forgetting_weights: bool = False,
                 number_recent_trial_at_full_weights=25):
        super().__init__()
        self.initial_auto_ml_algo = RandomSearchHyperparameterOptimizer()
        self.number_of_initial_random_step = number_of_initial_random_step
        self.quantile_threshold = quantile_threshold
        self.number_possible_hyperparams_candidates = number_possible_hyperparams_candidates
        self.prior_weight = prior_weight
        self.use_linear_forgetting_weights = use_linear_forgetting_weights
        self.number_recent_trial_at_full_weights = number_recent_trial_at_full_weights

    def _split_trials(self, success_trials):
        # Split trials into good and bad using quantile threshold.
        # TODO: maybe better place in the Trials class.
        trials_scores = np.array([trial.score for trial in success_trials])

        # TODO: do we want to clip the number of good trials like in hyperopt.
        percentile_thresholds = np.percentile(trials_scores, self.quantile_threshold)

        # In hyperopt they use this to split, where default_gamma_cap = 25. They clip the max of item they use in the good item.
        # default_gamma_cap is link to the number of recent_trial_at_full_weight also.
        # n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)

        good_trials = []
        bad_trials = []
        for trial in success_trials:
            if trial.score < percentile_thresholds:
                good_trials.append(trial)
            else:
                bad_trials.append(trial)
        return good_trials, bad_trials

    def _create_posterior(self, flat_hyperparameter_space, trials):
        # Create a list of all hyperparams and their trials.

        # Loop through all hyperparams
        posterior_distributions = {}
        for (hyperparam_key, hyperparam_distribution) in flat_hyperparameter_space.items():

            # Get distribution_trials
            distribution_trials = [trial.hyperparams.to_flat_as_dict_primitive()[hyperparam_key] for trial in trials]

            # TODO : create a discret and uniform class in order to be able to discretize them.
            if isinstance(hyperparam_distribution, discret_distribution):
                # If hyperparams is a discret distribution
                posterior_distribution = self._reweights_categorical(hyperparam_distribution, distribution_trials)

            else:
                # If hyperparams is a continuous distribution
                posterior_distribution = self._create_gaussian_mixture(hyperparam_distribution, distribution_trials)

            posterior_distributions[hyperparam_key] = posterior_distribution
        return posterior_distributions

    def _reweights_categorical(self, hyperparam_distribution, distribution_trials):

    # For discret categorical distribution
    # We need to reweights probability depending on trial counts.
    # TODO: need to add a way to access a list of all probabilities and a list of all values.

    def _create_gaussian_mixture(self, hyperparam_distribution, distribution_trials):

        # TODO: add Condition if log distribution or not.
        use_logs = False

        # TODO: add condition if quantized distribution.
        use_quantized_distributions = False

        # Find means, std, amplitudes, min and max.
        distribution_amplitudes, means, stds, distributions_mins, distributions_max = self._adaptive_parzen_normal(
            hyperparam_distribution,
            distribution_trials)

        # Create appropriate gaussian mixture that wrapped all hyperparams.
        gmm = DistributionMixture.build_gaussian_mixture(distribution_amplitudes, means=means, stds=stds,
                                                         distributions_mins=distributions_mins,
                                                         distributions_max=distributions_max, use_logs=use_logs,
                                                         use_quantized_distributions=use_quantized_distributions)

        return gmm

    def _adaptive_parzen_normal(self, hyperparam_distribution, distribution_trials):
        """This part of code is enterily inspire from Hyperopt (https://github.com/hyperopt) code."""

        use_prior = (self.prior_weight - 0.) > 1e-10

        prior_mean = hyperparam_distribution.mean()
        prior_sigma = hyperparam_distribution.std()

        means = distribution_trials
        distributions_mins = hyperparam_distribution.min() * len(means)
        distributions_max = hyperparam_distribution.max() * len(means)

        # Index to sort in increasing order the means.
        # Easier in order to insert prior.
        sort_indexes = np.argsort(means)

        if len(means) == 0:
            if use_prior:
                prior_pos = 0
                sorted_means = np.array([prior_mean])
                sorted_stds = np.array([prior_sigma])
        elif len(means) == 1:
            if use_prior and prior_mean < means[0]:
                prior_pos = 0
                sorted_means = np.array([prior_mean, means[0]])
                sorted_stds = np.array([prior_sigma, prior_sigma * 0.5])
            elif use_prior and prior_mean >= means[0]:
                prior_pos = 1
                sorted_means = np.array([means[0], prior_mean])
                sorted_stds = np.array([prior_sigma * 0.5, prior_sigma])
            else:
                sorted_means = means
                sorted_stds = prior_sigma
        else:
            if use_prior:
                # Insert the prior at the right place.
                prior_pos = np.searchsorted(means[sort_indexes], prior_mean)
                sorted_means = np.zeros(len(means) + 1)
                sorted_means[:prior_pos] = means[sort_indexes[:prior_pos]]
                sorted_means[prior_pos] = prior_mean
                sorted_means[prior_pos + 1:] = means[sort_indexes[prior_pos:]]
            else:
                sorted_means = means[sort_indexes]

            sorted_stds = np.zeros_like(sorted_means)
            sorted_stds[1:-1] = np.maximum(sorted_means[1:-1] - sorted_means[0:-2],
                                           sorted_means[2:] - sorted_means[1:-1])
            left_std = sorted_means[1] - sorted_means[0]
            right_std = sorted_means[-1] - sorted_means[-2]
            sorted_stds[0] = left_std
            sorted_stds[-1] = right_std

        # Magic formula from hyperopt.
        # -- magic formula:
        # maxsigma = old_div(prior_sigma, 1.0)
        # minsigma = old_div(prior_sigma, min(100.0, (1.0 + len(srtd_mus))))
        #
        # sigma = np.clip(sigma, minsigma, maxsigma)
        # sigma[prior_pos] = prior_sigma
        min_std = prior_sigma / min(100.0, (1.0 + len(sorted_means)))
        max_std = prior_sigma / 1.0
        sorted_stds = np.clip(sorted_stds, min_std, max_std)

        if self.use_linear_forgetting_weights:
            distribution_amplitudes = _linear_forgetting_Weights(len(means), self.number_recent_trial_at_full_weights)
        else:
            # From tpe article : TPE substitutes an equally-weighted mixture of that prior with Gaussians centered at each observations.
            distribution_amplitudes = np.ones(len(means))
            # distribution_amplitudes = [hyperparam_distribution.pdf(mean) for mean in means]

        if use_prior:
            sorted_stds[prior_pos] = prior_sigma
            sorted_distribution_amplitudes = np.zeros_like(sorted_means)
            sorted_distribution_amplitudes[:prior_pos] = distribution_amplitudes[sort_indexes[:prior_pos]]
            sorted_distribution_amplitudes[prior_pos] = sort_indexes
            sorted_distribution_amplitudes[prior_pos + 1:] = distribution_amplitudes[sort_indexes[prior_pos:]]
        else:
            sorted_distribution_amplitudes = distribution_amplitudes

        # Normalize distribution amplitudes.
        distribution_amplitudes = np.array(distribution_amplitudes)
        distribution_amplitudes /= np.sum(distribution_amplitudes)

        return sorted_distribution_amplitudes, sorted_means, sorted_stds, distributions_mins, distributions_max

    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :type auto_ml_container: Trials
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        # Flatten hyperparameter space
        flat_hyperparameter_space = auto_ml_container.hyperparameter_space.to_flat()

        if auto_ml_container.trial_number < self.number_of_initial_random_step:
            # Perform random search
            return self.initial_auto_ml_algo.find_next_best_hyperparams(auto_ml_container)

        # Keep only success trials
        success_trials = auto_ml_container.trials.filter(TRIAL_STATUS.SUCCESS)

        # Split trials into good and bad using quantile threshold.
        good_trials, bad_trials = self._split_trials(success_trials)

        # Create gaussian mixture of good and gaussian mixture of bads.
        good_posteriors = self._create_posterior(flat_hyperparameter_space, good_trials)
        bad_posteriors = self._create_posterior(flat_hyperparameter_space, bad_trials)

        best_hyperparams = []
        for (hyperparam_key, good_posterior) in good_posteriors.items():
            best_new_hyperparam_value = None
            best_ratio = None
            for _ in range(self.number_possible_hyperparams_candidates):
                # Sample possible new hyperparams in the good_trials.
                possible_new_hyperparm = good_posterior.rvs()

                # Verify if we use the ratio directly or we use the loglikelihood of b_post under both distribution like hyperopt.
                # In hyperopt they use :
                # # calculate the log likelihood of b_post under both distributions
                # below_llik = fn_lpdf(*([b_post] + b_post.pos_args), **b_kwargs)
                # above_llik = fn_lpdf(*([b_post] + a_post.pos_args), **a_kwargs)
                #
                # # improvement = below_llik - above_llik
                # # new_node = scope.broadcast_best(b_post, improvement)
                # new_node = scope.broadcast_best(b_post, below_llik, above_llik)

                # Verify ratio good pdf versus bad pdf for all possible new hyperparms.
                # Used what is describe in the article which is the ratio g(x) / l(x) that we have to maximize.
                # Only the best ratio is kept and is the new best hyperparams.
                ratio = bad_posteriors[hyperparam_key].pdf(possible_new_hyperparm) / good_posterior.pdf(
                    possible_new_hyperparm)

                if best_new_hyperparam_value is None:
                    best_new_hyperparam_value = possible_new_hyperparm
                    best_ratio = ratio
                else:
                    if ratio > best_ratio:
                        best_new_hyperparam_value = possible_new_hyperparm
                        best_ratio[hyperparam_key] = ratio

            best_hyperparams.append((hyperparam_key, best_new_hyperparam_value))
        return HyperparameterSamples(best_hyperparams)
