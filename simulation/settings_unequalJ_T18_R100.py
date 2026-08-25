from __future__ import annotations

N_VALUES = [100]
T_VALUES = [18]
J_VECTOR_VALUES = [[50, 60, 70, 80, 90, 100] * 3]

H_VALUES = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
C_LAMBDA_VALUES = [i * 0.1 for i in range(1, 21)]

R = 100
CV_FOLDS = 5
CV_JOBS = 8
BASE_SEED = 20260819
PROPOSED_THETA_PERTURBATION = 0.10

DATA_KWARGS = {
    "score_power": 1.0,
    "discrimination_low": 0.50,
    "discrimination_high": 1.50,
    "easiness_low": -1.00,
    "easiness_high": 1.00,
    "heterogeneity_structure": "random_alpha",
    "majority_proportion": 0.60,
    "minimum_relative_angle": 0.35,
    "theta_bound": 4.0,
    "shuffle_ai_labels": True,
    "population_common_trait_max_iter": 5000,
    "population_common_trait_tolerance": 1e-13,
    "shared_h_values": tuple(H_VALUES),
}

PROPOSED_CV_FIT_KWARGS = {
    "theta_bound": 4.0,
    "a_bound": 4.0,
    "d_bound": 4.0,
    "learning_rate": 0.02,
    "learning_rate_decay_factor": 0.5,
    "learning_rate_decay_steps": 400,
    "max_iter": 4000,
    "tolerance": 1e-9,
    "verbose": False,
}

LOCAL_FIT_KWARGS = {
    "theta_bound": 4.0,
    "a_bound": 4.0,
    "d_bound": 4.0,
    "learning_rate": 0.02,
    "learning_rate_decay_factor": 0.5,
    "learning_rate_decay_steps": 500,
    "max_iter": 2000,
    "tolerance": 1e-9,
    "verbose": False,
}

GLOBAL_FIT_KWARGS = {
    "theta_bound": 4.0,
    "a_bound": 4.0,
    "d_bound": 4.0,
    "learning_rate": 0.02,
    "learning_rate_decay_factor": 0.5,
    "learning_rate_decay_steps": 500,
    "max_iter": 5000,
    "tolerance": 1e-12,
    "verbose": False,
}

TABLE_FILENAMES = {
    "raw": "simulation_raw_results.csv",
    "domains": "simulation_domain_results.csv",
    "cv_tuning": "simulation_cv_tuning_results.csv",
    "cv_folds": "simulation_cv_fold_results.csv",
}

TOTAL_TASKS = len(H_VALUES) * R
