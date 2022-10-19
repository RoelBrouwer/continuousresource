from continuousresource.localsearch.utils import \
    sanitize_simulated_annealing_params, \
    sanitize_search_space_params

import pytest
import warnings

default_sa_params = {
    "initial_temperature": 100.0,
    "alfa": 0.95,
    "alfa_period": 100,
    "cutoff": (lambda a: a * 50)
}
default_ssp_params = {
    "infer_precedence": False,
    "fracs": {
        "swap": 0.95,
        "move": 0.025,
        "movepair": 0.025
    },
    "start_solution": "greedy"
}


def test_sanitize_simulated_annealing_params_null():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Call parameter sanitization
        params = sanitize_simulated_annealing_params(None)
        defaults = default_sa_params
        defaults["cutoff"] = defaults["cutoff"](defaults["alfa_period"])

        # Verify that a warning is triggered for each parameter
        assert len(w) == len(default_sa_params)
        for warning in w:
            assert issubclass(warning.category, RuntimeWarning)
            assert "Using default" in str(warning.message)

        assert len(defaults) == len(params)
        for key, value in params.items():
            assert defaults[key] == value


@pytest.mark.parametrize(
    'input_params,expected,nr_warnings',
    [
        ({
            "initial_temperature": 10.0,
            "alfa": 0.99,
            "alfa_period": 10,
            "cutoff": 3000
        }, {
            "initial_temperature": 10.0,
            "alfa": 0.99,
            "alfa_period": 10,
            "cutoff": 3000
        }, 0)
    ]
)
def test_sanitize_simulated_annealing_params(input_params, expected,
                                             nr_warnings):
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Call parameter sanitization
        params = sanitize_simulated_annealing_params(input_params)

        # Verify that a warning is triggered for each invalid parameter
        assert len(w) == nr_warnings
        for warning in w:
            assert issubclass(warning.category, RuntimeWarning)
            assert "Using default" in str(warning.message)

        assert len(expected) == len(params)
        for key, value in params.items():
            assert expected[key] == value


@pytest.mark.parametrize(
    'input_params,expected,nr_warnings',
    [
        (None, default_ssp_params, 3),
        ({
            "infer_precedence": True,
            "fracs": {
                "swap": 0.9,
                "move": 0.025,
                "movepair": 0.075
            },
            "start_solution": "random"
        }, {
            "infer_precedence": True,
            "fracs": {
                "swap": 0.9,
                "move": 0.025,
                "movepair": 0.075
            },
            "start_solution": "random"
        }, 0),
        ({
            "infer_precedence": False,
            "fracs": {
                "swap": 0.9,
                "move": 0.025,
                "movepair": 0.025
            },
            "start_solution": "greedy"
        }, {
            "infer_precedence": False,
            "fracs": {
                "swap": 0.95,
                "move": 0.025,
                "movepair": 0.025
            },
            "start_solution": "greedy"
        }, 1),
        ({
            "infer_precedence": False,
            "fracs": {
                "swap": 0.95,
                "move": 0.05
            },
            "start_solution": "greedy"
        }, {
            "infer_precedence": False,
            "fracs": {
                "swap": 0.95,
                "move": 0.025,
                "movepair": 0.025
            },
            "start_solution": "greedy"
        }, 1)
    ]
)
def test_sanitize_search_space_params(input_params, expected, nr_warnings):
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Call parameter sanitization
        params = sanitize_search_space_params(input_params)

        # Verify that a warning is triggered for each parameter
        assert len(w) == nr_warnings
        for warning in w:
            assert issubclass(warning.category, RuntimeWarning)
            assert "Using default" in str(warning.message)

        assert len(expected) == len(params)
        for key, value in params.items():
            assert expected[key] == value
