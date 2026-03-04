"""Shared test fixtures for xrag."""

import pytest


@pytest.fixture
def sample_opinion():
    """A simple SL opinion for testing."""
    from jsonld_ex.confidence_algebra import Opinion

    return Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)


@pytest.fixture
def vacuous_opinion():
    """A vacuous (maximum ignorance) opinion."""
    from jsonld_ex.confidence_algebra import Opinion

    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


@pytest.fixture
def dogmatic_opinion():
    """A dogmatic (full belief) opinion."""
    from jsonld_ex.confidence_algebra import Opinion

    return Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)


@pytest.fixture
def conflicting_opinions():
    """A pair of opinions that strongly disagree."""
    from jsonld_ex.confidence_algebra import Opinion

    return [
        Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05, base_rate=0.5),
        Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05, base_rate=0.5),
    ]


@pytest.fixture
def agreeing_opinions():
    """A set of opinions that mostly agree."""
    from jsonld_ex.confidence_algebra import Opinion

    return [
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5),
        Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17, base_rate=0.5),
    ]
