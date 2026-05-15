"""Tests for ondes.encoding."""

import math

from ondes.encoding import (
    NO_ENCODING,
    Encoding,
    dyadic,
    gaussian_fixed,
    gaussian_from_shape,
    gaussian_learn,
    nyquist_sigma,
)


def test_no_encoding_equals_kind_none_encoding():
    # Given: the module-level NO_ENCODING singleton
    # When: comparing to a freshly constructed Encoding(kind="none")
    # Then: they are equal (frozen dataclass value equality)
    assert NO_ENCODING == Encoding(kind="none")


def test_gaussian_fixed_sets_kind_and_sigma():
    # Given: a fixed-sigma factory call
    # When: building the Encoding
    # Then: kind is "gaussian" and sigma is the provided float
    enc = gaussian_fixed(2.5)
    assert enc.kind == "gaussian"
    assert enc.sigma == 2.5
    assert enc.learn_sigma is False
    assert enc.sigma_from_shape is None


def test_gaussian_from_shape_stores_rule_and_leaves_sigma_none():
    # Given: a per-leaf rule
    # When: constructing the encoding
    # Then: sigma_from_shape is the rule and sigma is None
    def rule(shape):
        return 1.0

    enc = gaussian_from_shape(rule)
    assert enc.kind == "gaussian"
    assert enc.sigma is None
    assert enc.sigma_from_shape is rule


def test_gaussian_learn_defaults_to_pi():
    # Given: gaussian_learn called with no arguments
    # When: inspecting the resulting Encoding
    # Then: sigma is pi and learn_sigma is True
    enc = gaussian_learn()
    assert enc.kind == "gaussian"
    assert enc.learn_sigma is True
    assert enc.sigma == math.pi


def test_dyadic_sets_num_bands():
    # Given: a dyadic factory call with L=6
    # When: inspecting the Encoding
    # Then: kind is "dyadic" and num_bands is 6
    enc = dyadic(L=6)
    assert enc.kind == "dyadic"
    assert enc.num_bands == 6


def test_nyquist_sigma_uses_longest_axis():
    # Given: shape (32, 1024) — longest axis is 1024
    # When: computing sigma
    # Then: sigma is (1024 - 1) / 4 = 255.75
    assert nyquist_sigma((32, 1024)) == (1024 - 1) / 4


def test_nyquist_sigma_handles_4d_conv_kernel():
    # Given: shape (3, 3, 1, 16) — longest axis is 16
    # When: computing sigma
    # Then: sigma is max(1, (16 - 1) / 4) = 3.75
    assert nyquist_sigma((3, 3, 1, 16)) == max(1.0, (16 - 1) / 4)


def test_nyquist_sigma_floor_is_one():
    # Given: a degenerate shape (1,) where (N-1)/4 = 0
    # When: computing sigma
    # Then: the 1.0 floor kicks in
    assert nyquist_sigma((1,)) == 1.0
