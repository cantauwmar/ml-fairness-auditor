"""Tests for pipeline_auditing.fairness.groups."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline_auditing.fairness.groups import (
    UnprivilegedGroups,
    ACSIncomeGroups,
)
from pipeline_auditing.fairness.auditor import FairnessAuditor


# --------------------------------------------------------------------------- #
# Abstract base class behaviour                                                 #
# --------------------------------------------------------------------------- #


def test_unprivileged_groups_cannot_be_instantiated():
    with pytest.raises(TypeError):
        UnprivilegedGroups()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated():
    class Incomplete(UnprivilegedGroups):
        pass  # missing get_groups

    with pytest.raises(TypeError):
        Incomplete()


# --------------------------------------------------------------------------- #
# Concrete subclass / callable contract                                        #
# --------------------------------------------------------------------------- #


def test_concrete_subclass_is_callable():
    instance = ACSIncomeGroups()
    assert callable(instance)


def test_call_equals_get_groups():
    instance = ACSIncomeGroups()
    assert instance() == instance.get_groups()


def test_acs_groups_native_country_is_non_us_list():
    groups = ACSIncomeGroups().get_groups()
    nc = groups["native-country"]
    assert isinstance(nc, list)
    assert 1.0 not in nc
    assert 13.0 in nc  # Mexico should be in the non-US list


def test_custom_subclass_works():
    class MyGroups(UnprivilegedGroups):
        def get_groups(self):
            return {"gender": "female"}

    g = MyGroups()
    assert g() == {"gender": "female"}
    assert g.get_groups() == {"gender": "female"}


# --------------------------------------------------------------------------- #
# Integration: callable instance produces same audit results as raw dict       #
# --------------------------------------------------------------------------- #


def test_callable_instance_matches_raw_dict(binary_matched_df):
    auditor = FairnessAuditor(positive_label=1)
    attrs = ["sex"]

    raw_dict = {
        "sex": 2.0,
        "race": 2.0,
        "native-country": [13.0],
        "marital-status": 5.0,
    }

    report_dict = auditor.audit(
        data=binary_matched_df,
        protected_attributes=attrs,
        pipeline_name="test_dict",
        unprivileged_groups=raw_dict,
    )
    report_callable = auditor.audit(
        data=binary_matched_df,
        protected_attributes=attrs,
        pipeline_name="test_callable",
        unprivileged_groups=ACSIncomeGroups(),
    )

    for attr in attrs:
        r_dict = report_dict.attribute_results[attr]
        r_call = report_callable.attribute_results[attr]
        assert r_dict.unprivileged_group == r_call.unprivileged_group
        assert r_dict.privileged_group == r_call.privileged_group
        assert (
            r_dict.demographic_parity_difference == r_call.demographic_parity_difference
        )
        assert (
            r_dict.equal_opportunity_difference == r_call.equal_opportunity_difference
        )
        assert (
            r_dict.predictive_parity_difference == r_call.predictive_parity_difference
        )
