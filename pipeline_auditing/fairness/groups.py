"""Definitions of unprivileged demographic groups for fairness analysis."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class UnprivilegedGroups(ABC):
    """
    Abstract base class for unprivileged-group definitions.

    Subclass this and implement get_groups() to declare which group within
    each protected attribute should be treated as historically disadvantaged
    by the fairness auditor.

    Return format of get_groups()
    ------------------------------
    A dict mapping attribute name → unprivileged group value(s).

    * Single value  → always treated as unprivileged:
        {"sex": 1.0}
    * List of values → auditor combines them into one unprivileged group:
        {"race": [1.0, 2.0, 3.0, 4.0]}

    Example
    -------
    ::
        class MyGroups(UnprivilegedGroups):
            def get_groups(self) -> dict:
                return {
                    "sex": 1.0,
                    "race": [1.0, 2.0, 3.0, 4.0],
                    ...
                }
    """

    @abstractmethod
    def get_groups(self) -> Dict[str, Any]:
        """Return mapping of protected attribute name → unprivileged group value(s)."""

    def __call__(self) -> Dict[str, Any]:
        """Delegate to get_groups() so instances are directly callable."""
        return self.get_groups()


class ACSIncomeGroups(UnprivilegedGroups):
    """
    Unprivileged group definitions for this ACSIncome Dataset

    TODO: Select protected attributes and unpriviledged groups (values) for each attribute.
          All values that are not declared as unpriviledged are automatically deemed as priviledged.


    """

    def get_groups(self) -> Dict[str, Any]:
        age_list = []
        for c in range(100):
            if c < 30 or c > 65:
                age_list.append(c)

        country_list = []
        country_list.append(13.0) #mexiko
        for c in range(554):
            if c == 303 or c > 310 and c < 399:  # latein amerika
                country_list.append(c)
            if c > 400 and c < 499:  # afrika
                country_list.append(c)
        return {
            "sex": 2.0,  # Frau
            "race": [2.0, 3.0, 4.0, 5.0, 7.0],
            # 2.0 .Black or African American alone
            # 3.0 .American Indian alone
            # 4.0 .Alaska Native alone
            # 5.0 .American Indian and Alaska Native tribes specified; or
            #   .American Indian or Alaska Native, not specified and no other
            #   .races
            # 6.0 .Asian alone
            # 7.0 .Native Hawaiian and Other Pacific Islander alone
            "age": age_list,
            "native-country": country_list
        }
