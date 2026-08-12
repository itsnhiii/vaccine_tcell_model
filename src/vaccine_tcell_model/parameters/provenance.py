"""Allowed provenance categories for parameter values.

Every Parameter (see metadata.py) must declare a SourceType. This is
the enforcement point for master-spec rule 11: "Every parameter must
have provenance: literature / fitted / assumed / user_defined / derived
/ model_extension." No other category is permitted.
"""

from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    """Where a parameter's numeric value comes from.

    literature:      Taken directly from a published paper/supplement,
                      with an exact citation in Parameter.source.
    fitted:           Estimated by fitting this package's own code to
                      data (Parameter.source should name the dataset/target).
    assumed:          A modeling assumption made by the source paper or
                      by this project, not measured or fit to data.
    user_defined:     Supplied by a user of this package at call time.
    derived:          Computed from other parameters/equations rather
                      than an independent value.
    model_extension:  Specific to this project's integrated-model
                      extension; not present in either source paper.
    """

    LITERATURE = "literature"
    FITTED = "fitted"
    ASSUMED = "assumed"
    USER_DEFINED = "user_defined"
    DERIVED = "derived"
    MODEL_EXTENSION = "model_extension"
