"""vaccine_tcell_model

Modular, scientifically-traceable Python package modeling:

    vaccine dose/schedule/adjuvant -> antigen/adjuvant kinetics
        -> innate cells/DC -> antigen-loaded activated DC -> pMHC
        -> antigen-specific T cells -> Tfh cells

Independently reproduces:
    - Bhagchandani et al., Science Immunology 9, eadl3755 (2024)
    - Mayer et al., PNAS 116, 5914-5919 (2019)

and introduces a clearly-labeled integrated model extension combining
the two. See docs/equations.md for the full source-to-equation mapping
and provenance of every equation and parameter in this package.
"""

__version__ = "0.1.0"
