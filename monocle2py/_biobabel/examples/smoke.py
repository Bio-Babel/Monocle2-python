"""Smoke test for monocle2py._biobabel.

Verifies the public API surface imports cleanly and the family + ncenter
helpers — which require no fixture data — return the expected types.
"""

from __future__ import annotations


def main() -> None:
    import monocle2py as m2

    # Public API surface — must round-trip imports.
    assert m2.__version__ == "2.9.0", m2.__version__
    fam = m2.negbinomial_size()
    assert fam.vfamily == "negbinomial.size", fam.vfamily
    assert m2.tobit(lower=0.1).lower == 0.1
    assert m2.cal_ncenter(500) > 0

    print(f"imported monocle2py {m2.__version__} successfully")


if __name__ == "__main__":
    main()
