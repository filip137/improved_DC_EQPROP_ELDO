perfect_amp_quadratic_diode = {
    "PARAMS": """

    """,

    "SUBCIRCUIT": """
*** Ideal quadratic "diode" subcircuit
* I = 10 * V(a,b)^2
.SUBCKT ideal_quadratic_model a b
GQUAD a b VALUE = { 10 * max(V(a,b), 0)^2 }
.ENDS ideal_quadratic_model

*** Library name: tests
*** Cell name: DOUBLE_DIODE_QUADRATIC
*** View name: schematic
.SUBCKT DOUBLE_DIODE_QUADRATIC VIN VOUT
    X0 VIN NET6 ideal_quadratic_model
    X1 NET7 VIN ideal_quadratic_model

    V2 NET6 0 DC VDIODE1
    V3 NET7 0 DC VDIODE2

    F0 0 VIN EVCVS1 AMPC
    EVCVS1 VOUT 0 VIN 0 AMP
.ENDS
*** End of subcircuit definition.
"""
}

# -*- coding: utf-8 -*-
