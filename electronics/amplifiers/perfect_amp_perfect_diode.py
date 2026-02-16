perfect_amp_perfect_diode_nonlin = {
    "PARAMS": """

    """,

    "SUBCIRCUIT": """
*** Ideal exponential "diode" subcircuit
* I = Is * (exp(V/Vt) - 1)
.SUBCKT ideal_model a b PARAMS: Is=1e-6 Vt=0.05
GEXP a b VALUE = { Is * (exp( V(a,b)/Vt ) - 1 ) }
.ENDS ideal_model

*** Library name: tests
*** Cell name: neuron
*** View name: schematic
.SUBCKT DOUBLE_DIODE_EXPONENTIAL VIN VOUT
    X0 VIN NET6 ideal_model    
    X1 NET7 VIN ideal_model     

    V2 NET6 0 DC VDIODE1
    V3 NET7 0 DC VDIODE2

    F0 0 VIN EVCVS1 AMPC
    EVCVS1 VOUT 0 VIN 0 AMP
.ENDS
*** End of subcircuit definition.
"""
}
# -*- coding: utf-8 -*-

