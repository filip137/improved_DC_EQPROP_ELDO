perfect_amp_nonlin = { "PARAMS" :
                      """
.PARAM AMP=1
.PARAM AMPC=1
.PARAM VDIODE2=-20
.PARAM VDIODE1=20
                      """,
    
"SUBCIRCUIT" :
""".LIB /home/filip/Documents/MyDiode.lib

*** Library name: tests
*** Cell name: neuron
*** View name: schematic
.SUBCKT NEURON VIN VOUT
    D0 VIN NET6 diode1
    D1 NET7 VIN diode1
    V2 NET6 0 DC VDIODE1
    V3 NET7 0 DC VDIODE2
    F0 0 VIN EVCVS1 AMPC
    EVCVS1 VOUT 0 VIN 0 AMP
.ENDS
*** End of subcircuit definition."""}