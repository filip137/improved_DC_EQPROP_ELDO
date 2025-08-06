pmos_cs = {'PARAMS' :
           """

.PARAM PMOS_CS_W=6u
.PARAM PMOS_CS_VDD=3.3
"""
,
'SUBCIRCUIT':"""
*** Library name: current_sources
*** Cell name: pmos_cs
*** View name: schematic
.SUBCKT PMOS_CS GROUND INOUTPUT_PMOS_CS
    V20 NET4 GROUND DC PMOS_CS_V_BIAS
    V21 NET2 GROUND DC PMOS_CS_VDD
XM16 INOUTPUT_PMOS_CS NET4 NET2 NET2 EP5V0_BS3JU w=PMOS_CS_W l=1e-06
+nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1
+dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0
.ENDS
*** End of subcircuit definition.
"""}

#.PARAM PMOS_CS_V_BIAS=2

##corrected version 3.6
self_biased_nmos_cs = {'PARAMS' : """
.PARAM NMOS_SELF_BIASED_CAP=5n
.PARAM NMOS_SELF_BIASED_CS_W_CS=3.5u
.PARAM NMOS_SELF_BIASED_CS_W_CASCODE=3.5u
.PARAM NMOS_SELF_BIASED_CS_R_BIAS2=1MEG
.PARAM NMOS_SELF_BIASED_CS_R_BIAS1=330k
.PARAM NMOS_SELF_BIASED_CS_BIAS_CASCODE=2
                       """
                       ,
                       "SUBCIRCUIT" : """
.SUBCKT NMOS_SELF_BIASED_CS GROUND INOUTPUT_SELF_BIASED_CS
XM17 GROUND NET07 NET7 GROUND EN5V0_BS3JU w=NMOS_SELF_BIASED_CS_W_CS
+l=1e-06 nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1
+dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0
XM7 NET7 NET10 INOUTPUT_SELF_BIASED_CS NET7 EN5V0_BS3JU
+w=NMOS_SELF_BIASED_CS_W_CASCODE l=1.5e-06 nfing=1 ncrsd=1 number=1
+srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1
+lpe=0
R3 INOUTPUT_SELF_BIASED_CS NET07 NMOS_SELF_BIASED_CS_R_BIAS1
R2 NET07 GROUND NMOS_SELF_BIASED_CS_R_BIAS2
V15 NET10 0 DC NMOS_SELF_BIASED_CS_BIAS_CASCODE
C0 NET07 GROUND NMOS_SELF_BIASED_CAP
.ENDS
*** End of subcircuit definition.
"""       


             }
"""Here one needs to take care that the subcircuit names are the same for all 
elements because when the netlist is build I always use NMOS_SELF_BIASED_CS name
and PMOS_CS
"""



##OUTPUT_READ_VOLT WILL BE INCLUDED VIA THE SIMULATION PARAMETERS
improved_pmos_cs = {'PARAMS' :
           """

.PARAM PMOS_CS_W=6u
.PARAM PMOS_CS_VDD = 3.3
.PARAM PMOS_CS_VDD_NEG = -3.3 
.PARAM LOW_NOISE_OPTION=0
"""
,
'SUBCIRCUIT':
    """.SUBCKT PMOS_CS GROUND INOUTPUT_PMOS_CS
    V21 NET2 NET07 DC PMOS_CS_VDD
XM16 INOUTPUT_PMOS_CS NET4 NET2 NET2 EP5V0_BS3JU w=PMOS_CS_W l=1e-06
+nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1
+dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0
    V18 NET07 GROUND PWL ( 0 PMOS_CS_VDD_NEG 1u 0 )
    V20 NET4 0 DC 0 PWL ( 0 2.7 START_READ_TIME 2.7 {START_READ_TIME +
+RISE_TIME} OUTPUT_READ_VOLT END_READ_TIME OUTPUT_READ_VOLT {END_READ_TIME
++ RISE_TIME} 3.3)
.ENDS"""}

###Improved self biased nmos with the lower capacitance to avoid issues when I am doing transient simulation. Compared to the previous version also the nmos width is larger and the cascode nmos is not diode connected anymore
improved_self_biased_nmos = {"PARAMS" : """
.PARAM NMOS_SELF_BIASED_CS_W_CS=5u
.PARAM NMOS_SELF_BIASED_CS_W_CASCODE=5u
.PARAM NMOS_SELF_BIASED_CS_R_BIAS1=330k
.PARAM NMOS_SELF_BIASED_CS_R_BIAS2=1MEG
.PARAM NMOS_SELF_BIASED_CS_BIAS_CASCODE=1.9
.PARAM NMOS_SELF_BIASED_CAP=3p""",
"SUBCIRCUIT" : """.SUBCKT NMOS_SELF_BIASED_CS GROUND INOUTPUT_SELF_BIASED_CS
XM17 GROUND NET07 NET7 GROUND EN5V0_BS3JU w=NMOS_SELF_BIASED_CS_W_CS
+l=1e-06 nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1
+dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0
XM7 NET7 NET10 INOUTPUT_SELF_BIASED_CS NET7 EN5V0_BS3JU
+w=NMOS_SELF_BIASED_CS_W_CASCODE l=1.5e-06 nfing=1 ncrsd=1 number=1
+srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1
+lpe=0
    R3 INOUTPUT_SELF_BIASED_CS NET07 NMOS_SELF_BIASED_CS_R_BIAS1
    R2 NET07 GROUND NMOS_SELF_BIASED_CS_R_BIAS2
    V15 NET10 0 DC NMOS_SELF_BIASED_CS_BIAS_CASCODE
    C0 NET07 GROUND NMOS_SELF_BIASED_CAP
.ENDS"""
}