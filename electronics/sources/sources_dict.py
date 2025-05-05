pmos_cs = {'PARAMS' :
           """

.PARAM PMOS_CS_W=6u
.PARAM PMOS_CS_VDD=3.3
.PARAM PMOS_CS_V_BIAS=1.5
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