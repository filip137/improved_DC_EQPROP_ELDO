discharge_tran = {'SUBCIRCUIT':""" *** Library name: discharge_element
*** Cell name: discharge_tran
*** View name: schematic
.SUBCKT DISCHARGE_TRAN GROUND OUTPUT
    V23 NET2 GROUND PWL ( 0 0 START_DISCHARGE_TIME 0
+{START_DISCHARGE_TIME+1u} 3.3 END_DISCHARGE_TIME 3.3
+{END_DISCHARGE_TIME+1u} 0)
XM6 OUTPUT NET2 GROUND GROUND EN5V0_BS3JU w=4e-6 l=2e-6 nfing=1
+ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0
+dmu_mdev=0 soa=1 lpe=0
.ENDS
*** End of subcircuit definition. """,
'PARAMS' : 
    """
.PARAM START_DISCHARGE_TIME = 32u
.PARAM END_DISCHARGE_TIME = 35u
"""}

#there params will get overwritten anyway