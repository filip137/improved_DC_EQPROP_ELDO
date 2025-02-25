NETLIST_DEFINITIONS = {
    "amp_ss": """\
.LIB /cao/DK/ST/HCMOS9A_10.9/Addon_NVM_H9A@2018.4.1/tools/eldo/model_oxram/OxRRAM.lib OxRRAM_TT
.LIB /home/filip/CMOS130/corners.eldo

*** Library name: amplifiers
*** Cell name: vcvs_ss
*** View name: schematic
.SUBCKT VCVS_SS INPUT_VCVS OUTPUT_VCVS
    R5 VDD OUTPUT_CS_2 RD
    R3 VDD OUTPUT_CS_1 RD
    R21 VDD NET03 R_VCVS_BIAS1
    R19 NET7 0 RS
    R2 NET07 0 RS_CD
    R4 NET8 0 RS
    R1 NET03 0 R_VCVS_BIAS2
XM13 VDD OUTPUT_CS_2 NET07 0 EN5V0_BS3JU w=CD1_W l=CD1_L nfing=1 ncrsd=1
+number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0
+soa=1 lpe=0
XM12 OUTPUT_CS_2 OUTPUT_CS_1 NET7 0 EN5V0_BS3JU w=2e-06 l=CS2_L nfing=1
+ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0
+dmu_mdev=0 soa=1 lpe=0
XM11 OUTPUT_CS_1 NET03 NET8 0 EN5V0_BS3JU w=CS1_W l=2e-06 nfing=1 ncrsd=1
+number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0
+soa=1 lpe=0
    C2 NET07 OUTPUT_VCVS CAP
    C0 INPUT_VCVS NET03 CAP
    V0 VDD 0 DC VDD
.ENDS
*** End of subcircuit definition.

*** Library name: amplifiers
*** Cell name: cccs_ss
*** View name: schematic
.SUBCKT CCCS_SS OUTPUT_CURRENT_CCCS INPUT_CCS_1 INPUT_CCS_2
XM1 NET11 INPUT_DIFFERENTIAL2 NET023 0 EN5V0_BS3JU w=WIDTH_NMOS_DIFF_A
+l=LENGTH_NMOS_DIFF_A nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1
+mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0
XM0 OUTPUT_DIFFERENTIAL_AMP INPUT_DIFFERENTIAL1 NET11 0 EN5V0_BS3JU
+w=WIDTH_NMOS_DIFF_A l=LENGTH_NMOS_DIFF_A nfing=1 ncrsd=1 number=1
+srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1
+lpe=0
XM9 OUTPUT_CURRENT_CCCS NET28 NET27 0 EN5V0_BS3JU w=1.6e-06 l=LENGH_CASC_2
+nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1
+dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0
XM8 NET27 INPUT_CASCADE 0 0 EN5V0_BS3JU w=1.2e-06 l=LENGH_CASC_1 nfing=1
+ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0
+dmu_mdev=0 soa=1 lpe=0
XM10 INPUT_CASCADE OUTPUT_DIFFERENTIAL_AMP 0 0 EN5V0_BS3JU w=8e-07 l=2e-06
+nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1
+dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0
    R0 INPUT_CCS_1 INPUT_CCS_2 R_SHUNT
    R3 VDD OUTPUT_DIFFERENTIAL_AMP RES_DIFF_AMP
    R4 VDD NET023 RES_DIFF_AMP
    R9 VDD INPUT_DIFFERENTIAL1 R_CCCS_BIAS1
    R11 VDD INPUT_DIFFERENTIAL2 R_CCCS_BIAS1
    R12 INPUT_DIFFERENTIAL2 0 R_CCCS_BIAS2
    R10 INPUT_DIFFERENTIAL1 0 R_CCCS_BIAS2
    R23 VDD INPUT_CASCADE R_D_DIFF_AMP
    I0 VDD OUTPUT_CURRENT_CCCS DC IBIAS_CASCODE
    I2 NET11 0 DC IBIAS_DIFF_A
    V0 VDD 0 DC VDD
    V3 NET28 0 DC V_CASCODE
    C0 INPUT_CCS_1 INPUT_DIFFERENTIAL1 CAP
    C1 INPUT_DIFFERENTIAL2 INPUT_CCS_2 CAP
.ENDS
*** End of subcircuit definition.

*** Library name: amplifiers
*** Cell name: amplification_ss
*** View name: schematic
.SUBCKT AMPLIFICATION_SS AMP_INPUT AMP_OUTPUT
    XI0 AMP_INPUT VCVS_OUTPUT VCVS_SS
    XI1 AMP_INPUT VCVS_OUTPUT AMP_OUTPUT CCCS_SS
.ENDS
*** End of subcircuit definition.
"""


, "perfect_amp" : """
.LIB /cao/DK/ST/HCMOS9A_10.9/Addon_NVM_H9A@2018.4.1/tools/eldo/model_oxram/OxRRAM.lib OxRRAM_TT
.LIB /home/filip/CMOS130/corners.eldo 
.LIB /home/filip/Documents/MyDiode.lib 
*** Library name: tests_new
*** Cell name: neuron
*** View name: schematic
.SUBCKT NEURON VIN VOUT
    D0 VIN NET6 diode1
    D1 NET7 VIN diode1
    V2 NET6 0 DC VDIODE1
    V3 NET7 0 DC VDIODE2
    F0 0 VIN EVCVS1 {1/AMP}
    EVCVS1 VOUT 0 VIN 0 AMP
.ENDS
*** End of subcircuit definition.
"""
}
    
    
PARAMS = {
    "amp_ss1": """\
.PARAM WIDTH_NMOS_DIFF_A=650n
.PARAM VDD=3.3
.PARAM V_CASCODE=2.9
.PARAM RS_CD=10k
.PARAM RS=10k
.PARAM RES_DIFF_AMP=32k
.PARAM RD=100k
.PARAM R_VCVS_BIAS2=10MEG
.PARAM R_VCVS_BIAS1=10MEG
.PARAM R_SHUNT=5k
.PARAM R_D_DIFF_AMP=65k
.PARAM R_CCCS_BIAS2=15MEG
.PARAM R_CCCS_BIAS1=10MEG
.PARAM LOW_NOISE_OPTION=0
.PARAM LENGTH_NMOS_DIFF_A=650n
.PARAM LENGH_CASC_2=3.6u
.PARAM LENGH_CASC_1=1.6u
.PARAM IBIAS_DIFF_A=90u
.PARAM IBIAS_CASCODE=30u
.PARAM CS2_L=2u
.PARAM CS1_W=1.2u
.PARAM CD1_W=4u
.PARAM CD1_L=650n
.PARAM CAP=1n
""",
"perfect_amp" : """
.PARAM VDIODE2=-0.5
.PARAM VDIODE1=0.5
.PARAM AMP=3
.PARAM AMPC=1"""
}
    
