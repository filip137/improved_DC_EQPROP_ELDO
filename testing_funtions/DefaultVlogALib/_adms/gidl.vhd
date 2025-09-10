LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;

LIBRARY STD;
USE STD.ALL;

ENTITY gidl IS
  GENERIC (
   \type\ :   real :=  1.000000e+00;
   w :   real :=  0.000000e+00;
   flagtat :   real :=  1.000000e+00;
   tagidl :   real :=  0.000000e+00;
   tox :   real :=  3.000000e-09;
   mstar :   real :=  2.500000e-01;
   btfgidl :   real :=  1.000000e+00;
   gmfgidl :   real :=  5.000000e-01;
   trpgidl :   real :=  0.000000e+00;
   epsgidl :   real :=  1.000000e-02;
   ephgidl :   real :=  1.000000e-01;
   tr :   real :=  2.700000e+01;
   tcelcius :   real :=  -2.740000e+02;
   trise :   real :=  0.000000e+00;
   tbgidl :   real :=  2.610000e+07;
   tegidl :   real :=  8.000000e-01;
   agidl :   real :=  0.000000e+00;
   bgidl :   real :=  2.300000e+09;
   cgidl :   real :=  5.000000e-01;
   egidl :   real :=  8.000000e-01;
   nbgidl :   real :=  1.500000e+00 );
  PORT (
    SIGNAL plus : INOUT STD_LOGIC;
    SIGNAL minus : INOUT STD_LOGIC;
    SIGNAL ctrl : INOUT STD_LOGIC);
END ENTITY gidl;

