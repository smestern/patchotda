from utils import *

def test_filter_MMS():
    filter_MMS('VISp_Viewer', 'VISp_T_3', {
  "checked": [
    "Sst",
    "Sst_Calb2",
    "Sst_Calb2_Necab1",
    "Sst_Calb2_Pdlim5",
    "Sst_Chodl",
    "Sst_Chrna2",
    "Sst_Chrna2_Glra3",
    "Sst_Chrna2_Ptgdr",
    "Sst_Crh",
    "Sst_Crh_4930553C11Rik",
    "Sst_Crhr2",
    "Sst_Crhr2_Efemp1",
    "Sst_Esm1",
    "Sst_Hpse",
    "Sst_Hpse_Cbln4",
    "Sst_Hpse_Sema3c",
    "Sst_Mme",
    "Sst_Mme_Fam114a1",
    "Sst_Myh8",
    "Sst_Myh8_Etv1",
    "Sst_Myh8_Fibin",
    "Sst_Nr2f2",
    "Sst_Nr2f2_Necab1",
    "Sst_Nts",
    "Sst_Rxfp1",
    "Sst_Rxfp1_Eya1",
    "Sst_Rxfp1_Prdm8",
    "Sst_Tac1",
    "Sst_Tac1_Htr1d",
    "Sst_Tac1_Tacr3",
    "Sst_Tac2",
    "Sst_Tac2_Myh4",
    "Sst_Tac2_Tacstd2"
  ],
  "expanded": [
    "inhibitory"
  ]
} )

if __name__ == '__main__':
    test_filter_MMS()
    print("Passed all tests!")