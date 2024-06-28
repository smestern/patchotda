from utils import *

def test_filter_MMS():
    filter_MMS('VISp_Viewer', 'VISp_MET_3', {
  "checked": [
    "inhibitory",
    "Lamp5",
    "Lamp5-MET-1",
    "Lamp5-MET-2",
    "Pvalb",
    "Pvalb-MET-1",
    "Pvalb-MET-2",
    "Pvalb-MET-3",
    "Pvalb-MET-4",
    "Pvalb-MET-5",
    "Sncg",
    "Sncg-MET-1",
    "Sncg-MET-2",
    "Sncg-MET-3",
    "Sst",
    "Sst-MET-1",
    "Sst-MET-10",
    "Sst-MET-11",
    "Sst-MET-12",
    "Sst-MET-13",
    "Sst-MET-2",
    "Sst-MET-3",
    "Sst-MET-4",
    "Sst-MET-5",
    "Sst-MET-6",
    "Sst-MET-7",
    "Sst-MET-8",
    "Sst-MET-9",
    "Vip",
    "Vip-MET-1",
    "Vip-MET-2",
    "Vip-MET-3",
    "Vip-MET-4",
    "Vip-MET-5"
  ],
  "expanded": []
} )

def test_ub_sink():
    import ot
    from ot.datasets import make_2D_samples_gauss
    OT = ot.da.UnbalancedSinkhornTransport()
    Xs = make_2D_samples_gauss(n=1000, m=1000, sigma=[[2, 1], [1, 2]], random_state=42)
    Xt = make_2D_samples_gauss(n=1000, m=1000, sigma=[[2, 1], [1, 2]], random_state=42)[0]
    Xs = Xs.astype('float32')
    Xt = Xs + 0.5
    Xt = Xt.astype('float32')
    OT.fit(Xs, Xt)
    OT.transform(Xs)
    return OT


if __name__ == '__main__':
    test_ub_sink()
    test_filter_MMS()
    print("Passed all tests!")