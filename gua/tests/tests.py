from gua import Gua
from unittest import TestCase
import numpy as np

CENT_BINS = [(0, 5)]
FNAME = "~/repos/c2/run/20170922_1735_15o_HIR_trklt_cut/merged_AnalysisResults.root"


class Test_analysis(TestCase):
    @classmethod
    def setupClass(self):
        self.gua = Gua(FNAME, CENT_BINS)

    def test_init(self):
        # eta, phi, cent, z
        self.assertEqual(len(self.gua.singles.shape), 4)
        self.assertEqual(len(self.gua.pairs.shape), 6)
        self.assertEqual(len(self.gua.evt_counter.shape), 2)
        # should not have NaNs
        self.assertTrue(np.all(~np.isnan(self.gua.pairs)))
        self.assertTrue(np.all(~np.isnan(self.gua.singles)))
        self.assertTrue(np.all(~np.isnan(self.gua.evt_counter)))

        # check edges
        self.assertEqual(self.gua.eta_edges.size, 41)

        # check dead regions
        self.assertEqual(self.gua.dead_regions.shape, (40, 20, 8))

    def test_rho2(self):
        rho2, sigma = self.gua.rho2()
        eta_width = self.gua.eta_edges[1] - self.gua.eta_edges[0]
        phi_width = self.gua.phi_edges[1] - self.gua.phi_edges[0]
        # integral over eta1, eta2, phi1, phi2 should be ~1
        self.assertAlmostEqual(np.nansum(rho2[..., 0, 0]) * eta_width**2 * phi_width**2, 1, places=5)
        np.testing.assert_array_equal(rho2.shape, sigma.shape)
        # NaN values should be at the same place in both cases
        np.testing.assert_array_equal(np.isnan(rho2), np.isnan(sigma))

    def test_rho2_dphi(self):
        rho2, sigma = self.gua.rho2_dphi()
        eta_width = self.gua.eta_edges[1] - self.gua.eta_edges[0]
        dphi_width = self.gua.phi_edges[1] - self.gua.phi_edges[0]
        # integral over eta1, eta2, dphi should be ~1
        self.assertAlmostEqual(np.nansum(rho2[..., 0, 0]) * eta_width**2 * dphi_width, 1, places=5)
        np.testing.assert_array_equal(rho2.shape, sigma.shape)
        # NaN values should be at the same place in both cases
        np.testing.assert_array_equal(np.isnan(rho2), np.isnan(sigma))

    def test_vnm(self):
        vnm, sigma = self.gua.vnm(n_boot=10)
        # n=0, m=0 should be normalized to 1
        v00 = vnm[:, :, 0, 0, ...]
        self.assertEqual(len(vnm.shape), 6)
        np.testing.assert_array_almost_equal(v00[~np.isnan(v00)],
                                             np.full_like(v00, 1.0)[~np.isnan(v00)])

    def test_vnn_from_dphi(self):
        vnn, sigma = self.gua.vnn_from_dphi(n_boot=10)
        self.assertEqual(len(vnn.shape), 5)
        # n=0 should be normalized to 1
        v00 = vnn[:, :, 0, ...]
        np.testing.assert_array_almost_equal(v00[~np.isnan(v00)],
                                             np.full_like(v00, 1.0)[~np.isnan(v00)])
