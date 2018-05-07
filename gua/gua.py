import numpy as np
from rootpy.io import root_open
import root_numpy
# from c2_post import utils
import utils
from rootstrap import Collector


class Gua():
    def __init__(self, fname, cent_bins, external_acceptance_file=None):
        self.singles, edges = get_singles_and_edges(fname, cent_bins)
        self.pairs = get_pairs(fname, cent_bins)
        self.evt_counter = get_n_events(fname, cent_bins)
        self.eta_edges = edges[0]
        self.phi_edges = edges[1]
        self.z_edges = edges[-1]
        if external_acceptance_file:
            max_res_map, max_res_edges = get_max_res_map(external_acceptance_file)
        else:
            max_res_map, max_res_edges = get_max_res_map(fname)
        # FIXME: hack around hi-res map not being hir res enough in z:
        # Double bins of max_res_map in z; fill each second bin with the old map
        # if external_acceptance_file:
        #     _max_res_map = np.full((max_res_map.shape[:2] + (max_res_map.shape[2] * 2, )), False)
        #     _max_res_map[:, :, ::2] = max_res_map
        #     _max_res_map[:, :, 1::2] = max_res_map
        #     max_res_map = _max_res_map
        #     # FIXME: more hack to double the edges
        #     max_res_z_edges = np.ones((max_res_edges[2].size * 2 - 1, ),
        #                               dtype=np.float)
        #     max_res_z_edges[:] = np.linspace(max_res_edges[-1][0], max_res_edges[-1][-1],
        #                                      num=max_res_z_edges.size)
        #     max_res_edges[2] = max_res_z_edges
        # False == Dead
        self.acceptance = ~utils.get_dead_regions_map(edges[0], edges[1], edges[-1],
                                                      max_res_map, max_res_edges)
        self.dphi_edges = None
        self.deta_edges = None
        self.cent_bins = cent_bins

    def merge_with_low_mult(self, low_mult):
        """
        Merge two GUA objects along their centrality axis

        Parameters
        ----------
        low_mult: nd.array
            `Gua` object with low multipicity
        """
        print self.singles.shape, low_mult.singles.shape
        self.singles = np.concatenate((self.singles, low_mult.singles), -2)
        self.pairs = np.concatenate((self.pairs, low_mult.pairs), -2)
        self.evt_counter = np.concatenate((self.evt_counter, low_mult.evt_counter), -2)
        self.cent_bins = self.cent_bins + low_mult.cent_bins

    def _r2_sig2(self):
        s = self.singles / self.evt_counter
        s_sig = np.sqrt(self.singles) / self.evt_counter
        p = self.pairs / self.evt_counter
        r2 = p / (s[None, :, None, :, ...] * s[:, None, :, None, ...])
        return (r2 - 1)**2 * ((s_sig / s)[None, :, None, :, ...]**2 + (s_sig / s)[:, None, :, None, ...]**2 + 2 * (r2 - 1))

    def r2(self):
        """
        Compute the normalized pair probability distribution and its uncertainties
        """
        ss = (self.singles[:, None, :, None, ...] * self.singles[None, :, None, :, ...])
        r2 = np.full_like(self.pairs, np.nan)
        r2[self.pairs > 0] = self.pairs[self.pairs > 0] / ss[self.pairs > 0]
        r2 *= self.evt_counter

        # disable the diagonal in (eta, eta) FIXME: Should only be the phi-phi diagonal!
        eta_diag_idx = np.diag_indices(r2.shape[0])
        r2[eta_diag_idx[0], eta_diag_idx[1], ...] = np.nan

        # Mask parts that overlap with dead regions
        # Copy the mask so that it can be later modified!~
        r2 = np.ma.masked_array(r2, mask=np.copy(self.pair_acceptance()))

        # Normalize each cent-z-slice
        # eta_width = self.eta_edges[1] - self.eta_edges[0]
        # phi_width = self.phi_edges[1] - self.phi_edges[0]

        # scalling = utils.reduce_all_but(r2, [4, 5], np.nansum) * eta_width**2 * phi_width**2
        # r2 /= scalling
        # Compute uncertainties based on pairs
        sig_rel = np.full_like(r2, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            # square of uncert of single dists (1/sqrt(s))^2 each; thus no np.sqrt in the terms
            # rel_singles_uncert2 = (1.0 / self.singles[:, None, :, None, ...] +
            #                        1.0 / self.singles[None, :, None, :, ...])
            # Rel. uncertainty of r2 is approximately given by 1/pairs
            sig_rel[~np.isnan(r2)] = np.sqrt((1.0 / self.pairs)[~np.isnan(r2)])
        sigma_abs = r2 * sig_rel
        # Set values where we do not have a r2
        sigma_abs[np.isnan(r2)] = np.nan
        sigma_abs = np.ma.masked_array(sigma_abs, mask=self.pair_acceptance())
        return r2, sigma_abs

    def r2_dphi(self):
        """
        Normalized distribution of pairs r2(eta1, eta2, dphi, cent, z)
        """
        r2, sig = self.r2()
        keep_axes = [0, 1, [2, 3], 4, 5]
        r2_dphi = utils.reduce_weighted_avg(r2, sig, axes=keep_axes, new_dphi_axis=2)
        sig_dphi = utils.reduce_all_but(sig, axes=keep_axes,
                                        func=utils.uncert_weighted_avg,
                                        new_dphi_axis=2)
        # Re-calculate scalling
        # eta_width = self.eta_edges[1] - self.eta_edges[0]
        # dphi_width = self.phi_edges[1] - self.phi_edges[0]
        # scalling = utils.reduce_all_but(rho2_dphi, [3, 4], np.nansum) * eta_width**2 * dphi_width
        # return r2_dphi / scalling, sig_dphi / scalling
        return r2_dphi, sig_dphi

    def vnm(self, n_boot=100):
        """
        Compute 2D Fourier transform of phi1/phi2 plane and its uncertainties (bootstrapped)
        """
        col = Collector()
        r2, sig = self.r2()
        for _ in range(n_boot):
            _strapped = np.full_like(r2.data, np.nan)
            _strapped[~np.isnan(r2)] = np.random.normal(loc=r2[~np.isnan(r2)],
                                                        scale=sig[~np.isnan(r2)])
            # Fourier over phi1, phi2
            Vnm = np.fft.fft2(_strapped, axes=(2, 3))
            # Align the origin such that n=m=0 is at 00 and the positive modes follow the bin indices
            Vnm = np.flip(np.roll(Vnm, -1, 2), 2)
            # Chop the negative modes
            Vnm = Vnm[:, :, :11, :11, ...]
            vnm = np.abs(Vnm)
            # Normalize to 0 modes
            scalling = vnm[:, :, :1, :1, ...]
            vnm /= scalling
            col.add(vnm)
        return col.mean(), col.sigma()

    def vnn_from_dphi(self, n_max=4, n_boot=100):
        """
        Two-particle Fourier coefficients with errors propagated by bootstrapping.
        Parameters
        ----------
        n_max: int
            Highes `n` included in the output
        n_boot: int
            Number of bootstrapping iterations

        Returns
        -------
        vnn, vnn_sig:
            MaskedArrays `v_nn` and the absolut uncertainties of shape (eta, eta, n, cent, z)
        """
        col = Collector()
        r2, sig = self.r2_dphi()
        for _ in range(n_boot):
            _strapped = np.full_like(r2, np.nan)
            _strapped[~np.isnan(r2)] = np.random.normal(loc=r2[~np.isnan(r2)],
                                                        scale=sig[~np.isnan(r2)])
            Vnn = np.fft.rfft(_strapped, axis=2)
            vnn = np.abs(Vnn)
            # Normalize to 0 mode
            scalling = vnn[:, :, [0], ...]
            vnn /= scalling
            col.add(vnn)
        mean, sig = col.mean(), col.sigma()
        mask = np.broadcast_to(self.vnn_acceptance()[:, :, None, ...], mean.shape)
        mean = np.ma.masked_array(mean, mask=mask)
        sig = np.ma.masked_array(sig, mask=mask)
        # Future proof: Subviews into the shared arrays also share the mask
        # See: https://stackoverflow.com/questions/41028253/numpy-1-13-maskedarrayfuturewarning-setting-an-item-on-a-masked-array-which-has
        mean._sharedmask = False
        sig._sharedmask = False
        return mean[:, :, :n_max + 1, ...], sig[:, :, :n_max + 1, ...]

    def cms_ratio(self):
        pass

    def pair_acceptance(self):
        acceptance = self.acceptance
        acceptance = acceptance[None, :, None, :, None, :] | acceptance[:, None, :, None, None, :]
        acceptance = np.broadcast_to(acceptance, self.pairs.shape)
        return acceptance

    def vnn_acceptance(self):
        mask = utils.reduce_all_but(self.pair_acceptance(), [0, 1, 4, 5], np.all)
        return mask


def _stich_all_eta(d):
    """
    Stich all histograms together to span full eta phase-space

    Parameters
    ----------
    d : dict
        Dictionary where the key denotes the pair region (e.g. 'bwdfwd')
    """
    nbins = []
    nbins.append(d['bwdbwd'].shape[0])
    nbins.append(d['itsits'].shape[0])
    nbins.append(d['fwdfwd'].shape[0])
    shape = (sum(nbins), sum(nbins)) + d['bwdbwd'].shape[2:]

    # accumulate in order to get upper bin edges indices
    up_edges = [sum(nbins[:y]) for y in range(1, len(nbins) + 1)]
    full = np.full(shape=shape, fill_value=np.nan, dtype=d['itsits'].dtype)
    full[:up_edges[0], :up_edges[0], ...] = d['bwdbwd']
    full[up_edges[0]:up_edges[1], up_edges[0]:up_edges[1], ...] = d['itsits']
    full[up_edges[1]:, up_edges[1]:, ...] = d['fwdfwd']
    full[:up_edges[0], up_edges[0]:up_edges[1], ...] = d['bwdits']
    full[:up_edges[0], up_edges[1]:, ...] = d['bwdfwd']
    full[up_edges[0]:up_edges[1], up_edges[1]:, ...] = d['itsfwd']
    return full


def get_pairs(fname, cent_bins):
    regs = ['bwdbwd', 'itsits', 'fwdfwd', 'bwdits', 'bwdfwd', 'itsfwd']
    with root_open(fname) as f:
        all_cents = []
        for cent_slice in f.c2_correlations:
            d = {}
            for h in cent_slice[:6]:
                hname = h.GetName()
                for reg in regs:
                    if reg in hname and 'pair' in hname:
                        d[reg] = root_numpy.hist2array(h)
            all_cents.append(_stich_all_eta(d))
    out = np.concatenate(all_cents, axis=-2)
    # treat Nan values as 0; strange that they are there!
    if np.any(np.isnan(out)):
        print "Found NaN in pure pair histogram!?"
        # out[np.isnan(out)] = 0
    eta1, eta2 = np.tril_indices(out.shape[0])
    out[eta1, eta2, ...] = 0  # out[np.tril_indices(out.shape[0]), ...][::-1]
    out += np.swapaxes(np.swapaxes(out, 0, 1), 2, 3)
    # mirror the lower triangle (eta_a < eta_b) to the upper one
    # kill unused pt dimension
    return out[..., 0, :, :]


def get_singles_and_edges(fname, cent_bins):
    regs = ['bwd', 'its', 'fwd']
    mult_edges = []
    with root_open(fname) as f:
        all_cents = []
        for cent_slice in f.c2_correlations:
            hs_eta = []
            eta_edges = []
            for h in cent_slice[6:9]:
                hname = h.GetName()
                for reg in regs:
                    if reg in hname and 'single' in hname:
                        a, edges = root_numpy.hist2array(h, return_edges=True)
                        mult_edges.append(edges[-2])
                        hs_eta.append(a)
                        eta_edges.append(edges[0])
            all_cents.append(np.concatenate(hs_eta, axis=0))
    # merge edges, but don't double count the ones at the borders between regions
    edges[0] = np.concatenate([eta_edges[0][:-1], eta_edges[1][:-1], eta_edges[2]],
                              axis=0)

    edges[-2] = mult_edges
    out = np.concatenate(all_cents, axis=-2)
    # kill unused pt dimension
    return out[..., 0, :, :], edges


def get_n_events(fname, cent_bins):
    with root_open(fname) as f:
        all_cents = []
        for cent_slice in f.c2_correlations:
            h = cent_slice[9]
            if h.GetName() != "eventCounter":
                raise ValueError("Expected event counter histogram")
            all_cents.append(root_numpy.hist2array(h))
    return np.concatenate(all_cents, axis=-2)


def get_max_res_map(fname):
    """Get the histogram showing the acceptance of the detector at maximal resolution"""
    with root_open(fname) as f:
        h = list(f.c2_correlations)[0][-1]
        if h.GetName() != "etaPhiZvtx_max_res":
            raise ValueError("Histogram order is different than expected")
        return root_numpy.hist2array(h, return_edges=True)
