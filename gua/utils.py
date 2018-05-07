import logging
import warnings
import numpy as np

from c2_post.vn_extractors.factorization import find_v, find_v2_twist


def _flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in _flatten(i):
                yield j
        else:
            yield i


def _get_diags(a, axes):
    """
    Take the diagonal between the two given axes. Axis2 is reduced to length 1

    Parameters
    ----------
    a : ndarray
    axis1, axis2 : int
        The axes between which the diagonal is taken

    Returns
    -------
    Lists, nested according to the number of given axes
    """
    _axes = axes[:]
    ax1, ax2 = _axes.pop(0)
    offsets = range(-a.shape[ax1] + 1, a.shape[ax2])
    diags = [np.expand_dims(np.rollaxis(a.diagonal(os, ax1, ax2), -1, ax1), ax2) for os in offsets]
    try:
        diags = [_get_diags(diag, _axes) for diag in diags]
    except IndexError:
        pass
    return diags


def reduce_all_but(a, axes, func, new_dphi_axis=None):
    """
    Reduce all dimensions except those specified by applying

    `func`. This function takes special care to reduce all at once so
    that calculations of the mean (including NaNs) are correct.

    Parameters
    ----------
    a : ndarray
        Input array which will be reduced
    axes : list
        List of axes that should be kept. If a tuple of integers is
        given, the diagonal between these axes will be kept
    func : function
        Function used to reduce the array. Eg. np.sums
    new_dphi_axis : int
        If given, treate this axis as the dphi axis and reduce it to
        [0, 2pi)

    Returns
    -------
    ndarray :
        New array with dimensionality len(axes)
    """

    if new_dphi_axis is not None:
        # concatenate the current array with itself on the phi2 axis
        old_phi2_ax = axes[new_dphi_axis][1]
        reduced = reduce_all_but(np.concatenate((a, a), axis=old_phi2_ax), axes, func)
        # select the "2pi" region if the option was given.
        out_selection = [slice(None, None) for dim in range(len(axes))]
        out_selection[new_dphi_axis] = slice(a.shape[old_phi2_ax] - 1,
                                             a.shape[old_phi2_ax] - 1 + a.shape[old_phi2_ax])
        return reduced[out_selection]

    diag_dims = [ax for ax in axes if isinstance(ax, (list, tuple))]
    squash_dims = [ax for ax in axes if isinstance(ax, int)]
    # figure out what the final shape will be:
    out_shape = []
    diag_shape = []
    linear_shape = []
    for el in axes:
        if isinstance(el, (list, tuple)):
            ax1, ax2 = el
            out_shape.append(a.shape[ax1] + (a.shape[ax2] - 1))
            diag_shape.append(a.shape[ax1] + (a.shape[ax2] - 1))
        else:
            out_shape.append(a.shape[el])
            linear_shape.append(a.shape[el])
    # do the diagonals first
    if diag_dims:
        diags = [el for el in _flatten(_get_diags(a, diag_dims))]
    else:
        diags = [a[:]]
    for idiag, _ in enumerate(diags):
        # put all the non-diagonal dimensions we want to keep into the front
        for ikeep, keep_dim in enumerate(squash_dims):
            diags[idiag] = diags[idiag].swapaxes(ikeep, keep_dim)
        # reshape (flatten) all the dimensions we don't want to keep into one dim
        diags[idiag] = diags[idiag].reshape(diags[idiag].shape[:len(squash_dims)] + (-1, ))
        with warnings.catch_warnings():
            # This function often ends up being np.nanmean([nan]) == nan.
            # This is exactly what we want, but it raises a runtime warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diags[idiag] = func(diags[idiag], axis=-1)
    diags = np.array(diags)
    # At this point, the deed is done but the diagonals are all in
    # front and in one single dimension.  First, reshape the diagonals
    # so that they are seperated from each other, then roll them to
    # to the place where they should be wrt the "straight" dimensions (if any)
    diags.reshape(diag_shape + linear_shape)
    # if there are no "straight" dimensions, there is nothing to roll here!
    if linear_shape:
        for idiag_dim, diag_dim in enumerate(diag_dims):
            this_diag_out_dim = axes.index(diag_dim)
            # the axis idiag_dim is rolled until it lies "before" the other argument; thus +1
            diags = np.rollaxis(diags, idiag_dim, this_diag_out_dim + 1)
    # reshape the diagonal part back to what it was
    diags = np.array(diags).reshape(out_shape)
    return diags


def reduce_weighted_avg(a, sigmas, axes, new_dphi_axis=None):
    """
    Reduce all but the specified `axes` with a weighted average. The
    weight is given by each points standard deviation.

    Taken from Taylor "Introduction to error analysis" p. 176ff

    Parameters
    ----------
    a : np.ndarray
        Mean values in each bin of the phase-space
    sigmas : np.ndarray
        Uncertainties of the mean in each bin of the phase-space
    axes : list
        Axes which will be kept; past on to `reduce_all_but`
    new_dphi_axis : int
        If given, treate this axis as the dphi axis and reduce it to
        [0, 2pi). See utils.reduce_all_but for more details.

    Returns
    -------
    np.ndarray : Means after combining the measurements over `axis`
    """
    if a.shape != sigmas.shape:
        raise ValueError("Shape of data and sigma array must match!")
    # weights have to be nan where ever `a` is nan! If not we screwed up the sum over the weights!
    weights = np.full_like(a, np.nan, dtype=a.dtype)
    weights[~np.isnan(a)] = 1 / sigmas[~np.isnan(a)] ** 2
    nom = reduce_all_but(weights * a, axes=axes, func=np.nansum, new_dphi_axis=new_dphi_axis)
    # nansum, yields 0 if all elements are nan. Fix this by setting such nan fields manualy
    denom = reduce_all_but(weights, axes=axes, func=np.nansum, new_dphi_axis=new_dphi_axis)
    _mask = reduce_all_but(weights, axes=axes, func=lambda a, axis: np.all(np.isnan(a), axis),
                           new_dphi_axis=new_dphi_axis)
    denom[_mask] = np.nan
    return nom / denom


def uncert_weighted_avg(sigmas, axis):
    """
    Calculate the uncertainty of a weighted average assuming purely
    statistical fluctuation between the combined values.
    Taken from Taylor "Introduction to error analysis" p. 176ff

    This function can be used wich c2_post.utils.reduce_all_but

    Parameters
    ----------
    sigmas : np.ndarray
        Uncertainties of the mean in each bin of the phase-space
    axis : int
        The axis over which we will combine our values.

    Returns
    -------
    np.ndarray : Uncertainties after combining the measurements over `axis`
    """
    weights = 1 / sigmas ** 2
    # nansum, yields 0 if all elements are nan. Fix this by setting such nan fields manualy
    nansum = np.nansum(weights, axis)
    if isinstance(nansum, np.ndarray):
        _mask = np.all(np.isnan(weights), axis)
        nansum[_mask] = np.nan
    return 1 / np.sqrt(nansum)


def weighted_mean(a, sigmas, **kwargs):
    """
    Compute the weighted arithmetic mean. `kwargs` are the same like np.nansum
    """
    w = 1 / sigmas**2
    return np.nansum(w * a, **kwargs) / np.nansum(w, **kwargs)


def weighted_std(a, sigmas, axis):
    """
    Weighted estimate of the standard deviation. This does not assume that each point is compatible with each other.

    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

    Parameters
    ----------
    a: ndarray
        Array of measured quantity
    sigmas : ndarray
        Standard deviation of each point in `a`
    axis : int
        Axis along which the weighted std is computed

    Returns
    -------
    ndarray : Weighted std of a along `axis`
    """
    # convert sigma to weights
    w = 1 / sigmas**2
    sum_w = np.nansum(w, axis)
    sum_w2 = np.nansum(w**2, axis)
    # keep the axis for broadcasting in the next step
    weighted_mean = np.nansum(w * a, axis, keepdims=True) / np.nansum(w, axis, keepdims=True)
    s2 = (np.nansum(w * (a - weighted_mean)**2, axis) /
          (sum_w - sum_w2 / sum_w))
    return np.sqrt(s2)


def merge_to_deta(a, sig):
    """
    Perform the coordinate transformation of eta_a and eta_b and merge
    along the "average-dimension". Expects the shape (eta, eta, ...)

    Parameters
    ----------
    a : np.ndarray
        Array of the measured quantity; shape=(eta, eta, ...)
    sig : np.ndarray
        Uncertainties of `a`; shape=(eta, eta, ...)

    Returns
    -------
    (np.array, np.array) : a_deta, sig_deta
        Merged values and uncertainties; shape=(deta, x, y, )
    """
    ratio_deta = reduce_weighted_avg(a, sig, [[0, 1], ] + range(2, a.ndim))
    ratio_deta_sig = reduce_all_but(sig, [[0, 1], ] + range(2, a.ndim), uncert_weighted_avg)
    return ratio_deta, ratio_deta_sig


def find_bin(edges, v):
    """Find the bin where `v` falls into"""
    if v < edges[0] or v > edges[-1]:
        raise ValueError("v is under/overflow")
    for i, edge in enumerate(edges):
        if v < edge:
            return i


def select_detector_region(a, eta_edges, reg1, reg2):
    """
    Return a view of the specified detectors in the given ndarray.

    Parameters
    a : np.ndarray
        shape must be (80, 80)
    eta_edges : np.ndarray
        Bin edges along eta; Note requirement of eta1 == eta2
    reg1, reg2 : str
        Three letter abbrevation of the region {bwd, its, fwd}
    """
    raise DeprecationWarning("Use of old `select_detector_region`;"
                             "use `eta_idxs_detector_region` instead")
    bwd_its_edge = -1.7
    its_fwd_edge = 1.7
    ieta1s_ieta2s = []
    if not (a.shape[0] == a.shape[1] == eta_edges.size - 1):
        raise ValueError("Incompatible shape and eta edges!")
    for reg in [reg1, reg2]:
        if reg == 'bwd':
            ieta1s_ieta2s.append(np.where(eta_edges < bwd_its_edge)[0])
        elif reg == 'its':
            ieta1s_ieta2s.append(np.where(np.logical_and(
                bwd_its_edge <= eta_edges,
                eta_edges < its_fwd_edge))[0])
        elif reg == 'fwd':
            ieta1s_ieta2s.append(np.where(eta_edges >= its_fwd_edge)[0])
        else:
            raise ValueError("Invalid region given!")
    eta1_slice = slice(ieta1s_ieta2s[0][0], ieta1s_ieta2s[0][-1] + 1)
    eta2_slice = slice(ieta1s_ieta2s[1][0], ieta1s_ieta2s[1][-1] + 1)
    return a[eta1_slice, eta2_slice, ...]


def eta_idxs_detector_region(eta_edges, reg1, reg2):
    """
    Return eta indices for the specified detectors in the given ndarray.

    Parameters
    a : np.ndarray
        shape must be (80, 80)
    eta_edges : np.ndarray
        Bin edges along eta; Note requirement of eta1 == eta2
    reg1, reg2 : str
        Three letter abbrevation of the region {bwd, its, fwd}
    """
    bwd_its_edge = -1.7
    its_fwd_edge = 1.7
    ieta1s_ieta2s = []
    # if not (a.shape[0] == a.shape[1] == eta_edges.size - 1):
    #     raise ValueError("Incompatible shape and eta edges!")
    for reg in [reg1, reg2]:
        if reg == 'bwd':
            ieta1s_ieta2s.append(np.where(eta_edges < bwd_its_edge)[0])
        elif reg == 'its':
            ieta1s_ieta2s.append(np.where(np.logical_and(
                bwd_its_edge <= eta_edges,
                eta_edges < its_fwd_edge))[0])
        elif reg == 'fwd':
            ieta1s_ieta2s.append(np.where(eta_edges >= its_fwd_edge)[0])
        else:
            raise ValueError("Invalid region given!")
    eta1_slice = slice(ieta1s_ieta2s[0][0], ieta1s_ieta2s[0][-1] + 1)
    eta2_slice = slice(ieta1s_ieta2s[1][0], ieta1s_ieta2s[1][-1] + 1)
    return eta1_slice, eta2_slice


def get_dead_regions_map(eta_edges, phi_edges, z_edges, max_res_counts, max_res_edges):
    """
    Find which bins of the given edges include any dead detector regions.
    Uses a histogram constructed from the counts in (eta, phi, z) at maxial resolution.

    Parameters
    ----------
    eta_edges, phi_edges, z_edges : np.ndarray
        Edges of the region of interest.
    max_res_counts : np.array
        ndarray containing the counts of particles at the highest resolution. Shape: (eta, phi, z)
    max_res_edges : list
        List of Edges for the max_res_counts ndarray (eta, phi, z)

    Returns
    -------
    ndarray :
        bool array with dimensions of `like_h`. False, if a bin includes a dead region.
    """
    def find_index(val, fine_edges):
        try:
            # FIXME: This fails if we the fine bins are not a multiple of the corse ones!
            return np.where(np.round(fine_edges, 2) == np.round(val, 2))[0][0]
        except IndexError:
            return None
    # renaming some things
    fine_vals, (fine_etas, fine_phis, fine_zs) = max_res_counts, max_res_edges
    shape = (eta_edges.shape[0] - 1, phi_edges.shape[0] - 1, z_edges.shape[0] - 1)
    mask = np.full(shape, fill_value=False, dtype=np.bool)
    for i, (low_eta, up_eta) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):
        for j, (low_phi, up_phi) in enumerate(zip(phi_edges[:-1], phi_edges[1:])):
            for k, (low_z, up_z) in enumerate(zip(z_edges[:-1], z_edges[1:])):
                slice_eta = slice(find_index(low_eta, fine_etas), find_index(up_eta, fine_etas))
                slice_phi = slice(find_index(low_phi, fine_phis), find_index(up_phi, fine_phis))
                slice_z = slice(find_index(low_z, fine_zs), find_index(up_z, fine_zs))
                mask[i, j, k] = np.min(fine_vals[slice_eta, slice_phi, slice_z]) > 0
    return mask


def edges2centers(edges):
    centers = [np.mean([e1, e2]) for (e1, e2) in zip(edges[:-1], edges[1:])]
    return np.array(centers)


def ratio_pure(vnn, vn, vnn_sig, vn_sig):
    """
    Compute the factorization ratio for pure factorization. It makes sense to do the
    factorization on the caller side, since its not apriori clear
    which phase-space regions to include.

    The returned uncertainties are purely based on the uncertainties
    of v_nn, not on the uncertainties of the fit v_n

    Returns
    -------
    ndarray : ratio
    ndarray : sigma of ratio
    """
    vn_rel_sig = vn_sig / vn
    vnn_rel_sig = vnn_sig / vnn
    rel_sig = np.sqrt((vn_rel_sig[:, None, ...]**2 + vn_rel_sig[None, :, ...]**2) + vnn_rel_sig**2)
    ratio = vnn / (vn[:, None, ...] * vn[None, :, ...])
    return ratio, ratio * rel_sig  # vnn_rel_sig


def ratio_twist_v2(vnn, vnn_sig, vn, twist, eta_edges):
    """
    Ratio of measurement to fit for the twist model; v2 only!
    Returns
    -------
    (ratio, ratio_sig): (nd.array, ndarray)
        Each has shape (eta, eta, cent, z); NOTE: no `n`!!!
    """
    vnn_rel_sig = vnn_sig / vnn
    ratio = vnn[:, :, 2, ...] / (vn[:, None, ...] * vn[None, :, ...])
    # Broadcast to z-dimension
    ratio *= exp_twist(twist, eta_edges)[:, :, :, None]
    # should include twist uncert?
    ratio_sig = ratio * vnn_rel_sig[:, :, 2, ...]
    return ratio, ratio_sig


def mask_diagonal(a, k=10):
    """
    Return a copy of `a` where the (lower) diagonal is set to NaN
    """
    import warnings
    warnings.warn("Use `deta_mask` instead!", DeprecationWarning)
    # eta gap
    # k = 20 # eta width 0.1
    # k = 15 # eta width 0.2
    a = np.copy(a)
    indices = np.tril_indices(n=a.shape[0], m=a.shape[1], k=k)
    a[indices[0], indices[1], ...] = np.nan
    return a


def fmd_mask(eta_edges):
    """
    Mask for the short range FMD region in the (eta_a, eta_b)-plane. Masked == True
    """
    shape = (eta_edges.size - 1, eta_edges.size - 1)
    fmds_mask = np.full(shape, fill_value=np.False_)
    idxs_eta1, idxs_eta2 = eta_idxs_detector_region(eta_edges, 'fwd', 'fwd')
    fmds_mask[idxs_eta1, idxs_eta2] = True
    idxs_eta1, idxs_eta2 = eta_idxs_detector_region(eta_edges, 'bwd', 'bwd')
    fmds_mask[idxs_eta1, idxs_eta2] = True
    # retunr as masked array for easier plotting
    return fmds_mask


def kth_diag_indices(n, k):
    rows, cols = np.diag_indices(n)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def deta_mask(eta_edges, nbins_deta_gap):
    """
    Return a mask, where the `|\Delta\eta| < k*\eta-binwidth` is `True`
    """
    eta_eta_shape = (eta_edges.size - 1, eta_edges.size - 1)
    excld_mask = np.full(eta_eta_shape, fill_value=np.False_)
    for k in range(-nbins_deta_gap, nbins_deta_gap + 1):
        eta_1, eta_2 = kth_diag_indices(eta_eta_shape[0], k)
        excld_mask[eta_1, eta_2, ...] = True
    # retunr as masked array for easier plotting
    return np.ma.masked_where(excld_mask, excld_mask)


def exp_twist(twist, eta_edges):
    """Compute `e^{-twist * |eta_1 - eta_2|}

    Parameters
    ----------
    twist: np.array
        Empircal twist parameter vs. cent
    eta_edges: np.array
        eta edges used in ananlysis

    Returns
    -------
    np.ndarray:
        Decorrelation value on the (eta_a, eta_b, cent)-grid
    """
    eta_centers = edges2centers(eta_edges)
    detas = np.abs(eta_centers[:, None] - eta_centers[None, :])
    return np.exp(twist[None, None, :] * detas[:, :, None])


# def fact_ratio(vnn, vnn_sig, eta_edges, nbins_deta_gap, with_twist,
#                exclude_short_range_fmd):
#     fact_me = np.copy(vnn.data)
#     fact_me[vnn.mask] = np.nan
#     if exclude_short_range_fmd:
#         (eta1_idxs, eta2_idxs) = eta_idxs_detector_region(eta_edges, 'bwd', 'bwd')
#         fact_me[eta1_idxs, eta2_idxs, ...] = np.nan
#         (eta1_idxs, eta2_idxs) = eta_idxs_detector_region(eta_edges, 'fwd', 'fwd')
#         fact_me[eta1_idxs, eta2_idxs, ...] = np.nan

#     fact_me = mask_diagonal(fact_me, k=nbins_deta_gap)
#     if not with_twist:
#         vn, _vn_sig = find_v(fact_me, sigma=vnn_sig)
#         ratio, ratio_sig = vnn_div_vnvn(vnn, vn, vnn_sig)
#         ratio = ratio[:, :, 2, ...]
#         ratio_sig = ratio_sig[:, :, 2, ...]
#     else:
#         (vn, twist), (vn_sig, twist_sig) = find_v2_twist(fact_me, vnn_sig, )
#         ratio, ratio_sig = vnn_div_vnvn(vnn[:, :, 2, ...], vn, vnn_sig[:, :, 2, ...])
#         # multiply with the exp-part
#         eta_centers = edges2centers(eta_edges)
#         detas = np.abs(eta_centers[:, None] - eta_centers[None, :])
#         ratio *= np.exp(twist[None, None, :, None] * detas[:, :, None, None])
#     return ratio, ratio_sig


def compute_pure_vns(vnn, vnn_sig, eta_edges, exclude_fmd, gaps_nbins):
    """
    Compute v_n pure-factorization-model

    Expects numpy array; not masked arrays!
    """
    logging.info("computing vns")
    if isinstance(vnn, np.ma.MaskedArray):
        vnn = np.ma.filled(vnn, np.nan)
    if isinstance(vnn_sig, np.ma.MaskedArray):
        vnn_sig = np.ma.filled(vnn_sig, np.nan)

    vns, vns_sig = [], []
    for nbins in gaps_nbins:
        fact_me = np.copy(vnn)
        fact_me[deta_mask(eta_edges, nbins), ...] = np.nan
        if exclude_fmd:
            fact_me[fmd_mask(eta_edges), ...] = np.nan
        vn, vn_sig = find_v(fact_me, sigma=vnn_sig, )
        vns.append(vn)
        vns_sig.append(vn_sig)
    return np.array(vns), np.array(vns_sig)


def compute_twist_v2s(vnn, vnn_sig, eta_edges, exclude_fmd, gaps_nbins, nstraps=5):
    """
    Compute v_2 and the event plane twist with the twist-model
    """
    logging.info("computing vns with twist model")
    if isinstance(vnn, np.ma.MaskedArray):
        vnn = np.ma.filled(vnn, np.nan)
    if isinstance(vnn_sig, np.ma.MaskedArray):
        vnn_sig = np.ma.filled(vnn_sig, np.nan)
    v2s, v2s_sig = [], []
    twists, twists_sig = [], []

    for nbins in gaps_nbins:
        fact_me = np.copy(vnn)
        fact_me[deta_mask(eta_edges, nbins), ...] = np.nan

        if exclude_fmd:
            fact_me[fmd_mask(eta_edges), ...] = np.nan

        # Twist currently only implemented for n=2!!!!!!! OBS: Shape has no `n`!!!
        (v2, twist), (v2_sig, twist_sig) = find_v2_twist(fact_me, vnn_sig=vnn_sig, eta_edges=eta_edges, nstraps=nstraps)
        v2s.append(v2)
        v2s_sig.append(v2_sig)
        twists.append(twist)
        twists_sig.append(twist_sig)
    return np.array(v2s), twists, np.array(v2s_sig), twists_sig


def make_delta_centers(edges1, edges2):
    """
    Convert coordinates edges as edges2 - edges1
    """
    cent1 = np.array([np.mean([low, up])
                      for low, up in zip(edges1[:-1], edges1[0:])])
    cent2 = np.array([np.mean([low, up])
                      for low, up in zip(edges2[:-1], edges2[0:])])
    tmp = (cent2[:, None] - cent1[None, :])
    return np.unique(np.round(tmp.flatten(), 2))
