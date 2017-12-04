import warnings
import numpy as np


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
