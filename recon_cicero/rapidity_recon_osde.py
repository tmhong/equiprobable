#from wigner import wigner_dl
from scipy.special import sph_harm_y, hermite, legendre
#from sympy.physics.wigner import wigner_d as wig_d
from scipy.special import gamma
from itertools import product
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import argparse
import uproot
import os
import spherical
import sys
import quaternionic

def factorial(x):
    return gamma(x + 1)

def wigner_d(j, m1, m2, alpha, beta, gamma):
    R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
    ell = int(round(j))  # max ell for Wigner object
    wigner = spherical.Wigner(ell)
    D = wigner.D(R)
    # D_index takes half-integer values as 2*j integers
    idx = wigner.Dindex(j, int(m1), int(m2))
    if D.ndim > 1:
        return D[..., idx]
    return D[idx]

def M_function(phi, theta, phistar, thetastar, j1, j2, m1, m2):
    return (1 / (4 * np.pi)) * np.sqrt(2 * j1 + 1) * np.sqrt(2 * j2 + 1) * wigner_d(j1, m1, m2, phi, theta, 0) * wigner_d(j2, m2, 0, phistar, thetastar, 0)

def hermite_func(x, n):
    return hermite(n)(x) * np.exp(-x**2 / 2) / np.sqrt((2**n)*factorial(n)*np.sqrt(np.pi))

def legendre_func(x, n):
    return np.sqrt(((2*n +1) / 2)) * legendre(n)(x)

def sph_harm(theta, phi, n, m):
    if phi < 0:
        phi += 2*np.pi

    return sph_harm_y(n, m, theta, phi)

top_basis = {}
antitop_basis = {}
hermite_basis = {}
legendre_basis = {}

#data first, params second
def osde(func, data, param_list, xrange, axis=None, samples=None,data_idx = 0,jacobian=lambda _ : 1, filepath=None, plotting_data=[], ylabel="Probability", xlabel=""):
    logname = func.__name__
    print(f"Performing fits for {logname}")
    if filepath is None:
        filepath = f"plots/osde/{logname}_{data_idx}"
    n_events = len(data)

    coeffs = np.zeros(len(param_list), dtype=np.complex128)
    cov = np.zeros((len(param_list), len(param_list)), dtype=np.complex128)

    for event in range(n_events):
        entry = data[event]
        M = np.array([np.conj(func(*entry, *params)).item() for params in param_list])
        coeffs += M
        cov += np.outer(M, np.conj(M))

    cov /= n_events
    cov -= np.outer(coeffs, np.conj(coeffs))
    cov /= n_events

    print(f"fitted {logname} coefficients:")
    for i , coeff in enumerate(coeffs):
        print(f"{param_list[i]}:  {coeff}")

    pred = np.zeros_like(xrange, dtype=np.complex128)
    entries = np.array(samples).copy()

    if data.shape[1] == 1:
        for coeff, params in zip(coeffs, param_list):
            pred += coeff * func(xrange, *params)
    else:
        for i,x in enumerate(xrange):
            entries[data_idx] = x
            total = 0
            for coeff,params in zip(coeffs,param_list):
                    total += coeff * np.sum(func(*entries, *params))

            pred[i]=total

    pred = np.real(pred)

    pred *= jacobian(xrange)

    integral = np.trapezoid(pred, xrange)
    pred /= integral
    print(f"Integral is {integral}")

    axis.plot(xrange, pred, label=f"OSDE Fitted {logname}", color='b')

    # Histogram as counts
    counts, bins = np.histogram(plotting_data, bins=50)

    # Plot normalized histogram for display only
    axis.hist(
        plotting_data,
        bins=bins,
        density=True,
        alpha=0.5,
        color='y',
        label="Measured data"
    )

    # Compute expected counts in each bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)
    N = np.sum(counts)

    expected = np.interp(bin_centers, xrange, pred) * bin_widths * N

    # Ignore empty bins to avoid divide-by-zero
    mask = expected > 0

    chi2 = np.sum((counts[mask] - expected[mask])**2 / expected[mask])

    ndof = np.count_nonzero(mask) - len(coeffs)
    red_chi2 = chi2 / ndof

    print(f"Chi² = {chi2:.3f}")
    print(f"Reduced Chi² = {red_chi2:.3f}")

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{logname} -- {xlabel}")
    axis.legend()

    errors = np.sqrt(np.diag(cov))

    print(f"---- FINAL REAL COEFFICIENTS ----")
    for i, (param, coeff) in enumerate(zip(param_list, coeffs)):
        print(f"  {param}: {np.real(coeff):.6f} +/- {np.real(errors[i]):.6f}")

    return red_chi2


def plotMarginalOSDEprod(func_name, coeffs, basis_funcs, basis_params, xrange, samples, plotting_data, func_idx_map, volume=1,jacobian=lambda _ : 1, xname="",data_idx=0,axis=None):
    if xname == "":
        xname = str(data_idx)
    print(f"Plotting marginal for {func_name} -- {xname}")

    indices_list = ['i','j','k','l']
    func_idx = func_idx_map[data_idx]
    data_idx = data_idx % len(samples[func_idx])
    nfuncs = len(basis_funcs)
    param_lengths = [len(l) for l in basis_params]

    pred = np.zeros_like(xrange, dtype=np.complex128)

    mc_x = np.zeros((param_lengths[func_idx], len(xrange)), dtype=np.complex128)
    mc_samples = [np.zeros((num_params, len(samples[0][0])), dtype=np.complex128) for num_params in param_lengths]

    #Need to calculate a basis for testing
    for i, param_set in enumerate(basis_params): 
        for j, param in enumerate(param_set):
            func_index = i % nfuncs
            if func_index == func_idx:
                entries = samples[func_idx]
                if np.array(samples[func_idx]).ndim > 1:
                    args = [np.array(a)[None, :] for a in samples[func_idx]]
                    args[data_idx] = np.array(xrange[:,None])
                    vals = basis_funcs[func_idx](
                        *args, *param
                    )
                    mc_x[j] = np.mean(vals, axis=1)
                else:
                    mc_x[j] = basis_funcs[func_idx](
                        xrange,
                        param
                    )
            else:
                if np.array(samples[func_index]).ndim > 1:
                    mc_samples[func_index][j] = basis_funcs[func_index](*samples[func_index], *param)
                else:
                    mc_samples[func_index][j] = basis_funcs[func_index](samples[func_index], param)

    indices1 = ''.join(indices_list[:len(basis_funcs)])
    for i in range(len(basis_funcs)):
        if i == func_idx:
            continue
        else:
            indices1 += f",{indices_list[i]}m"
    indices1 += f"->{indices_list[func_idx]}m"

    tmp = np.einsum(
        indices1,
        coeffs,
        *mc_samples[:func_idx],
        *mc_samples[func_idx+1:],
        optimize=True
    )
    pred = np.einsum(
        "im,ix->xm",
        tmp,
        mc_x,
        optimize=True
    )

    pred = np.real(pred.mean(axis=1,))
    pred *= volume / (max(xrange) - min(xrange))
    pred *= jacobian(xrange)

    mask = (xrange >= min(plotting_data)) & (xrange <= max(plotting_data))

    integral = np.trapezoid(pred[mask], np.linspace(min(plotting_data), max(plotting_data), len(pred[mask])))
    pred /= integral


    axis.plot(xrange, pred, label=f"OSDE Fitted {func_name}",color='b')

    # Histogram as counts
    counts, bins = np.histogram(plotting_data, bins=50)

    # Plot normalized histogram for display only
    axis.hist(
        plotting_data,
        bins=bins,
        density=True,
        alpha=0.5,
        color='y',
        label="Measured data"
    )

    # Compute expected counts in each bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)
    N = np.sum(counts)

    expected = np.interp(bin_centers, xrange, pred) * bin_widths * N

    # Ignore empty bins to avoid divide-by-zero
    mask = expected > 0

    chi2 = np.sum((counts[mask] - expected[mask])**2 / expected[mask])

    ndof = np.count_nonzero(mask) - len(coeffs)
    red_chi2 = chi2 / ndof

    print(f"Chi² = {chi2:.3f}")
    print(f"Reduced Chi² = {red_chi2:.3f}")

    axis.set_xlabel(xname)
    axis.set_ylabel("Probability")
    axis.set_title(f"{func_name} -- {xname}")

    return plotting_data, pred, red_chi2

#From a basis of other orthogonal functions, perform an osde on the tensor product
def osdeFromProd(logname, precalculated_data, param_tuples):
    print(f"Performing fits for {logname}")
    n_events = len(precalculated_data[0])

    coeffs = np.zeros(len(param_tuples), dtype=np.complex128)
    cov = np.zeros((len(param_tuples), len(param_tuples)), dtype=np.complex128)

    n_events = precalculated_data[0].shape[0]
    batch_size = 250  

    shape = (
        precalculated_data[0].shape[1],
        precalculated_data[1].shape[1],
        precalculated_data[2].shape[1],
        precalculated_data[3].shape[1],
    )

    sum_prod = np.zeros(shape, dtype=np.complex128)
    sum_sq = np.zeros(shape, dtype=np.float64)

    #This matrix will explode your ram if you don't batch
    for start in range(0, n_events, batch_size):
        end = min(start + batch_size, n_events)

        batch = [x[start:end] for x in precalculated_data]

        prod_batch = np.einsum(
            "ei,ej,ek,el->eijkl",
            *batch,
            optimize=True,
        )

        # accumulate sums
        sum_prod += np.sum(prod_batch, axis=0)
        sum_sq += np.sum(np.abs(prod_batch)**2, axis=0)

    # coefficient mean
    coeffs = sum_prod / n_events

    # variance of each coefficient
    variance = (sum_sq / n_events) - np.abs(coeffs)**2

    # numerical precision can make tiny negatives
    variance = np.maximum(variance, 0)

    # standard error of the mean
    errs = np.sqrt(variance / n_events)

    # significance pruning
    coeffs[np.abs(coeffs) < 3*errs] = 0

    n_kept = np.sum(np.abs(coeffs) >= 3*errs)
    print(f"coefficients kept: {n_kept} / {coeffs.size}  ({100*n_kept/coeffs.size:.1f}%)")

    np.save("osde_saved/coeffs.npy", coeffs)

    return coeffs

def evalOSDEProd(coeffs, data, basis_funcs, basis_params):

    print("Evaluating...")
    
    param_sizes = [len(l) for l in basis_params]

    nfuncs = len(basis_funcs)
    evaluated = [np.zeros((param_size, len(data[0][0])), dtype=np.complex128) for param_size in param_sizes]

    for i,param_set in enumerate(basis_params):
        for j,param in enumerate(param_set):
            if np.array(data[i]).ndim > 1 :
                evaluated[i][j] = basis_funcs[i](*data[i], *param)
            else:
                evaluated[i][j] = basis_funcs[i](data[i], param)

    final = np.real(np.einsum(
        "ijkl,ix,jx,kx,lx->x",
        coeffs,
        *evaluated
    ))

    return final

def printAssignmentAccuracy(path, tag):
    flag_branches = ["truth_matched", "bassign_correct", "downtype_correct",
                     "nu_branch_correct", "nu_real"]
    mass_branches = ["m_top_lep", "m_top_had", "m_w_had", "m_w_lep", "m_bb",
                     "chi2", "nu_pz_reco", "nu_pz_truth"]
    try:
        with uproot.open(path) as file:
            tree = file["ttbar"]
            keys = [k.split(";")[0] for k in tree.keys()]
            present = [b for b in (flag_branches + mass_branches) if b in keys]
            if not present:
                print(f"[{tag}] no validation branches in this tree "
                      f"(was it made by the old macro?)")
                return
            d = tree.arrays(present, library="np")
    except Exception as exc:
        print(f"[{tag}] could not read validation branches: {exc}")
        return

    n = len(d[present[0]])
    print(f"\n===== ASSIGNMENT ACCURACY: {tag} =====")
    print(f"  events in tree                : {n}")

    if "nu_real" in d:
        v = d["nu_real"]
        print(f"  real neutrino solution        : {100.0*np.sum(v == 1)/len(v):5.1f}%"
              f"   (failures are exactly mT > m_target)")
    if "truth_matched" in d:
        v = d["truth_matched"]
        print(f"  truth-matched events          : {100.0*np.sum(v == 1)/len(v):5.1f}%"
              f"   ({np.sum(v == 1)}/{len(v)})")

    for name, label, ref in [
            ("bassign_correct",   "b-jet assignment correct", "[ref 62-80%]"),
            ("downtype_correct",  "down-type jet correct",    "[ref ~61%]"),
            ("nu_branch_correct", "neutrino branch correct",  "")]:
        if name not in d:
            continue
        v = d[name]
        tried = int(np.sum(v >= 0))
        if tried == 0:
            print(f"  {label:<30}: N/A (no truth definition, or no matched events)")
            continue
        ok = int(np.sum(v == 1))
        print(f"  {label:<30}: {100.0*ok/tried:5.1f}%  ({ok}/{tried}) {ref}")

    for name, label in [("m_top_lep", "m(l nu b)"), ("m_top_had", "m(j j b)"),
                        ("m_w_had", "m(j j)"), ("m_w_lep", "m(l nu)"),
                        ("m_bb", "m(b b)")]:
        if name in d:
            v = d[name][np.isfinite(d[name])]
            if len(v):
                print(f"  {label:<30}: mean {np.mean(v):7.2f}   RMS {np.std(v):6.2f} GeV")

    if "nu_pz_reco" in d and "nu_pz_truth" in d:
        mask = d["nu_pz_truth"] > -9000.0
        if np.sum(mask) > 0:
            res = d["nu_pz_reco"][mask] - d["nu_pz_truth"][mask]
            print(f"  {'pz(nu) reco - truth':<30}: mean {np.mean(res):7.2f}   "
                  f"RMS {np.std(res):6.2f} GeV")
    print("=" * 46 + "\n")


def main():
    print("Beginning Orthogonal Series Density Estimation fitting...")

    parser = argparse.ArgumentParser(description='Perform orthogonal series density estimation fitting on a ROOT file.')
    parser.add_argument('--top_file', type=str, default ='/home/cicero/Pitt/Hong/sim-tools/analysis/rootFiles/ttbar_angles_truth_recon.root')
    parser.add_argument('--hh_file', type=str, default ='/home/cicero/Pitt/Hong/sim-tools/analysis/rootFiles/hh_angles_truth_recon.root')
    args = parser.parse_args()

    printAssignmentAccuracy(args.top_file, "ttbar")
    printAssignmentAccuracy(args.hh_file, "di-Higgs")

    test_split = 0.8

    with uproot.open(args.top_file) as file:
        tree = file["ttbar"] 

        t_branches = tree.arrays(["t_theta", "t_phi", "t_thetaprime", "t_phistar",
                                 "t~_theta", "t~_phi", "t~_thetastar", "t~_phistar", 
                                 "rapidity", "cosprodangle"], library="np")
        
        t_branches["t_theta"] = np.arccos(np.clip(t_branches["t_theta"], -1.0,1.0))
        t_branches["t_thetaprime"] = np.arccos(np.clip(t_branches["t_thetaprime"], -1.0,1.0))
        t_branches["t~_theta"] = np.arccos(np.clip(t_branches["t~_theta"], -1.0,1.0))
        t_branches["t~_thetastar"] = np.arccos(np.clip(t_branches["t~_thetastar"], -1.0,1.0))


    t_train = {}
    t_test = {}

    _n_t = len(t_branches["t_theta"])
    _perm_t = np.random.permutation(_n_t)
    _cut_t = int(test_split * _n_t)

    for branch in t_branches:
        t_branches[branch] = t_branches[branch][_perm_t]
        t_train[branch] = t_branches[branch][:_cut_t]
        t_test[branch] = t_branches[branch][_cut_t:]

    #for branch in t_branches:
        
        #t_branches[branch] = np.random.permutation(t_branches[branch])
        #t_train[branch] = t_branches[branch][:int(test_split * len(t_branches[branch]))]
        #t_test[branch] = t_branches[branch][int(test_split * len(t_branches[branch])):]

    with uproot.open(args.hh_file) as file:
        tree = file["ttbar"] 

        h_branches = tree.arrays(["t_theta", "t_phi", "t_thetaprime", "t_phistar",
                                    "t~_theta", "t~_phi", "t~_thetastar", "t~_phistar", 
                                    "rapidity", "cosprodangle"], library="np")

        h_branches["t_theta"] = np.arccos(np.clip(h_branches["t_theta"], -1.0,1.0))
        h_branches["t_thetaprime"] = np.arccos(np.clip(h_branches["t_thetaprime"], -1.0,1.0))
        h_branches["t~_theta"] = np.arccos(np.clip(h_branches["t~_theta"], -1.0,1.0))
        h_branches["t~_thetastar"] = np.arccos(np.clip(h_branches["t~_thetastar"], -1.0,1.0))



    h_train = {}
    h_test = {}

    _n_h = len(h_branches["t_theta"])
    _perm_h = np.random.permutation(_n_h)
    _cut_h = int(test_split * _n_h)

    for branch in h_branches:
        h_branches[branch] = h_branches[branch][_perm_h]
        h_train[branch] = h_branches[branch][:_cut_h]
        h_test[branch] = h_branches[branch][_cut_h:]

    #for branch in h_branches:
        #h_branches[branch] = np.random.permutation(h_branches[branch])
        #h_train[branch] = h_branches[branch][:int(test_split * len(h_branches[branch]))]
        #h_test[branch] = h_branches[branch][int(test_split * len(h_branches[branch])):]

    n_events = len(t_branches["t_theta"])
    print(f"Fitting for {n_events} events")

    n_mc = 2000
    n_x = 500
    plots_dir = "plots/osde"

    m_parameters = [(0,0,0,0), (1,0,0,0), (1,0,1,0), (1,0,-1,0), 
                    (0,1,0,0), (1,1,0,0), (1,1,1,0), (1,1,-1,0), 
                    (1,1,0,1), (1,1,0,-1), (1,1,1,1), (1,1,-1,-1), 
                    (1,1,1,-1), (1,1,-1,1), (0,2,0,0), (1,2,0,0), 
                    (1,2,1,0), (1,2,-1,0), (1,2,0,1), (1,2,0,-1), 
                    (1,2,1,1), (1,2,1,-1), (1,2,-1,1), (1,2,-1,-1) ] 
    nN = 5

    top_basis = np.column_stack([
        np.conj(M_function(t_train["t_phi"], t_train["t_theta"], t_train["t_phistar"], t_train["t_thetaprime"], *p))
        for p in m_parameters
    ])

    topbar_basis = np.column_stack([
        np.conj(M_function(t_train["t~_phi"], t_train["t~_theta"], t_train["t~_phistar"], t_train["t~_thetastar"], *p))
        for p in m_parameters
    ])

    legendre_basis = np.column_stack([
        legendre_func(t_train["cosprodangle"], n)
        for n in range(nN)
    ])

    hermite_basis = np.column_stack([
        hermite_func(t_train["rapidity"], n)
        for n in range(nN)
    ])

    plotting_data = np.column_stack([
        t_branches["t_phi"], t_branches["t_theta"], t_branches["t_phistar"], t_branches["t_thetaprime"],
        t_branches["t~_phi"], t_branches["t~_theta"], t_branches["t~_phistar"], t_branches["t~_thetastar"],
        t_branches["rapidity"], t_branches["cosprodangle"]
    ])

    beaudreau_parameters = beaudreau_parameters = [m_parameters,m_parameters,range(nN),range(nN)]
    beaudreau_precalculated = [top_basis, topbar_basis, hermite_basis, legendre_basis]
    beaudreau_basis = [M_function, M_function, hermite_func, legendre_func]
    basis_idx_map = [0,0,0,0,1,1,1,1,2,3]

    beaudreau_mc_samples = [
        (np.random.uniform(-np.pi, np.pi, n_mc), np.arccos(np.random.uniform(-1,1, n_mc)),
        np.random.uniform(-np.pi, np.pi, n_mc), np.arccos(np.random.uniform(-1, 1, n_mc))),
        (np.random.uniform(-np.pi, np.pi, n_mc), np.arccos(np.random.uniform(-1,1, n_mc)),
        np.random.uniform(-np.pi, np.pi, n_mc), np.arccos(np.random.uniform(-1,1, n_mc))),
        (np.random.uniform(-np.pi, np.pi, n_mc)), (np.random.uniform(-1, 1, n_mc))
    ]

    pred_ranges = [
        np.linspace(-np.pi, np.pi, n_x), np.linspace(0, np.pi, n_x), 
        np.linspace(-np.pi, np.pi, n_x), np.linspace(0, np.pi, n_x), 
        np.linspace(-np.pi, np.pi, n_x), np.linspace(0, np.pi, n_x), 
        np.linspace(-np.pi, np.pi, n_x), np.linspace(0, np.pi, n_x), 
        np.linspace(-3.1, 3.1, n_x), np.linspace(-1, 1, n_x)
    ]

    volume = 1
    for range_i in pred_ranges:
        volume *= max(range_i) - min(range_i)
    
    identity = lambda _ : 1
    jacobians = [identity, np.sin, 
                 identity, np.sin, 
                 identity, np.sin, 
                 identity, np.sin, 
                 identity, identity]

    labels=[r"$\phi$", r"$\theta$", r"$\phi*$",r"$\theta*$",
            r"$\tilde{\phi}$",r"$\tilde{\theta}$",r"$\tilde{\phi'}$",r"$\tilde{\theta'}$",
            r"Y",r"$\Theta$"]

    
    coeffs = osdeFromProd("Combined Functions", beaudreau_precalculated, beaudreau_parameters)

    chi_2s = {}
    plt_data = {}
    preds = {}

    # fig, axs = plt.subplots(2, 5, figsize=(60, 32))
    # axs_ids = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3),(0,4),(1,4)]
    # fig.suptitle(r"OSDE Fits 1D Projection", fontsize=24)

    # for i in range(10):
    #     plot_data, pred, chi_2 = plotMarginalOSDEprod("Beaudreau Functions", coeffs, beaudreau_basis, beaudreau_parameters, pred_ranges[i], beaudreau_mc_samples,
    #                         plotting_data[:, i], basis_idx_map, volume=volume, axis = axs[*axs_ids[i]],data_idx=i,
    #                         xname=labels[i], jacobian=jacobians[i])
        
    #     plt_data[f"{labels[i]}_truth"] = plot_data
    #     preds[f"{labels[i]}_pred"] = pred
    #     chi_2s[f"{labels[i]}_chi_2"] = chi_2

    # plt.tight_layout()
    # plt.savefig(f"{plots_dir}/beadreau_function.png")
    # plt.close()

    # np.savez("osde_saved/1d_projections.npz", **plt_data, **preds)

    # plt.figure(figsize=(8, 6))
    # plt.bar(range(len(chi_2s)), chi_2s, color='b', alpha=0.7)
    # plt.xticks(range(len(chi_2s)), ['phi', 'theta', 'phi*', 'theta*','antiphi', 'antitheta', 'antiphi*', 'antitheta*', 'rapidity', 'cosprodangle'], rotation=45)
    # plt.ylabel(r'$\chi^2_{\text{red}}$')
    # plt.title(r'$\chi^2_{\text{red}}$ for OSDE Fits')
    # plt.savefig(f"{plots_dir}/reduced_chi2.png")
    # plt.close()

    h_data = [
        (h_test["t_phi"], h_test["t_theta"], h_test["t_phistar"], h_test["t_thetaprime"]),
        (h_test["t~_phi"], h_test["t~_theta"], h_test["t~_phistar"], h_test ["t~_thetastar"]),
        (h_test["rapidity"]), (h_test["cosprodangle"])
    ]

    t_data = [
        (t_test["t_phi"], t_test["t_theta"], t_test["t_phistar"], t_test["t_thetaprime"]),
        (t_test["t~_phi"], t_test["t~_theta"], t_test["t~_phistar"], t_test ["t~_thetastar"]),
        (t_test["rapidity"]), (t_test["cosprodangle"])
    ]


    vals_H = evalOSDEProd(coeffs, h_data, beaudreau_basis, beaudreau_parameters)
    vals_t = evalOSDEProd(coeffs, t_data, beaudreau_basis, beaudreau_parameters)

    np.savez("osde_saved/g_vals.npz", tt_g=vals_t, hh_g=vals_H)

    all_vals = np.concatenate([vals_t, vals_H])

    bins = np.linspace(
        np.min(all_vals),
        np.max(all_vals),
        101
    )

    nt, bins_t = np.histogram(vals_t, bins=bins, density=True)
    nh, bins_h = np.histogram(vals_H, bins=bins, density=True)

    plt.stairs(nt, bins_t, label="ttbar", alpha=0.7)
    plt.stairs(nh, bins_h, label="HH", alpha=0.7)

    plt.title(r"Predicted $g(\boldsymbol{\Theta})$ From OSDE Fitted Discriminator")
    plt.xlabel(r"$g(\boldsymbol{\Theta})$")
    plt.ylabel(f"Events / {bins[1]-bins[0]:.1e}")
    plt.legend()
    plt.savefig(f"{plots_dir}/beaudreau_function_out.png")
    plt.close()


    eps_tt = []
    eps_hh = []
    for i in range(len(bins)):
        eps_tt.append(np.sum(nt[:i]/np.sum(nt)))
        eps_hh.append(np.sum(nh[:i]/np.sum(nh)))


    np.savez("osde_saved/roc.npz", eps_tt=np.array(eps_tt), eps_hh=np.array(eps_hh))

    plt.plot(eps_hh, eps_tt, label="OSDE ROC")
    plt.title(r"ROC Curve OSDE")
    plt.ylabel(r"$\epsilon_{t\bar{t}}$")
    plt.xlabel(r"$\epsilon_{hh}$")
    plt.legend()
    plt.savefig(f"{plots_dir}/osde_roc_hh.png")
    plt.close()


if __name__ == "__main__":
    main()


#just storing these out of the way

# mdata = [(p,t,ps,ts,) for p,t,ps,ts in zip(branches["tbar_phi"], branches["tbar_theta"], branches["tbar_phistar"], branches["tbar_thetastar"])]
# mdatabar = [(p,t,ps,ts,) for p,t,ps,ts in zip(branches["t_phi"], branches["t_theta"], branches["t_phistar"], branches["t_thetastar"])]
# m_mc_samples = [np.random.uniform(-np.pi, np.pi,n_x), np.cos(np.random.uniform(-1, 1,n_x)), np.random.uniform(-np.pi, np.pi, n_x), np.cos(np.random.uniform(-1, 1, n_x))]

# chi_2s[0] = osde(M_function, np.array(mdata), m_parameters, np.linspace(-np.pi, np.pi), axis = axs[0,0], samples=m_mc_samples,data_idx=0,filepath=f"{plots_dir}/m_function_phi.png", xlabel=r"$\phi$")
# chi_2s[1] = osde(M_function, np.array(mdata), m_parameters, np.linspace(0, np.pi), axis = axs[0,1], samples = m_mc_samples,data_idx=1,jacobian=np.sin,filepath=f"{plots_dir}/m_function_theta.png",xlabel=r"$\theta$")
# chi_2s[2] = osde(M_function, np.array(mdata), m_parameters, np.linspace(-np.pi, np.pi), axis = axs[1,0], samples=m_mc_samples,data_idx=2,filepath=f"{plots_dir}/m_function_phistar.png", xlabel=r"$\phi^*$")
# chi_2s[3] = osde(M_function, np.array(mdata), m_parameters, np.linspace(0, np.pi), axis = axs[1,1], samples = m_mc_samples,data_idx=3,jacobian=np.sin,filepath=f"{plots_dir}/m_function_thetastar.png", xlabel=r"$\theta^*$")

# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# fig.suptitle(r"OSDE Fitting for ttbar rapidity and $\cos(\theta_{prod})$", fontsize=16)

# hermite_params = list((n,) for n in range(13))
# chi_2s[4] = osde(hermite_func, np.column_stack([branches["rapidity"],]), hermite_params, np.linspace(-3.1,3.1, 500), axis = axs[0], filepath=f"{plots_dir}/rapidity.png", xlabel="rapidity")

# legendre_params = list((n,) for n in range(13))
# chi_2s[5] = osde(legendre_func, np.column_stack([branches["cosprodangle"],]), legendre_params, np.linspace(-1, 1, 500), axis = axs[1], filepath=f"{plots_dir}/cosprodangle.png",xlabel=r"$\cos(\theta_{prod})$")

# plt.tight_layout()
# plt.savefig(f"{plots_dir}/rapidity_cosprodangle.png")
# plt.close()
