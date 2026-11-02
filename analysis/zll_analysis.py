"""Find the Z 报告用的补充分析：pT 阈值扫描、效率、Voigt 拟合、电荷对比。

用法（读取 Colab 用过的同一个文件，fraction 与 notebook 默认一致）：
    python analysis/zll_analysis.py <文件路径或URL> [更多文件...] --fraction 0.5 --out results

输入的量和 notebook 一样以 GeV 为单位。计算部分只依赖 numpy/scipy，
只有 load_events 才需要 uproot 和 awkward。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

GAMMA_Z = 2.4955  # Z 的固有全宽 [GeV]（PDG 2024），拟合时固定，只让分辨率 sigma 浮动
EDGES = np.arange(70, 110, 1)  # 与 notebook cell-22 相同：70-109 GeV，39 个 bin
DEFAULT_CUTS = [0, 25, 30, 40, 45]
FIT_RANGE = (85.0, 97.0)  # 模型不含末态辐射的低质量拖尾，所以主要拟合峰的核心区
FIT_RANGE_VARIATIONS = [(87.0, 95.0), (85.0, 97.0), (80.0, 102.0), (75.0, 105.0)]

# 拟合只有在每个 bin 都有足够事件时才有意义，否则不报结果
MIN_FIT_EVENTS = 500


def select_muon_pair(batch, names):
    """选出至少有两个 μ 子的事件，每个事件保留 pT 最高的两个 μ 子。

    数据的 lep_* 数组里混有电子，也不按 pT 排序，所以不能直接取前两个元素。
    """
    import awkward as ak

    is_muon = batch.lep_type == 13
    order = ak.argsort(batch.lep_pt[is_muon], axis=1, ascending=False)
    has_pair = ak.sum(is_muon, axis=1) >= 2
    return {name: ak.to_numpy(batch[name][is_muon][order][has_pair][:, :2]) for name in names}


def add_count(table, key, n):
    table[key] = table.get(key, 0) + n


def load_events(paths, fraction=0.5):
    """读取 μ 子对的运动学量，返回 numpy 数组字典和读取信息。"""
    import awkward as ak
    import uproot

    kinematics = ["lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge"]
    chunks = []
    info = {"files": [], "lepton_counts": {}, "muon_counts": {}, "first_two_flavours": {},
            "notebook_first_two_window_events": 0, "branches": None}
    for path in paths:
        tree = uproot.open(f"{path}:analysis")
        info["branches"] = sorted(tree.keys())
        stop = int(tree.num_entries * fraction)
        info["files"].append({"path": str(path), "entries": tree.num_entries, "read": stop})
        for batch in tree.iterate(kinematics + ["lep_type"], entry_stop=stop, library="ak"):
            for name, values in [("lepton_counts", ak.num(batch.lep_pt)),
                                 ("muon_counts", ak.sum(batch.lep_type == 13, axis=1))]:
                counts, freq = np.unique(ak.to_numpy(values), return_counts=True)
                for count, n in zip(counts.tolist(), freq.tolist()):
                    add_count(info[name], count, n)

            # 按 notebook 的算法取前两个轻子，只用于和 Colab 的结果交叉检查
            first_two = {name: ak.to_numpy(batch[name][:, :2]) for name in kinematics + ["lep_type"]}
            flavours, freq = np.unique(first_two["lep_type"], axis=0, return_counts=True)
            for pair, n in zip(flavours.tolist(), freq.tolist()):
                add_count(info["first_two_flavours"], f"{pair[0]},{pair[1]}", n)
            info["notebook_first_two_window_events"] += int(histogram(invariant_mass(
                first_two["lep_pt"], first_two["lep_eta"], first_two["lep_phi"], first_two["lep_e"])).sum())

            chunks.append(select_muon_pair(batch, kinematics))

    events = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    return events, info


def invariant_mass(pt, eta, phi, e):
    """由 shape (N, 2) 的数组计算双轻子不变质量。"""
    px = np.sum(pt * np.cos(phi), axis=1)
    py = np.sum(pt * np.sin(phi), axis=1)
    pz = np.sum(pt * np.sinh(eta), axis=1)
    energy = np.sum(e, axis=1)
    return np.sqrt(np.clip(energy**2 - px**2 - py**2 - pz**2, 0, None))


def pass_pt_cut(pt, cut):
    """要求事件里的两个轻子 pT 都大于 cut。"""
    return np.all(pt > cut, axis=1)


def histogram(mass):
    counts, _ = np.histogram(mass, bins=EDGES)
    return counts


def bin_centres():
    return (EDGES[:-1] + EDGES[1:]) / 2


def efficiency(passed, total):
    """逐 bin 保留比例和二项误差：passed 是 total 的子集，两者不独立。"""
    passed = np.asarray(passed, dtype=float)
    total = np.asarray(total, dtype=float)
    eff = np.divide(passed, total, out=np.zeros_like(total), where=total > 0)
    err = np.sqrt(np.divide(eff * (1 - eff), total, out=np.zeros_like(total), where=total > 0))
    return eff, err


def model(m, norm, mass, sigma, bkg, slope):
    """每 GeV 的事件数：Voigt 信号（固有宽度固定）加指数本底。"""
    return norm * voigt_profile(m - mass, sigma, GAMMA_Z / 2) + bkg * np.exp(-slope * (m - 91.0))


def fit_peak(counts, fit_range=FIT_RANGE):
    """在 fit_range 内做分 bin 最小二乘拟合，事件太少或不收敛时返回 None。"""
    centres = bin_centres()
    in_range = (centres >= fit_range[0]) & (centres <= fit_range[1])
    x, y = centres[in_range], counts[in_range].astype(float)
    if y.sum() < MIN_FIT_EVENTS:
        return None
    err = np.sqrt(np.maximum(y, 1.0))
    start = [y.sum(), x[np.argmax(y)], 1.5, max(y.min(), 1.0), 0.03]
    bounds = ([0, 80, 0.05, 0, -0.5], [np.inf, 102, 15, np.inf, 0.5])
    try:
        popt, pcov = curve_fit(model, x, y, p0=start, sigma=err, absolute_sigma=True,
                               bounds=bounds, maxfev=20000)
    except RuntimeError:
        return None
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((y - model(x, *popt)) / err) ** 2))
    return {
        "mass": float(popt[1]), "mass_err": float(perr[1]),
        "sigma": float(popt[2]), "sigma_err": float(perr[2]),
        "chi2": chi2, "ndf": int(len(x) - len(popt)), "popt": popt.tolist(),
    }


def scan_cuts(events, cuts):
    """对每个阈值计算直方图、保留比例和拟合结果。第一个阈值是基准。"""
    pt = events["lep_pt"]
    mass = events["mll"]
    baseline = histogram(mass[pass_pt_cut(pt, cuts[0])])
    rows = []
    for cut in cuts:
        counts = histogram(mass[pass_pt_cut(pt, cut)])
        eff, eff_err = efficiency(counts, baseline)
        rows.append({
            "cut": cut, "counts": counts, "eff": eff, "eff_err": eff_err,
            "n_window": int(counts.sum()),
            "kept": counts.sum() / baseline.sum() if baseline.sum() else float("nan"),
            "peak_bin": float(bin_centres()[np.argmax(counts)]),
            "fit": fit_peak(counts),
        })
    return rows


def write_table(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["pt_cut_GeV", "events_70_109", "kept_fraction", "peak_bin_GeV",
                         "fit_mass_GeV", "fit_mass_err", "fit_sigma_GeV", "fit_sigma_err",
                         "chi2", "ndf"])
        for r in rows:
            fit = r["fit"]
            fit_cells = (["", "", "", "", "", ""] if fit is None else
                         [f"{fit['mass']:.3f}", f"{fit['mass_err']:.3f}",
                          f"{fit['sigma']:.3f}", f"{fit['sigma_err']:.3f}",
                          f"{fit['chi2']:.1f}", fit["ndf"]])
            writer.writerow([r["cut"], r["n_window"], f"{r['kept']:.4f}", r["peak_bin"], *fit_cells])


def write_fit_range_scan(counts, path):
    """改变拟合范围，看峰位和分辨率对范围有多敏感。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["range_low_GeV", "range_high_GeV", "fit_mass_GeV", "fit_mass_err",
                         "fit_sigma_GeV", "fit_sigma_err", "chi2", "ndf"])
        for low, high in FIT_RANGE_VARIATIONS:
            fit = fit_peak(counts, (low, high))
            if fit is not None:
                writer.writerow([low, high, f"{fit['mass']:.3f}", f"{fit['mass_err']:.3f}",
                                 f"{fit['sigma']:.3f}", f"{fit['sigma_err']:.3f}",
                                 f"{fit['chi2']:.1f}", fit["ndf"]])


def plot_scan(rows, path):
    import matplotlib.pyplot as plt

    # 阈值大小有顺序，所以同一色相由浅到深；标记形状不同，黑白打印也能区分
    colors = ["#111111", "#a9c7ee", "#6a9fe0", "#2a6fc4", "#123f7a"]
    markers = ["o", "s", "^", "D", "v"]
    x = bin_centres()
    fig, (ax, rax) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 6.2),
                                  gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06})
    for r, color, marker in zip(rows, colors, markers):
        label = "No extra cut" if r["cut"] == 0 else rf"$p_T > {r['cut']}$ GeV"
        ax.errorbar(x, r["counts"], yerr=np.sqrt(r["counts"]), fmt=marker, ms=4, color=color,
                    elinewidth=1, label=label)
        if r is not rows[0]:
            rax.errorbar(x, r["eff"], yerr=r["eff_err"], fmt=marker, ms=4, color=color, elinewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Events / GeV")
    ax.legend(frameon=False, fontsize=9)
    rax.set_ylabel("Kept fraction")
    rax.set_ylim(0, 1.05)
    rax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    for a in (ax, rax):
        a.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fit(row, path):
    import matplotlib.pyplot as plt

    fit = row["fit"]
    x = bin_centres()
    y = row["counts"]
    fine = np.linspace(*FIT_RANGE, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(x, y, yerr=np.sqrt(y), fmt="o", ms=4, color="#111111", elinewidth=1, label="Data")
    ax.plot(fine, model(fine, *fit["popt"]), color="#2a6fc4", lw=2,
            label="Voigt + exponential fit")
    note = (rf"$m = {fit['mass']:.2f} \pm {fit['mass_err']:.2f}$ GeV" "\n"
            rf"$\sigma = {fit['sigma']:.2f} \pm {fit['sigma_err']:.2f}$ GeV" "\n"
            rf"$\chi^2/\mathrm{{ndf}} = {fit['chi2']:.0f}/{fit['ndf']}$" "\n"
            r"(statistical only)")
    ax.text(0.03, 0.95, note, transform=ax.transAxes, va="top", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    ax.set_ylabel("Events / GeV")
    ax.legend(frameon=False, loc="upper right")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_charge(events, path):
    """同号与异号事件的质量分布：Z 中性，只有异号组才应该出现 Z 峰。"""
    import matplotlib.pyplot as plt

    product = events["lep_charge"][:, 0] * events["lep_charge"][:, 1]
    x = bin_centres()
    opposite = histogram(events["mll"][product < 0])
    same = histogram(events["mll"][product > 0])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(x, opposite, yerr=np.sqrt(opposite), fmt="o", ms=4, color="#111111",
                elinewidth=1, label=f"Opposite charge (N = {opposite.sum()})")
    ax.errorbar(x, same, yerr=np.sqrt(same), fmt="s", ms=4, color="#2a6fc4",
                elinewidth=1, label=f"Same charge (N = {same.sum()})")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    ax.set_ylabel("Events / GeV")
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return int(opposite.sum()), int(same.sum())


def run(events, info, cuts, out):
    """计算全部结果并写入 out 目录，返回阈值扫描结果。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    events = dict(events)
    events["mll"] = invariant_mass(events["lep_pt"], events["lep_eta"],
                                   events["lep_phi"], events["lep_e"])
    rows = scan_cuts(events, cuts)
    write_table(rows, out / "cut_scan.csv")
    write_fit_range_scan(rows[0]["counts"], out / "fit_range_scan.csv")
    plot_scan(rows, out / "cut_comparison.png")
    if rows[0]["fit"] is not None:
        plot_fit(rows[0], out / "default_fit.png")

    diagnostics = dict(info)
    diagnostics.update({
        "events_used": int(len(events["mll"])),
        "min_lepton_pt_GeV": float(events["lep_pt"].min()),
        "cuts_GeV": list(cuts),
    })
    if "lep_charge" in events:
        opposite, same = plot_charge(events, out / "charge_comparison.png")
        diagnostics["charge_counts_70_109"] = {"opposite": opposite, "same": same}
    with open(out / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="ROOT 文件路径或 URL")
    parser.add_argument("--fraction", type=float, default=0.5, help="每个文件读取的事件比例")
    parser.add_argument("--cuts", type=float, nargs="+", default=DEFAULT_CUTS,
                        help="pT 阈值 [GeV]，第一个作为基准")
    parser.add_argument("--out", default="results", help="输出目录")
    args = parser.parse_args()

    events, info = load_events(args.paths, args.fraction)
    rows = run(events, info, args.cuts, args.out)
    for r in rows:
        fit = r["fit"]
        fit_text = "拟合无结果" if fit is None else (
            f"m={fit['mass']:.2f}±{fit['mass_err']:.2f}  sigma={fit['sigma']:.2f}  "
            f"chi2/ndf={fit['chi2']:.0f}/{fit['ndf']}")
        print(f"cut={r['cut']:>4}  N={r['n_window']:>8}  kept={r['kept']:.3f}  "
              f"peak_bin={r['peak_bin']}  {fit_text}")


if __name__ == "__main__":
    main()
