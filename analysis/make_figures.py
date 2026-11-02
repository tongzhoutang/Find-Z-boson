"""生成报告和展示用的插图。所有数据都来自第一个文件的全部事件。

运行：python analysis/make_figures.py <ROOT 文件> <输出目录>
"""
import json
import sys
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle
from scipy.special import voigt_profile

sys.path.insert(0, str(Path(__file__).parent))
import zll_analysis as za

INK = "#1c1b19"
MUTED = "#6b6860"
FAINT = "#c9c5b9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
M_Z = 91.1880  # PDG
EVENT_FOR_DISPLAY = 1569374
EVENT_WITH_ELECTRON = 2757  # 文件中的序号

font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({
    "font.family": "Microsoft YaHei", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "font.size": 12, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "xtick.direction": "out", "ytick.direction": "out",
})
MASS_LABEL = r"$m_{\mu\mu}$ [GeV]"


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=260, bbox_inches="tight", transparent=True)
    plt.close(fig)


def load(path):
    tree = uproot.open(f"{path}:analysis")
    names = ["lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge"]
    raw = tree.arrays(names + ["lep_type", "met", "eventNumber"], library="ak")
    events = za.select_muon_pair(raw, names)
    events["mll"] = za.invariant_mass(events["lep_pt"], events["lep_eta"], events["lep_phi"], events["lep_e"])
    events["met"] = ak.to_numpy(raw.met)
    events["eventNumber"] = ak.to_numpy(raw.eventNumber)
    return raw, events


def errorbars(ax, counts, color=INK, ms=4.5, label=None):
    ax.errorbar(za.bin_centres(), counts, yerr=np.sqrt(counts), fmt="o", ms=ms, color=color,
                elinewidth=1, capsize=0, label=label)


def event_display(events, out, numbers):
    i = int(np.nonzero(events["eventNumber"] == EVENT_FOR_DISPLAY)[0][0])
    pt, eta, phi, e, q = [events[k][i].astype(float) for k in
                          ["lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge"]]
    numbers["display_event"] = {
        "eventNumber": EVENT_FOR_DISPLAY, "pt": pt.tolist(), "eta": eta.tolist(), "phi": phi.tolist(),
        "e": e.tolist(), "charge": q.tolist(), "mass": float(events["mll"][i]), "met": float(events["met"][i]),
    }
    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    layers = [(10.0, "#e4e0d4", "μ 子谱仪"), (6.2, "#d6d1c1", "量能器"), (3.3, "#c7c1ae", "径迹室")]
    for radius, color, _ in layers:
        ax.add_patch(Circle((0, 0), radius, facecolor=color, edgecolor="none", alpha=0.8))
    for radius in [10.0, 6.2, 3.3]:
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=MUTED, lw=0.7))
    ax.plot(0, 0, "o", ms=7, color=INK, zorder=5)
    for k, color in enumerate([BLUE, ORANGE]):
        x, y = 10.9 * np.cos(phi[k]), 10.9 * np.sin(phi[k])
        ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6, shrinkA=0, shrinkB=0))
        sign = "+" if q[k] > 0 else "−"
        ax.text(x + 3.3, y + np.sign(y) * 0.8, rf"$\mu^{{{sign}}}$" + f"\n{pt[k]:.1f} GeV", color=color,
                fontsize=13, ha="center", va="center", fontweight="bold")
    ax.set_xlim(-11.5, 13.5)
    ax.set_ylim(-12.5, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, out, "event_display")


def emergence(events, out):
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.6))
    for ax, n in zip(axes, [100, 1000, 10000, len(events["mll"])]):
        counts = za.histogram(events["mll"][:n])
        ax.bar(za.bin_centres(), counts, width=0.92, color=INK if n < len(events["mll"]) else BLUE)
        ax.set_title(f"前 {n:,} 个事件", fontsize=12.5, loc="left", color=INK)
        ax.set_xlabel(MASS_LABEL, fontsize=11)
        ax.set_xlim(70, 110)
    axes[0].set_ylabel("每 GeV 事件数", fontsize=11)
    fig.tight_layout(w_pad=1.6)
    save(fig, out, "emergence")


def spectrum(events, out):
    edges = np.geomspace(0.25, 200, 180)
    counts, _ = np.histogram(events["mll"], bins=edges)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.stairs(counts, edges, color=INK, lw=1.2, fill=True, facecolor="#d9d5c8", edgecolor=INK)
    for mass, text, size, color in [(3.097, r"$J/\psi$", 16, BLUE), (9.46, r"$\Upsilon$", 16, BLUE), (91.19, "Z", 16, BLUE),
                                    (0.78, r"$\rho,\omega$", 11, MUTED), (1.02, r"$\phi$", 11, MUTED),
                                    (3.686, r"$\psi(2S)$", 11, MUTED)]:
        i = np.searchsorted(edges, mass) - 1
        peak = counts[max(i - 1, 0):i + 2].max()
        ax.text(mass, peak * 1.2, text, ha="left" if mass == 3.686 else "center", fontsize=size, color=color, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 200)
    ax.set_ylim(40, counts.max() * 2.1)
    ax.set_xlabel(MASS_LABEL)
    ax.set_ylabel("每个 bin 的事件数")
    save(fig, out, "spectrum")


def filmstrip(events, out, numbers):
    cuts = [0, 25, 30, 40, 45, 50]
    baseline = None
    fig, axes = plt.subplots(1, 6, figsize=(9.0, 2.6), sharey=True)
    table = {}
    for ax, cut in zip(axes, cuts):
        counts = za.histogram(events["mll"][za.pass_pt_cut(events["lep_pt"], cut)])
        baseline = counts.sum() if baseline is None else baseline
        table[cut] = int(counts.sum())
        ax.bar(za.bin_centres(), counts, width=0.92, color=ORANGE if cut >= 45 else INK)
        ax.set_xlim(70, 110)
        ax.set_ylim(0, 7000)
        ax.set_xticks([70, 90, 110])
        ax.set_title("不加阈值" if cut == 0 else rf"$p_T>{cut}$ GeV", fontsize=12.5, loc="left")
        ax.text(0.97, 0.93, f"{counts.sum() / baseline:.1%}", transform=ax.transAxes, ha="right",
                fontsize=14, fontweight="bold", color=ORANGE if cut >= 45 else INK)
        ax.set_xlabel(MASS_LABEL, fontsize=10.5)
    axes[0].set_ylabel("每 GeV 事件数", fontsize=11)
    fig.tight_layout(w_pad=0.7)
    save(fig, out, "filmstrip")
    numbers["filmstrip_window_events"] = table


def cliff(events, out, numbers):
    selected = (events["mll"] > 81) & (events["mll"] < 101) & (events["lep_charge"].prod(axis=1) < 0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.8, 3.05), gridspec_kw={"wspace": 0.3})
    pt = events["lep_pt"][selected].ravel()
    left.hist(pt, bins=np.arange(10, 90, 1), color="#d9d5c8", edgecolor=INK, linewidth=0.5)
    left.axvline(M_Z / 2, color=ORANGE, lw=2)
    left.text(M_Z / 2 + 1.5, left.get_ylim()[1] * 0.93, r"$m_Z/2$" + f" = {M_Z / 2:.1f} GeV", color=ORANGE, fontsize=12.5,
              fontweight="bold", va="top")
    left.set_xlabel(r"μ 子 $p_T$ [GeV]（81–101 GeV 的异号事件）")
    left.set_ylabel("每 GeV 的 μ 子数")

    cuts = np.arange(10, 71, 1)
    peak_region = (events["mll"] > 85) & (events["mll"] < 97)
    survivors = np.array([(za.pass_pt_cut(events["lep_pt"], c) & peak_region).sum() for c in cuts])
    right.plot(cuts, survivors / survivors[0], color=INK, lw=2.4)
    right.axvline(M_Z / 2, color=ORANGE, lw=2)
    right.set_xlabel(r"阈值 $p_T^{\mathrm{cut}}$ [GeV]（两个 μ 子都要满足）")
    right.set_ylabel("85–97 GeV 内的保留比例")
    right.set_ylim(0, 1.04)
    right.set_xlim(10, 70)
    save(fig, out, "cliff")
    peak_bin = int(np.arange(10, 90, 1)[np.argmax(np.histogram(pt, bins=np.arange(10, 90, 1))[0])])
    numbers["cliff"] = {"muon_pt_mode_bin_low_edge": peak_bin,
                        "kept_at": {int(c): float(s / survivors[0]) for c, s in zip(cuts, survivors) if c in (25, 30, 40, 44, 45, 46, 50, 60)}}


def drift_and_trap(events, out, numbers):
    cuts = [10, 15, 20, 25, 30, 35, 38, 40, 42, 44, 45]
    fits = []
    for cut in cuts:
        fit = za.fit_peak(za.histogram(events["mll"][za.pass_pt_cut(events["lep_pt"], cut)]))
        if fit is not None:
            fits.append((cut, fit))
    numbers["drift"] = [{"cut": c, "mass": f["mass"], "err": f["mass_err"], "sigma": f["sigma"],
                         "chi2": f["chi2"], "ndf": f["ndf"]} for c, f in fits]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.8, 3.05), gridspec_kw={"wspace": 0.3})
    x = [c for c, _ in fits]
    y = [f["mass"] for _, f in fits]
    err = [f["mass_err"] for _, f in fits]
    left.errorbar(x, y, yerr=err, fmt="o-", ms=5, color=INK, lw=1.2, elinewidth=1)
    left.axhline(M_Z, color=BLUE, lw=1.6, ls="--")
    left.text(10, M_Z + 0.08, "PDG $m_Z$", color=BLUE, fontsize=12)
    at_40 = next(f for c, f in fits if c == 40)
    left.plot(40, at_40["mass"], "o", ms=11, mfc="none", mec=ORANGE, mew=2.2)
    left.annotate(rf"40 GeV：{at_40['mass']:.2f} ± {at_40['mass_err']:.2f}" + "\n恰好“对上”，但不代表测得更准", xy=(40, at_40["mass"]),
                  xytext=(11, 92.0), color=ORANGE, fontsize=10.5, fontweight="bold",
                  arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2, shrinkB=7))
    left.set_xlabel(r"阈值 $p_T^{\mathrm{cut}}$ [GeV]")
    left.set_ylabel("拟合峰位 [GeV]")

    counts = za.histogram(events["mll"][za.pass_pt_cut(events["lep_pt"], 45)])
    fit = za.fit_peak(counts)
    errorbars(right, counts)
    fine = np.linspace(*za.FIT_RANGE, 300)
    right.plot(fine, za.model(fine, *fit["popt"]), color=ORANGE, lw=2.4)
    right.axvline(M_Z, color=BLUE, lw=1.6, ls="--")
    right.text(0.65, 0.97, r"$p_T>45$ GeV" + "\n" + rf"$\chi^2/\mathrm{{ndf}}={fit['chi2']:.1f}/{fit['ndf']}$" + "\n"
               + rf"$m={fit['mass']:.2f}\pm{fit['mass_err']:.2f}$ GeV", transform=right.transAxes, va="top", fontsize=9)
    right.set_xlabel(MASS_LABEL)
    right.set_ylabel("每 GeV 事件数")
    right.set_xlim(70, 110)
    save(fig, out, "drift_trap")


def blur(events, out, numbers):
    counts = za.histogram(events["mll"])
    fit = za.fit_peak(counts)
    centres = za.bin_centres()
    fine = np.linspace(70, 109, 600)
    sigma = fit["sigma"]
    voigt = voigt_profile(fine - fit["mass"], sigma, za.GAMMA_Z / 2)
    half = voigt.max() / 2
    above = fine[voigt >= half]
    numbers["default_fit"] = {k: fit[k] for k in ["mass", "mass_err", "sigma", "sigma_err", "chi2", "ndf"]}
    numbers["default_fit"]["voigt_fwhm"] = float(above[-1] - above[0])
    norm, _, _, bkg, slope = fit["popt"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    core = fine[(fine >= 85) & (fine <= 97)]
    ideal = norm * voigt_profile(core - fit["mass"], 1e-4, za.GAMMA_Z / 2) + bkg * np.exp(-slope * (core - 91))
    ax.plot(core, ideal, color=BLUE, lw=2, label=rf"去掉分辨率（只剩 $\Gamma_Z$）")
    errorbars(ax, counts, label="真实数据")
    ax.plot(core, za.model(core, *fit["popt"]), color=ORANGE, lw=2.2, label=rf"拟合（$\sigma$ = {sigma:.2f} GeV）")
    ax.set_xlabel(MASS_LABEL)
    ax.set_ylabel("每 GeV 事件数")
    ax.set_xlim(78, 104)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    save(fig, out, "blur")
    # 让标题里的 FWHM 可追溯
    numbers["default_counts"] = {"total_window": int(counts.sum()), "peak_bin_centre": float(centres[np.argmax(counts)]),
                                 "peak_count": int(counts.max())}


def charge(events, out, numbers):
    product = events["lep_charge"].prod(axis=1)
    opposite = za.histogram(events["mll"][product < 0])
    same = za.histogram(events["mll"][product > 0])
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    errorbars(ax, opposite, label=f"电荷相反  N = {opposite.sum():,}")
    errorbars(ax, same, color=ORANGE, label=f"电荷相同  N = {same.sum():,}")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 3e4)
    ax.set_xlabel(MASS_LABEL)
    ax.set_ylabel("每 GeV 事件数")
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.3)
    save(fig, out, "charge")
    numbers["charge"] = {"opposite": int(opposite.sum()), "same": int(same.sum())}


def waffle(out):
    shares = [("强子", 69.911, "#d9d5c8"), ("中微子（不可见）", 20.000, ORANGE),
              (r"$e^+e^-$", 3.3632, "#8f8a7c"), (r"$\mu^+\mu^-$", 3.3662, BLUE), (r"$\tau^+\tau^-$", 3.3696, "#514e47")]
    fig, ax = plt.subplots(figsize=(5.7, 5.0))
    position = 0.0
    cells = [(col, row) for row in range(9, -1, -1) for col in range(10)]
    for _, share, color in shares:
        remaining = share
        while remaining > 1e-9:
            index = int(position)
            if index >= 100:
                break
            used = position - index
            take = min(1 - used, remaining)
            col, row = cells[index]
            ax.add_patch(Rectangle((col + 0.06 + used * 0.88, row + 0.06), take * 0.88, 0.88, facecolor=color,
                                   edgecolor="none"))
            position += take
            remaining -= take
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, out, "waffle")


def met(events, out, numbers):
    selected = (events["mll"] > 81) & (events["mll"] < 101) & (events["lep_charge"].prod(axis=1) < 0)
    values = events["met"][selected]
    fig, ax = plt.subplots(figsize=(4.3, 2.95))
    ax.hist(values, bins=np.arange(0, 305, 5), color="#d9d5c8", edgecolor=INK, linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"丢失横能量 $E_T^{\mathrm{miss}}$ [GeV]")
    ax.set_ylabel("每 5 GeV 事件数")
    ax.set_xlim(0, 300)
    ax.axvline(100, color=ORANGE, lw=2)
    tail = int((values > 100).sum())
    ax.text(105, ax.get_ylim()[1] * 0.45, f"> 100 GeV：{tail} 个事件\n占 {tail / len(values):.2%}", color=ORANGE,
            fontsize=13, fontweight="bold", va="top")
    save(fig, out, "met")
    numbers["met"] = {"events": int(len(values)), "median": float(np.median(values)), "above_100": tail,
                      "above_150": int((values > 150).sum())}


def flavour_numbers(raw, events, numbers):
    pairs = ak.to_numpy(raw.lep_type[:, :2])
    flavours, freq = np.unique(pairs, axis=0, return_counts=True)
    numbers["first_two"] = {f"{a},{b}": int(n) for (a, b), n in zip(flavours.tolist(), freq.tolist())}
    i = EVENT_WITH_ELECTRON
    first_two = {k: ak.to_numpy(raw[k][i:i + 1, :2]) for k in ["lep_pt", "lep_eta", "lep_phi", "lep_e"]}
    numbers["electron_example"] = {
        "types": raw.lep_type[i].tolist(), "pt": raw.lep_pt[i].tolist(),
        "notebook_mass": float(za.invariant_mass(first_two["lep_pt"], first_two["lep_eta"],
                                                 first_two["lep_phi"], first_two["lep_e"])[0]),
        "muon_mass": float(events["mll"][i]),
    }
    numbers["muon_counts"] = {int(a): int(b) for a, b in zip(*np.unique(ak.to_numpy(ak.sum(raw.lep_type == 13, axis=1)),
                                                                      return_counts=True))}


def main():
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    raw, events = load(sys.argv[1])
    numbers = {"events": len(events["mll"])}
    event_display(events, out, numbers)
    emergence(events, out)
    spectrum(events, out)
    filmstrip(events, out, numbers)
    cliff(events, out, numbers)
    drift_and_trap(events, out, numbers)
    blur(events, out, numbers)
    charge(events, out, numbers)
    waffle(out)
    met(events, out, numbers)
    flavour_numbers(raw, events, numbers)
    with open(out / "numbers.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
