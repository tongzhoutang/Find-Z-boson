"""按 notebook cell-19、cell-22 的原始算法重算，并与 notebook 保存的图逐像素比较。

运行：python analysis/crosscheck_notebook.py <前一半事件所在的 ROOT 文件>
"""
import sys
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
import vector
from matplotlib.ticker import AutoMinorLocator

STORED_PLOT = Path(__file__).parent.parent / "images" / "default_mll.png"


def calc_mll(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    return (p4[:, 0] + p4[:, 1]).M


def notebook_masses(path):
    tree = uproot.open(f"{path}:analysis")
    masses = []
    for data in tree.iterate(["lep_pt", "lep_eta", "lep_phi", "lep_e"],
                             entry_stop=tree.num_entries * 0.5, library="ak"):
        masses.append(calc_mll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_e))
    return ak.to_numpy(ak.concatenate(masses))


def render(counts, centres, out):
    ax = plt.gca()
    ax.errorbar(x=centres, y=counts, yerr=np.sqrt(counts), fmt="ko", label="Data")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_xlabel("Di-lepton Mass [GeV]", fontsize=13, x=1, horizontalalignment="right")
    ax.set_ylabel("Events", fontsize=13, y=1, horizontalalignment="right")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(frameon=False)
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    masses = notebook_masses(sys.argv[1])
    edges = np.arange(70, 110, 1)
    centres = (edges[:-1] + edges[1:]) / 2
    counts = np.histogram(masses, bins=edges)[0]
    print("读取事件数:", len(masses))
    print("70-109 GeV 窗口内事件数:", int(counts.sum()))
    print("各 bin:", counts.tolist())

    out = Path(sys.argv[1]).with_name("crosscheck_render.png")
    render(counts, centres, out)
    stored = plt.imread(STORED_PLOT)[..., :3]
    rendered = plt.imread(out)[..., :3]
    out.unlink()
    if stored.shape != rendered.shape:
        print("图片尺寸不同:", stored.shape, rendered.shape)
        return 1
    differing = int((np.abs(stored - rendered).max(axis=2) > 0).sum())
    print("与 notebook 保存的图不同的像素数:", differing)
    return 0 if differing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
