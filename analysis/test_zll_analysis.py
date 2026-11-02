"""检查 zll_analysis 的计算逻辑。这里全部是人造数据，结果不能写进报告。

运行：python analysis/test_zll_analysis.py [输出图的目录]
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import zll_analysis as za


def toy_events(n=400_000, mass_res=0.02, seed=1):
    """Z -> mu mu 玩具样本：固有宽度加逐事件的质量分辨率，再混入指数本底。"""
    rng = np.random.default_rng(seed)
    m = za.GAMMA_Z / 2 * rng.standard_cauchy(n) + 91.19 + rng.normal(0, 91.19 * mass_res, n)
    background = 60 + rng.exponential(25, n // 20)
    m = np.concatenate([m, background])
    m = m[(m > 50) & (m < 150)]
    n = len(m)

    # 在母粒子静止系里各向同性衰变，再沿 z 方向和横向推动
    cos_t = rng.uniform(-1, 1, n)
    sin_t = np.sqrt(1 - cos_t**2)
    phi = rng.uniform(0, 2 * np.pi, n)
    p = np.stack([m / 2 * sin_t * np.cos(phi), m / 2 * sin_t * np.sin(phi), m / 2 * cos_t], axis=1)
    pt_z = rng.gamma(1.3, 12, n)
    phi_z = rng.uniform(0, 2 * np.pi, n)
    y_z = rng.normal(0, 1.6, n)
    mt = np.hypot(m, pt_z)
    energy = mt * np.cosh(y_z)
    beta = np.stack([pt_z * np.cos(phi_z), pt_z * np.sin(phi_z), mt * np.sinh(y_z)], axis=1) / energy[:, None]
    gamma = energy / m
    b2 = np.sum(beta**2, axis=1)

    def boost(vec):
        along = np.sum(beta * vec, axis=1)
        return vec + beta * ((gamma - 1) / b2 * along + gamma * m / 2)[:, None]

    p1, p2 = boost(p), boost(-p)
    pt = np.stack([np.hypot(p1[:, 0], p1[:, 1]), np.hypot(p2[:, 0], p2[:, 1])], axis=1)
    eta = np.stack([np.arcsinh(p1[:, 2] / pt[:, 0]), np.arcsinh(p2[:, 2] / pt[:, 1])], axis=1)
    phi_l = np.stack([np.arctan2(p1[:, 1], p1[:, 0]), np.arctan2(p2[:, 1], p2[:, 0])], axis=1)
    e = pt * np.cosh(eta)
    keep = np.all(np.abs(eta) < 2.5, axis=1)
    charge = np.tile([1.0, -1.0], (n, 1))
    return {"lep_pt": pt[keep], "lep_eta": eta[keep], "lep_phi": phi_l[keep],
            "lep_e": e[keep], "lep_charge": charge[keep]}


def test_hand_calculation():
    # notebook cell-23 的手算题，单位换算成 GeV，答案应为 92.15 GeV
    px = np.array([[32.387, -18.772]])
    py = np.array([[7.047, -9.563]])
    pz = np.array([[144.152, -2.942]])
    e = np.array([[147.913, 21.272]])
    pt = np.hypot(px, py)
    mass = za.invariant_mass(pt, np.arcsinh(pz / pt), np.arctan2(py, px), e)
    assert abs(mass[0] - 92.150) < 0.001, mass


def test_cut_requires_both_leptons():
    pt = np.array([[50.0, 10.0], [50.0, 40.0], [20.0, 20.0]])
    assert za.pass_pt_cut(pt, 30).tolist() == [False, True, False]


def test_binning_matches_notebook():
    assert len(za.EDGES) - 1 == 39 and za.EDGES[0] == 70 and za.EDGES[-1] == 109


def test_efficiency_handles_empty_bins():
    eff, err = za.efficiency([5, 0, 10], [10, 0, 10])
    assert eff.tolist() == [0.5, 0.0, 1.0]
    assert abs(err[0] - np.sqrt(0.025)) < 1e-12 and err[1] == 0 and err[2] == 0


def test_fit_recovers_toy_truth():
    events = toy_events()
    mass = za.invariant_mass(events["lep_pt"], events["lep_eta"], events["lep_phi"], events["lep_e"])
    fit = za.fit_peak(za.histogram(mass))
    assert abs(fit["mass"] - 91.19) < 0.05, fit
    assert abs(fit["sigma"] - 91.19 * 0.02) < 0.05, fit


def test_selects_highest_pt_muons_not_first_two_leptons():
    import awkward as ak

    batch = ak.Array({
        "lep_type": [[11, 13, 13], [13, 13], [13, 11], [13, 13, 13]],
        "lep_pt": [[90.0, 20.0, 50.0], [30.0, 40.0], [60.0, 70.0], [10.0, 80.0, 45.0]],
    })
    selected = za.select_muon_pair(batch, ["lep_pt"])["lep_pt"]
    # 第三个事件只有一个 μ 子，应被去掉；电子不进入 μ 子对
    assert selected.tolist() == [[50.0, 20.0], [40.0, 30.0], [80.0, 45.0]], selected


def test_fit_refuses_too_few_events():
    assert za.fit_peak(np.full(39, 2)) is None


def test_full_run_writes_outputs(out):
    rows = za.run(toy_events(), {"files": [], "lepton_counts": {2: 0}}, za.DEFAULT_CUTS, out)
    kept = [r["kept"] for r in rows]
    assert kept[0] == 1 and all(a > b for a, b in zip(kept, kept[1:])), kept
    # 高阈值扭曲形状：cut=40 时高质量一侧的保留比例应高于低质量一侧
    eff40 = rows[3]["eff"]
    assert eff40[:10].mean() < eff40[-10:].mean(), eff40
    for name in ["cut_scan.csv", "fit_range_scan.csv", "cut_comparison.png", "default_fit.png",
                 "charge_comparison.png", "diagnostics.json"]:
        assert (Path(out) / name).exists(), name
    with open(Path(out) / "cut_scan.csv", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == len(za.DEFAULT_CUTS)
    return rows


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    test_hand_calculation()
    test_cut_requires_both_leptons()
    test_binning_matches_notebook()
    test_efficiency_handles_empty_bins()
    test_fit_recovers_toy_truth()
    test_selects_highest_pt_muons_not_first_two_leptons()
    test_fit_refuses_too_few_events()
    for r in test_full_run_writes_outputs(out):
        fit = r["fit"]
        print(r["cut"], r["n_window"], round(r["kept"], 3), r["peak_bin"],
              None if fit is None else (round(fit["mass"], 2), round(fit["sigma"], 2),
                                        round(fit["chi2"]), fit["ndf"]))
    print("全部检查通过，输出目录：", out)
