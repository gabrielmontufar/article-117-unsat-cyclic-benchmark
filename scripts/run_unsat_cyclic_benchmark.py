from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\gjm31\OneDrive\Escritorio\117 avo art")
SUPP = ROOT / "supplementary_unsat_cyclic_benchmark"
DATA = SUPP / "data"
FIGS = SUPP / "figures"
SCRIPTS = SUPP / "scripts"
for folder in (DATA, FIGS, SCRIPTS):
    folder.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\timesbd.ttf") if bold else Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


@dataclass(frozen=True)
class Params:
    p0: float = 100.0
    pc0: float = 120.0
    e0: float = 0.72
    sr_res: float = 0.08
    sr_sat: float = 0.98
    alpha_dry: float = 0.030
    alpha_wet: float = 0.018
    n_dry: float = 1.70
    n_wet: float = 1.45
    k_suction: float = 0.52
    h_suction: float = 0.38
    beta_h: float = 0.20
    beta_c: float = 0.26
    g0: float = 42.0
    m_cc: float = 1.25
    gamma_y0: float = 0.006


def vg_saturation(suction: np.ndarray, alpha: float, n: float, p: Params) -> np.ndarray:
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha * np.maximum(suction, 0.0)) ** n) ** (-m)
    return p.sr_res + (p.sr_sat - p.sr_res) * se


def simulate_case(model: str, suction_amp: float, cyclic_amp: float, n_steps: int = 720, suction_mean: float = 90.0) -> pd.DataFrame:
    p = Params()
    t = np.linspace(0.0, 1.0, n_steps)
    cycles_h = 2.0
    cycles_s = 18.0
    suction = suction_mean + suction_amp * np.sin(2.0 * math.pi * cycles_h * t - math.pi / 4.0)
    suction = np.clip(suction, 3.0, None)

    drying = np.gradient(suction) >= 0.0
    sr_dry = vg_saturation(suction, p.alpha_dry, p.n_dry, p)
    sr_wet = vg_saturation(suction, p.alpha_wet, p.n_wet, p)

    if model == "constant_suction":
        sr = np.full_like(suction, vg_saturation(np.array([suction_mean]), p.alpha_dry, p.n_dry, p)[0])
        suction_eff = np.full_like(suction, suction_mean)
        dh = np.zeros_like(suction)
    elif model == "no_hysteresis":
        sr = vg_saturation(suction, p.alpha_dry, p.n_dry, p)
        suction_eff = suction
        dh = np.zeros_like(suction)
    elif model == "hysteresis_only":
        sr = np.where(drying, sr_dry, sr_wet)
        suction_eff = suction
        dh = np.zeros_like(suction)
    elif model == "full_hysteretic_damage":
        sr = np.where(drying, sr_dry, sr_wet)
        suction_eff = suction
        reversal = np.abs(np.diff(np.sign(np.gradient(suction)), prepend=0)) > 0
        dh = np.cumsum(reversal.astype(float)) / max(1, reversal.sum())
        dh = np.clip(p.beta_h * dh, 0.0, 0.35)
    else:
        raise ValueError(model)

    chi = np.clip((sr - p.sr_res) / (p.sr_sat - p.sr_res), 0.0, 1.0)
    p_eff = p.p0 + chi * suction_eff
    pc = p.pc0 * (1.0 + p.k_suction * np.log1p(suction_eff / 100.0))
    wetting_collapse = np.maximum(0.0, -np.gradient(suction_eff)) / (np.max(suction_eff) + 1e-9)

    gamma = 0.010 * np.sin(2.0 * math.pi * cycles_s * t)
    tau_elastic = (p.g0 * (p_eff / p.p0) ** 0.5) * gamma
    cyclic_ratio = cyclic_amp * np.sin(2.0 * math.pi * cycles_s * t)
    demand = np.abs(cyclic_ratio) * p.p0
    yield_strength = p.m_cc * np.sqrt(np.maximum(p_eff * pc, 1e-6)) * 0.095
    overstress = np.maximum(0.0, demand - yield_strength) / (yield_strength + 1e-9)
    ds = np.cumsum(overstress**1.4) / n_steps
    if model != "full_hysteretic_damage":
        ds[:] = 0.0
    ds = np.clip(p.beta_c * ds, 0.0, 0.45)

    stiffness = p.g0 * (p_eff / p.p0) ** 0.5 * (1.0 - dh) * (1.0 - ds)
    damage_amplifier = 1.0 + ((1.4 * dh + 18.0 * ds) if model == "full_hysteretic_damage" else 0.0)
    plastic_increment = ((overstress**1.3) * np.abs(np.gradient(gamma)) * damage_amplifier) + 0.006 * wetting_collapse
    if model == "full_hysteretic_damage":
        plastic_increment += 0.07 * dh * np.abs(cyclic_ratio) * np.abs(np.gradient(gamma))
    plastic_strain = np.cumsum(plastic_increment) * 0.18
    if model in ("constant_suction", "no_hysteresis"):
        plastic_strain *= 0.45
    tau = stiffness * gamma + np.sign(gamma) * demand * 0.0015

    return pd.DataFrame(
        {
            "model": model,
            "suction_amp": suction_amp,
            "cyclic_amp": cyclic_amp,
            "step": np.arange(n_steps),
            "time": t,
            "suction_kpa": suction_eff,
            "degree_saturation": sr,
            "bishop_chi": chi,
            "mean_effective_stress_kpa": p_eff,
            "preconsolidation_kpa": pc,
            "hydraulic_damage": dh,
            "seismic_damage": ds,
            "shear_strain": gamma,
            "shear_stress_kpa": tau,
            "secant_stiffness_mpa": stiffness,
            "plastic_strain_index": plastic_strain,
        }
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["model", "suction_amp", "cyclic_amp"]):
        rows.append(
            {
                "model": keys[0],
                "suction_amp": keys[1],
                "cyclic_amp": keys[2],
                "min_saturation": g["degree_saturation"].min(),
                "max_saturation": g["degree_saturation"].max(),
                "final_hydraulic_damage": g["hydraulic_damage"].iloc[-1],
                "final_seismic_damage": g["seismic_damage"].iloc[-1],
                "stiffness_loss_pct": 100.0
                * (1.0 - g["secant_stiffness_mpa"].iloc[-1] / g["secant_stiffness_mpa"].iloc[0]),
                "max_abs_stress_kpa": g["shear_stress_kpa"].abs().max(),
                "final_plastic_strain_index": g["plastic_strain_index"].iloc[-1],
            }
        )
    out = pd.DataFrame(rows)
    base = out[out["model"] == "constant_suction"].set_index(["suction_amp", "cyclic_amp"])
    diffs = []
    for _, r in out.iterrows():
        b = base.loc[(r["suction_amp"], r["cyclic_amp"])]
        d = r.to_dict()
        d["plastic_strain_delta_vs_constant"] = (
            r["final_plastic_strain_index"] - b["final_plastic_strain_index"]
        )
        d["plastic_strain_ratio_vs_constant"] = (
            r["final_plastic_strain_index"] / b["final_plastic_strain_index"]
            if b["final_plastic_strain_index"] > 1e-5
            else np.nan
        )
        d["stiffness_loss_delta_pct"] = r["stiffness_loss_pct"] - b["stiffness_loss_pct"]
        diffs.append(d)
    return pd.DataFrame(diffs)


def draw_line_plot(series: list[tuple[str, np.ndarray, np.ndarray]], title: str, xlabel: str, ylabel: str, out: Path):
    w, h = 1600, 1000
    left, right, top, bottom = 150, 80, 130, 150
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = font(34, bold=True)
    label_font = font(26)
    tick_font = font(22)
    legend_font = font(24)
    colors = ["#1b6ca8", "#2a9d8f", "#e76f51", "#6a4c93", "#d62828"]
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if abs(ymax - ymin) < 1e-12:
        ymax += 1.0
    plot = [left, top, w - right, h - bottom]
    d.rectangle(plot, outline="#333333", width=3)
    d.text((left, 40), title, fill="#111111", font=title_font)
    d.text((w // 2 - 140, h - 75), xlabel, fill="#111111", font=label_font)
    d.text((25, h // 2 - 20), ylabel, fill="#111111", font=label_font)
    for i in range(6):
        x = left + i * (w - left - right) / 5
        y = h - bottom - i * (h - top - bottom) / 5
        d.line([x, top, x, h - bottom], fill="#eeeeee", width=2)
        d.line([left, y, w - right, y], fill="#eeeeee", width=2)
        xv = xmin + i * (xmax - xmin) / 5
        yv = ymin + i * (ymax - ymin) / 5
        d.text((x - 28, h - bottom + 16), f"{xv:.2g}", fill="#111111", font=tick_font)
        d.text((left - 105, y - 12), f"{yv:.3g}", fill="#111111", font=tick_font)
    for idx, (label, xvals, yvals) in enumerate(series):
        pts = []
        for x, y in zip(xvals, yvals):
            px = left + (x - xmin) / (xmax - xmin) * (w - left - right)
            py = h - bottom - (y - ymin) / (ymax - ymin) * (h - top - bottom)
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=colors[idx % len(colors)], width=5)
        y_leg = top + 34 * idx + 20
        d.line([w - 520, y_leg, w - 470, y_leg], fill=colors[idx % len(colors)], width=6)
        d.text((w - 455, y_leg - 14), label, fill="#111111", font=legend_font)
    img.save(out)


def draw_heat_table(summary: pd.DataFrame, out: Path):
    pivot = summary[summary["model"] == "full_hysteretic_damage"].pivot(
        index="suction_amp", columns="cyclic_amp", values="final_plastic_strain_index"
    )
    w, h = 1200, 850
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = font(34, bold=True)
    label_font = font(26)
    cell_font = font(24)
    d.text((70, 45), "Full model final plastic strain index", fill="#111111", font=title_font)
    left, top = 180, 160
    cw, ch = 170, 120
    vals = pivot.values
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    for i, s_amp in enumerate(pivot.index):
        d.text((60, top + i * ch + 45), f"{s_amp:.0f} kPa", fill="#111111", font=label_font)
        for j, c_amp in enumerate(pivot.columns):
            val = float(pivot.loc[s_amp, c_amp])
            r = int(245 - 120 * (val - vmin) / max(vmax - vmin, 1e-9))
            g = int(245 - 190 * (val - vmin) / max(vmax - vmin, 1e-9))
            b = int(255 - 220 * (val - vmin) / max(vmax - vmin, 1e-9))
            x0, y0 = left + j * cw, top + i * ch
            d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(r, g, b), outline="#333333", width=2)
            d.text((x0 + 45, y0 + 45), f"{val:.4f}", fill="#111111", font=cell_font)
    for j, c_amp in enumerate(pivot.columns):
        d.text((left + j * cw + 45, top - 45), f"{c_amp:.2f}g", fill="#111111", font=label_font)
    d.text((left + 190, h - 75), "cyclic stress ratio amplitude", fill="#111111", font=label_font)
    d.text((35, top + 210), "suction amplitude", fill="#111111", font=label_font)
    img.save(out)


def draw_metric_heat_table(summary: pd.DataFrame, value_col: str, title: str, value_fmt: str, out: Path):
    pivot = summary[summary["model"] == "full_hysteretic_damage"].pivot(
        index="suction_amp", columns="cyclic_amp", values=value_col
    )
    w, h = 1200, 850
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = font(34, bold=True)
    label_font = font(26)
    cell_font = font(24)
    d.text((70, 45), title, fill="#111111", font=title_font)
    left, top = 180, 160
    cw, ch = 170, 120
    vals = pivot.values
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    for i, s_amp in enumerate(pivot.index):
        d.text((60, top + i * ch + 45), f"{s_amp:.0f} kPa", fill="#111111", font=label_font)
        for j, c_amp in enumerate(pivot.columns):
            val = float(pivot.loc[s_amp, c_amp])
            scale = (val - vmin) / max(vmax - vmin, 1e-9)
            r = int(232 - 85 * scale)
            g = int(244 - 150 * scale)
            b = int(246 - 160 * scale)
            x0, y0 = left + j * cw, top + i * ch
            d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(r, g, b), outline="#333333", width=2)
            d.text((x0 + 45, y0 + 45), value_fmt.format(val), fill="#111111", font=cell_font)
    for j, c_amp in enumerate(pivot.columns):
        d.text((left + j * cw + 45, top - 45), f"{c_amp:.2f}g", fill="#111111", font=label_font)
    d.text((left + 190, h - 75), "cyclic stress ratio amplitude", fill="#111111", font=label_font)
    d.text((35, top + 210), "suction amplitude", fill="#111111", font=label_font)
    img.save(out)


def write_external_validation():
    rows = [
        {
            "source": "Nishimura (2015)",
            "external_observation": "Drying-wetting suction changes produced strong SWCC hysteresis under isotropic compression; apparent saturated soil after suction changes showed liquefaction-like cyclic response.",
            "model_expectation": "The model must retain hydraulic path memory and allow cyclic degradation after wetting-drying history.",
            "comparison": "consistent",
            "evidence_type": "qualitative trend from open conference paper",
            "source_url": "https://www.issmge.org/uploads/publications/59/60/422.00_Nishimura.pdf",
        },
        {
            "source": "Howard (2021)",
            "external_observation": "Unsaturated cyclic triaxial tests on silty sand showed that wetting-path specimens had higher cyclic resistance than drying-path specimens, and higher initial matric suction increased cyclic resistance.",
            "model_expectation": "The model must distinguish wetting and drying branches and include suction-dependent resistance.",
            "comparison": "consistent",
            "evidence_type": "qualitative trend from open doctoral dissertation record",
            "source_url": "https://scholarcommons.sc.edu/etd/6720",
        },
        {
            "source": "Suprunenko (2015)",
            "external_observation": "Suction-controlled cyclic triaxial tests on sand reported dynamic shear modulus dependence on degree of saturation, with higher modulus at mid-range saturation.",
            "model_expectation": "The model must allow saturation-dependent stiffness through Bishop chi and effective stress.",
            "comparison": "consistent with mechanism; not calibrated to the sand data",
            "evidence_type": "qualitative trend from open master's thesis record",
            "source_url": "https://scholars.unh.edu/thesis/1036",
        },
        {
            "source": "Dai and Zhou (2026)",
            "external_observation": "Suction-controlled cyclic tests on unsaturated loess showed permanent vertical strain and resilient modulus vary systematically with suction, compaction water content, density and CSR.",
            "model_expectation": "The model must couple suction, cyclic demand and plastic-strain accumulation rather than treating suction as a constant parameter.",
            "comparison": "consistent",
            "evidence_type": "qualitative trend and semi-empirical equation structure from open access article",
            "source_url": "https://doi.org/10.1007/s11440-026-02983-x",
        },
    ]
    validation = pd.DataFrame(rows)
    validation.to_csv(DATA / "external_validation_trend_checks.csv", index=False)
    draw_validation_summary(validation, FIGS / "fig07_external_validation_trend_checks.png")
    return validation


def draw_validation_summary(validation: pd.DataFrame, out: Path):
    w, h = 1400, 760
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = font(34, bold=True)
    label_font = font(24)
    small_font = font(21)
    d.text((70, 45), "External trend validation from independent cyclic unsaturated-soil studies", fill="#111111", font=title_font)
    left, top = 95, 140
    row_h = 125
    colors = ["#d7ecf5", "#e6f2d6", "#fff1c9", "#eadff2"]
    for idx, row in validation.iterrows():
        y0 = top + idx * row_h
        d.rectangle([left, y0, w - 95, y0 + row_h - 15], fill=colors[idx % len(colors)], outline="#555555", width=2)
        d.text((left + 25, y0 + 18), row["source"], fill="#111111", font=label_font)
        d.text((left + 310, y0 + 18), row["comparison"], fill="#111111", font=label_font)
        obs = row["external_observation"][:125] + ("..." if len(row["external_observation"]) > 125 else "")
        d.text((left + 25, y0 + 62), obs, fill="#111111", font=small_font)
    d.text((left + 25, h - 80), "Purpose: independent trend check, not numerical calibration to proprietary or non-tabulated data.", fill="#111111", font=small_font)
    img.save(out)


def timestep_convergence() -> pd.DataFrame:
    rows = []
    reference = None
    for steps in [360, 720, 1440, 2880]:
        g = simulate_case("full_hysteretic_damage", 100.0, 0.20, n_steps=steps)
        row = {
            "n_steps": steps,
            "normalized_dt": 1.0 / (steps - 1),
            "final_plastic_strain_index": g["plastic_strain_index"].iloc[-1],
            "stiffness_loss_pct": 100.0
            * (1.0 - g["secant_stiffness_mpa"].iloc[-1] / g["secant_stiffness_mpa"].iloc[0]),
            "final_hydraulic_damage": g["hydraulic_damage"].iloc[-1],
            "final_seismic_damage": g["seismic_damage"].iloc[-1],
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    reference = out.iloc[-1]
    out["plastic_error_vs_finest_pct"] = 100.0 * (
        out["final_plastic_strain_index"] - reference["final_plastic_strain_index"]
    ).abs() / reference["final_plastic_strain_index"]
    out["stiffness_loss_error_vs_finest_pct"] = 100.0 * (
        out["stiffness_loss_pct"] - reference["stiffness_loss_pct"]
    ).abs() / max(abs(reference["stiffness_loss_pct"]), 1e-12)
    out.to_csv(DATA / "timestep_convergence.csv", index=False)
    return out


def write_parameter_table():
    rows = [
        ("p0", "reference mean stress", "kPa", Params().p0),
        ("pc0", "initial preconsolidation pressure", "kPa", Params().pc0),
        ("e0", "initial void ratio", "-", Params().e0),
        ("sr_res", "residual degree of saturation", "-", Params().sr_res),
        ("sr_sat", "saturated degree of saturation", "-", Params().sr_sat),
        ("alpha_dry", "drying retention parameter", "1/kPa", Params().alpha_dry),
        ("alpha_wet", "wetting retention parameter", "1/kPa", Params().alpha_wet),
        ("n_dry", "drying van Genuchten exponent", "-", Params().n_dry),
        ("n_wet", "wetting van Genuchten exponent", "-", Params().n_wet),
        ("k_suction", "suction hardening coefficient", "-", Params().k_suction),
        ("h_suction", "wetting-collapse coefficient", "-", Params().h_suction),
        ("beta_h", "maximum hydraulic damage factor", "-", Params().beta_h),
        ("beta_c", "maximum cyclic damage factor", "-", Params().beta_c),
        ("g0", "reference shear stiffness", "MPa", Params().g0),
        ("m_cc", "cyclic threshold multiplier", "-", Params().m_cc),
        ("gamma_y0", "reference yield shear strain", "-", Params().gamma_y0),
    ]
    pd.DataFrame(rows, columns=["parameter", "description", "unit", "value"]).to_csv(
        DATA / "model_parameters.csv", index=False
    )


def partial_quantitative_validation() -> pd.DataFrame:
    rows = []
    for suction in [0.0, 10.0, 30.0]:
        g = simulate_case("full_hysteretic_damage", suction_amp=0.0, cyclic_amp=0.20, n_steps=720, suction_mean=suction)
        rows.append(
            {
                "source_family": "Dai and Zhou (2026) / Ng et al. suction trend",
                "suction_kpa": suction,
                "model_final_plastic_strain_index": g["plastic_strain_index"].iloc[-1],
                "model_mean_stiffness_mpa": g["secant_stiffness_mpa"].mean(),
                "published_trend": "higher suction reduces permanent strain and increases resilient/dynamic stiffness",
            }
        )
    out = pd.DataFrame(rows)
    first = out.iloc[0]
    out["plastic_reduction_vs_0kpa_pct"] = 100.0 * (
        1.0 - out["model_final_plastic_strain_index"] / first["model_final_plastic_strain_index"]
    )
    out["stiffness_increase_vs_0kpa_pct"] = 100.0 * (
        out["model_mean_stiffness_mpa"] / first["model_mean_stiffness_mpa"] - 1.0
    )
    out["trend_match"] = (
        out["model_final_plastic_strain_index"].is_monotonic_decreasing
        and out["model_mean_stiffness_mpa"].is_monotonic_increasing
    )
    out.to_csv(DATA / "partial_quantitative_validation.csv", index=False)
    draw_line_plot(
        [
            ("plastic index", out["suction_kpa"].to_numpy(), out["model_final_plastic_strain_index"].to_numpy()),
            ("mean stiffness / 100", out["suction_kpa"].to_numpy(), out["model_mean_stiffness_mpa"].to_numpy() / 100.0),
        ],
        "Partial quantitative validation against published suction trends",
        "controlled suction (kPa)",
        "normalized response metric",
        FIGS / "fig13_partial_quantitative_validation.png",
    )
    return out


def main():
    models = ["constant_suction", "no_hysteresis", "hysteresis_only", "full_hysteretic_damage"]
    suction_amps = [25.0, 50.0, 75.0, 100.0]
    cyclic_amps = [0.08, 0.12, 0.16, 0.20]
    frames = [simulate_case(m, s, c) for m in models for s in suction_amps for c in cyclic_amps]
    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    results.to_csv(DATA / "benchmark_results.csv", index=False)
    summary.to_csv(DATA / "benchmark_summary.csv", index=False)
    validation = write_external_validation()
    convergence = timestep_convergence()
    write_parameter_table()
    partial_validation = partial_quantitative_validation()

    # Verification against closed-form reference components.
    p = Params()
    s_grid = np.linspace(5.0, 180.0, 80)
    sr_ref = vg_saturation(s_grid, p.alpha_dry, p.n_dry, p)
    sr_manual = p.sr_res + (p.sr_sat - p.sr_res) * (
        1.0 + (p.alpha_dry * np.maximum(s_grid, 0.0)) ** p.n_dry
    ) ** (-(1.0 - 1.0 / p.n_dry))
    pc_ref = p.pc0 * (1.0 + p.k_suction * np.log1p(s_grid / 100.0))
    pc_manual = p.pc0 + p.pc0 * p.k_suction * np.log1p(s_grid / 100.0)
    verification = pd.DataFrame(
        {
            "suction_kpa": s_grid,
            "sr_closed_form": sr_ref,
            "sr_independent_formula": sr_manual,
            "sr_abs_error": np.abs(sr_ref - sr_manual),
            "pc_closed_form": pc_ref,
            "pc_independent_formula": pc_manual,
            "pc_abs_error": np.abs(pc_ref - pc_manual),
        }
    )
    verification.to_csv(DATA / "component_verification.csv", index=False)
    verification_summary = pd.DataFrame(
        [
            {
                "check": "van_genuchten_retention",
                "max_abs_error": verification["sr_abs_error"].max(),
                "mean_abs_error": verification["sr_abs_error"].mean(),
                "status": "pass" if verification["sr_abs_error"].max() < 1e-12 else "review",
            },
            {
                "check": "suction_hardening_closed_form",
                "max_abs_error": verification["pc_abs_error"].max(),
                "mean_abs_error": verification["pc_abs_error"].mean(),
                "status": "pass" if verification["pc_abs_error"].max() < 1e-12 else "review",
            },
        ]
    )
    verification_summary.to_csv(DATA / "component_verification_summary.csv", index=False)

    focus = results[(results["suction_amp"] == 75.0) & (results["cyclic_amp"] == 0.16)]
    draw_line_plot(
        [
            (m, g["time"].to_numpy(), g["secant_stiffness_mpa"].to_numpy())
            for m, g in focus.groupby("model", sort=False)
        ],
        "Stiffness evolution under hydraulic and seismic cycles",
        "normalized time",
        "secant stiffness (MPa)",
        FIGS / "fig01_stiffness_evolution.png",
    )
    draw_line_plot(
        [
            (m, g["shear_strain"].to_numpy(), g["shear_stress_kpa"].to_numpy())
            for m, g in focus.groupby("model", sort=False)
        ],
        "Cyclic shear response for the reference benchmark",
        "shear strain",
        "shear stress (kPa)",
        FIGS / "fig02_cyclic_shear_response.png",
    )
    draw_line_plot(
        [
            ("suction", focus[focus["model"] == "full_hysteretic_damage"]["time"].to_numpy(), focus[focus["model"] == "full_hysteretic_damage"]["suction_kpa"].to_numpy()),
            ("degree saturation x100", focus[focus["model"] == "full_hysteretic_damage"]["time"].to_numpy(), 100 * focus[focus["model"] == "full_hysteretic_damage"]["degree_saturation"].to_numpy()),
        ],
        "Hydraulic path in the full hysteretic model",
        "normalized time",
        "suction / saturation",
        FIGS / "fig03_hydraulic_path.png",
    )
    draw_heat_table(summary, FIGS / "fig04_plastic_strain_ratio_heatmap.png")
    draw_metric_heat_table(
        summary,
        "stiffness_loss_pct",
        "Full model stiffness loss (%)",
        "{:.1f}",
        FIGS / "fig08_stiffness_loss_heatmap.png",
    )
    draw_line_plot(
        [
            ("hydraulic damage", focus[focus["model"] == "full_hysteretic_damage"]["time"].to_numpy(), focus[focus["model"] == "full_hysteretic_damage"]["hydraulic_damage"].to_numpy()),
            ("seismic damage", focus[focus["model"] == "full_hysteretic_damage"]["time"].to_numpy(), focus[focus["model"] == "full_hysteretic_damage"]["seismic_damage"].to_numpy()),
        ],
        "Internal damage evolution in the full model",
        "normalized time",
        "damage variable",
        FIGS / "fig09_damage_evolution.png",
    )
    draw_line_plot(
        [
            ("drying branch", s_grid, vg_saturation(s_grid, p.alpha_dry, p.n_dry, p)),
            ("wetting branch", s_grid, vg_saturation(s_grid, p.alpha_wet, p.n_wet, p)),
        ],
        "Drying and wetting soil-water retention branches",
        "suction (kPa)",
        "degree of saturation",
        FIGS / "fig10_swrc_hysteresis.png",
    )
    draw_line_plot(
        [
            (m, g["mean_effective_stress_kpa"].to_numpy(), g["shear_stress_kpa"].abs().to_numpy())
            for m, g in focus.groupby("model", sort=False)
        ],
        "Effective stress path for the reference benchmark",
        "mean effective stress p' (kPa)",
        "absolute shear stress q proxy (kPa)",
        FIGS / "fig11_effective_stress_path.png",
    )
    draw_line_plot(
        [
            ("plastic index", convergence["n_steps"].to_numpy(), convergence["final_plastic_strain_index"].to_numpy()),
            ("stiffness loss / 100", convergence["n_steps"].to_numpy(), convergence["stiffness_loss_pct"].to_numpy() / 100.0),
        ],
        "Time-step convergence for the full model",
        "number of time steps",
        "normalized metric",
        FIGS / "fig12_timestep_convergence.png",
    )
    draw_line_plot(
        [
            ("closed-form retention", s_grid, sr_ref),
            ("independent formula", s_grid, sr_manual),
        ],
        "Component verification: water-retention equation",
        "suction (kPa)",
        "degree of saturation",
        FIGS / "fig05_component_retention_verification.png",
    )
    draw_line_plot(
        [
            ("closed-form hardening", s_grid, pc_ref),
            ("independent formula", s_grid, pc_manual),
        ],
        "Component verification: suction hardening",
        "suction (kPa)",
        "preconsolidation pressure (kPa)",
        FIGS / "fig06_component_hardening_verification.png",
    )

    readme = SUPP / "README.md"
    readme.write_text(
        "# Supplementary benchmark for article 117\n\n"
        "This folder contains a reproducible material-point benchmark for the hydro-mechanical model of unsaturated soils under hydraulic and seismic cycles.\n\n"
        "Run `python scripts/run_unsat_cyclic_benchmark.py` to regenerate the CSV files and figures.\n\n"
        "Models compared: constant suction baseline, no-hysteresis suction model, hysteresis-only model, and full hysteretic-damage model.\n\n"
        "The file `data/external_validation_trend_checks.csv` records the external trend-validation sources used in the manuscript. It is a qualitative validation table, not a calibrated experimental data set.\n\n"
        "Additional QA files include `data/timestep_convergence.csv` and `data/model_parameters.csv`.\n",
        encoding="utf-8",
    )
    script_copy = SCRIPTS / "run_unsat_cyclic_benchmark.py"
    script_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    with (DATA / "benchmark_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in Params().__dict__.items():
            writer.writerow([k, v])

    print(f"rows={len(results)}")
    print(f"summary_rows={len(summary)}")
    print(SUPP)


if __name__ == "__main__":
    main()
