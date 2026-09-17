"""
AV1 - Introducao a Inferencia I
Estudo de Monte Carlo: EMV vs EMM na distribuicao Lomax(alpha, lambda)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)


# ---------- 1. Geracao de amostras (metodo da inversa) ----------
# F(x) = 1 - (1 + x/lam)^(-a)  =>  X = lam * (U^(-1/a) - 1)
def rlomax(n, a, lam, rng=RNG):
    u = rng.random(n)
    return lam * (u ** (-1.0 / a) - 1.0)


# ---------- 2. Estimador de Maxima Verossimilhanca ----------
# A verossimilhanca da Lomax tem uma crista plana (alpha->inf, lambda->inf com
# lambda/alpha fixo => Exponencial). Usamos a log-veros. PERFILADA em lambda:
#   alpha_hat(lam) = n / S(lam),  S(lam) = sum log(1 + x_i/lam)
#   lp(lam) = n*log(n) - n*log(S) - n*log(lam) - S - n
# e sinalizamos as replicas em que o otimo bate na fronteira (nao convergiu).
def neg_perfil(u, x):
    lam = np.exp(u)
    S = np.sum(np.log1p(x / lam))
    n = len(x)
    return -(n * np.log(n) - n * np.log(S) - n * np.log(lam) - S - n)


def emv(x):
    # ancorar a busca na MEDIANA (a media nao existe se alpha <= 1)
    c = np.log(np.median(x))
    lo, hi = c - 10.0, c + 10.0
    res = minimize_scalar(neg_perfil, bounds=(lo, hi), args=(x,), method="bounded")
    u = res.x
    if not res.success or min(abs(u - lo), abs(u - hi)) < 1e-3:
        return np.array([np.nan, np.nan])        # crista plana: sem otimo interior
    lam = np.exp(u)
    a = len(x) / np.sum(np.log1p(x / lam))
    return np.array([a, lam])


# ---------- 3. Estimador de Momentos ----------
def emm(x):
    m1 = np.mean(x)
    m2 = np.mean(x ** 2)
    den = m2 - 2.0 * m1 ** 2
    if den <= 0:                               # momentos nao existem (alpha <= 2)
        return np.array([np.nan, np.nan])
    a = 2.0 * (m2 - m1 ** 2) / den
    lam = m1 * (a - 1.0)
    return np.array([a, lam])


# ---------- 4. Simulacao ----------
R = 5000
NS = [50, 200, 1000]
ALPHAS = [0.5, 2.0, 10.0]
LAM = 2.0

linhas = []
for a0 in ALPHAS:
    for n in NS:
        est = {"EMV": np.full((R, 2), np.nan), "EMM": np.full((R, 2), np.nan)}
        for r in range(R):
            x = rlomax(n, a0, LAM)
            est["EMV"][r] = emv(x)
            est["EMM"][r] = emm(x)

        for met in ("EMV", "EMM"):
            E = est[met]
            ok = np.isfinite(E).all(axis=1)
            E = E[ok]
            for j, (nome, verdade) in enumerate([("alpha", a0), ("lambda", LAM)]):
                v = E[:, j]
                linhas.append({
                    "metodo": met, "param": nome, "alpha_real": a0, "n": n,
                    "media": v.mean() if len(v) else np.nan,
                    "vies": v.mean() - verdade if len(v) else np.nan,
                    "eqm": np.mean((v - verdade) ** 2) if len(v) else np.nan,
                    "taxa_valida": ok.mean(),
                })

df = pd.DataFrame(linhas)
df.to_csv("resultados_lomax.csv", index=False)
print(df.to_string(index=False, float_format=lambda z: f"{z:10.4f}"))


# ---------- 5. Graficos do vies ----------
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
for c, a0 in enumerate(ALPHAS):
    for l, p in enumerate(["alpha", "lambda"]):
        ax = axes[l, c]
        for met, mk in [("EMV", "o-"), ("EMM", "s--")]:
            s = df[(df.metodo == met) & (df.param == p) & (df.alpha_real == a0)]
            ax.plot(s.n, s.vies, mk, label=met)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(NS); ax.set_xticklabels(NS)
        ax.set_title(f"vies de {p}  |  alpha = {a0}")
        ax.set_xlabel("n"); ax.set_ylabel("vies")
        ax.legend()
plt.tight_layout()
plt.savefig("vies_lomax.png", dpi=150)
print("\nSalvo: resultados_lomax.csv, vies_lomax.png")