#!/usr/bin/env python3
"""
================================================================================
LAB 4 - INTERPOLATION TECHNIQUES
Temperature dependence of Boltzmann populations in a two-level system
================================================================================

PROBLEM
-------
A two-level system has gap dE, ground degeneracy g_g and excited degeneracy g_e.
The excited-state population fraction is

        P_e(T) = g_e exp(-dE/kB T) / ( g_g + g_e exp(-dE/kB T) )

with dE = 0.7 eV, g_g = 1, g_e = 20 and N = 1000 particles.  An experiment
supplies 15 pairs (temperature, excited counts).

MAP OF THE ASSIGNMENT ONTO THIS FILE
------------------------------------
Every bullet of the question is answered by exactly one function:

  Q1  question_1_interpolate()   "Use simple linear interpolation and Lagrange
                                  interpolation to determine the temperatures at
                                  which 30 %, 60 % and 90 % of the particles are
                                  excited."

  Q2  question_2_theory()        "Calculate the corresponding temperatures
                                  theoretically using the Boltzmann expression."

  Q3  question_3_compare()       "Compare the interpolated and theoretical
                                  values."

  Q4  question_4_which_closer()  "Determine which interpolation method gives the
                                  closer result."

  Q5  question_5_accuracy()      "Comment on the accuracy of the two methods for
                                  the three population levels."

 NOTE verification()             "Use built-in functions for verification only."
                                  Every call to a library interpolation or
                                  root-finding routine is confined to this one
                                  function.

SECTION 0 holds the constants, the experimental data, the analytic formulae and
the four hand-written numerical routines.  Nothing in Q1-Q5 calls a library
interpolation or root-finder.

Run:  python3 lab4_interpolation.py
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend: write PNGs, open no window
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 0 - CONSTANTS, DATA, ANALYTIC FORMULAE, NUMERICAL ROUTINES
# ==============================================================================

# ---- 0.1  physical constants and system parameters ---------------------------
KB   = 8.617333262e-5     # Boltzmann constant [eV/K], CODATA 2018
DE   = 0.7                # level splitting    [eV]
G_G  = 1.0                # ground-state degeneracy
G_E  = 20.0               # excited-state degeneracy
NTOT = 1000               # total number of particles

TARGETS = (0.30, 0.60, 0.90)          # the three population levels asked for
NPTS_DEFAULT = 4                      # Lagrange stencil size -> cubic

# ---- 0.2  experimental data (as supplied in the question) --------------------
T_exp = np.array([1000, 1400, 1800, 2000, 2200, 2500, 2800, 3200,
                  4000, 5000, 6500, 8000, 10000, 12000, 15000], dtype=float)
C_exp = np.array([   8,   54,  184,  254,  338,  433,  527,  607,
                   728,  795,  853,  875,  902,  908,  925], dtype=float)
P_exp = C_exp / NTOT                  # measured excited fraction

# High-temperature saturation limit.  Note this is 0.952, NOT 1.0 - the excited
# manifold is only 20-fold degenerate, so the 90 % target sits at 94.5 % of the
# asymptote.  This single fact explains most of what happens in Q5.
P_SAT = G_E / (G_G + G_E)


# ---- 0.3  analytic (theoretical) forms ---------------------------------------
def P_theory(T):
    """
    Boltzmann excited-state fraction at temperature T.  Vectorised over T.

        P_e(T) = g_e x / (g_g + g_e x),      x = exp(-dE / kB T)
    """
    x = np.exp(-DE / (KB * np.asarray(T, dtype=float)))
    return G_E * x / (G_G + G_E * x)


def T_theory(p):
    """
    EXACT inversion of the Boltzmann expression - the reference used in Q2.

    Starting from  p = g_e x / (g_g + g_e x)  with  x = exp(-dE/kB T):

        p (g_g + g_e x) = g_e x
        p g_g           = g_e x (1 - p)
        x               = g_g p / [ g_e (1 - p) ]
        -dE/(kB T)      = ln{ g_g p / [ g_e (1 - p) ] }
        T               = dE / ( kB ln[ g_e (1 - p) / (g_g p) ] )

    No root-finding is required: the inverse is closed form.  Valid for
    0 < p < P_SAT; the logarithm's argument tends to 1 (and T diverges) as
    p -> P_SAT, which is the analytic signature of saturation.
    """
    p = np.asarray(p, dtype=float)
    return DE / (KB * np.log(G_E * (1.0 - p) / (G_G * p)))


def dCdT_theory(T):
    """
    Analytic sensitivity of count to temperature, used in Q5.

    Differentiating P_e(T) and simplifying gives the compact result

        dP/dT = P(1-P) dE / (kB T^2)      =>   dC/dT = N dP/dT

    The reciprocal dT/dC is the CONDITION NUMBER of the inverse problem: how
    many kelvin of error one count of measurement error produces.
    """
    P = P_theory(T)
    return NTOT * P * (1.0 - P) * DE / (KB * np.asarray(T, float) ** 2)


def logit_var(P):
    """
    Linearising change of variable (cross-check in Q5).

        y = ln[ g_g P / ( g_e (1 - P) ) ] = -dE / (kB T)

    i.e. y is EXACTLY linear in 1/T.  Linear interpolation of 1/T against y
    therefore carries zero truncation error at any node spacing; the only error
    left is measurement noise.
    """
    P = np.asarray(P, float)
    return np.log(G_G * P / (G_E * (1.0 - P)))


# ---- 0.4  hand-written numerical routines ------------------------------------
# These are the "from scratch" content of the lab.  No library interpolation is
# used here or anywhere in Q1-Q5.

def bracket(x_nodes, xq):
    """
    Binary search for the interval containing xq.

    Returns index i such that  x_nodes[i] <= xq <= x_nodes[i+1].
    x_nodes must be strictly increasing.  Cost O(log n) instead of the O(n) of
    a linear scan.  Extrapolation is refused rather than silently allowed,
    because an interpolation formula evaluated outside its node range carries
    no error bound.
    """
    n = len(x_nodes)
    if xq < x_nodes[0] or xq > x_nodes[-1]:
        raise ValueError(f"xq = {xq} lies outside the data range "
                         f"[{x_nodes[0]}, {x_nodes[-1]}] - extrapolation refused.")
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xq < x_nodes[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def linear_interp(x_nodes, y_nodes, xq):
    """
    Piecewise LINEAR interpolation - method (a) of Q1.

        y(xq) = y_i + (xq - x_i) (y_{i+1} - y_i) / (x_{i+1} - x_i)

    The straight chord joining the two bracketing nodes.  Truncation error
    E = (1/2) y''(xi) (xq - x_i)(xq - x_{i+1}) = O(h^2 y'').
    """
    i = bracket(x_nodes, xq)
    x0, x1 = x_nodes[i], x_nodes[i + 1]
    y0, y1 = y_nodes[i], y_nodes[i + 1]
    return y0 + (xq - x0) * (y1 - y0) / (x1 - x0)


def lagrange_basis(x_nodes, xq):
    """
    Cardinal (basis) functions of the Lagrange polynomial.

        L_k(xq) = prod_{j != k} (xq - x_j) / (x_k - x_j)

    By construction L_k(x_m) = 1 when m == k and 0 otherwise, so the polynomial
    sum_k y_k L_k passes exactly through every node.  Two useful properties:
      * sum_k L_k(xq) = 1 identically  -> used as a numerical self-check;
      * if any |L_k| > 1 the interpolant is locally EXTRAPOLATING from that
        node rather than averaging between nodes - a warning of oscillation.
    """
    n = len(x_nodes)
    L = np.ones(n, dtype=float)
    for k in range(n):
        for j in range(n):
            if j != k:
                L[k] *= (xq - x_nodes[j]) / (x_nodes[k] - x_nodes[j])
    return L


def lagrange_interp(x_nodes, y_nodes, xq):
    """
    LAGRANGE interpolation - method (b) of Q1:  y(xq) = sum_k y_k L_k(xq).

    Error term  E = y^(n)(xi) / n! * prod_j (xq - x_j).  With n = 2 nodes this
    reduces exactly to linear_interp, so the two methods of Q1 differ only in
    the order of the local polynomial.
    """
    L = lagrange_basis(np.asarray(x_nodes, float), xq)
    return float(np.dot(L, np.asarray(y_nodes, float)))


def select_stencil(x_nodes, xq, npts):
    """
    Choose which npts nodes the Lagrange polynomial is built on.

    Takes npts CONTIGUOUS nodes centred as symmetrically as possible about xq,
    clipped at the ends of the table.  Centring matters: an off-centre stencil
    turns interpolation into partial extrapolation and inflates |L_k|.
    """
    n = len(x_nodes)
    npts = min(npts, n)
    i = bracket(x_nodes, xq)
    start = i - (npts // 2 - 1)
    start = max(0, min(start, n - npts))
    return slice(start, start + npts)


def lagrange_local(x_nodes, y_nodes, xq, npts=NPTS_DEFAULT):
    """Local Lagrange interpolation.  Returns (value, stencil_slice)."""
    sl = select_stencil(x_nodes, xq, npts)
    return lagrange_interp(x_nodes[sl], y_nodes[sl], xq), sl


def linearised_interp(T_nodes, P_nodes, p_target):
    """Linear interpolation of 1/T against the linearising variable y (Q5)."""
    y_nodes = logit_var(P_nodes)
    return 1.0 / linear_interp(y_nodes, 1.0 / T_nodes, logit_var(p_target))


# ---- 0.5  small output helpers -----------------------------------------------
def rule(char="-", n=78):
    return char * n


def banner(text):
    print("\n" + rule("="))
    print(text)
    print(rule("="))


# ==============================================================================
# QUESTION 1
# "Use simple linear interpolation and Lagrange interpolation to determine the
#  temperatures at which 30 %, 60 % and 90 % of the particles are excited."
# ==============================================================================
def question_1_interpolate(npts=NPTS_DEFAULT):
    """
    STRATEGY - inverse interpolation.

    The table gives counts as a function of temperature, C(T), but the question
    asks for the temperature at a prescribed count, T(C).  Because the tabulated
    counts are strictly monotonic, the roles of abscissa and ordinate can be
    exchanged and T(C) interpolated directly.  This avoids nesting a root-finder
    inside a forward interpolant.

    So throughout:  x-nodes = C_exp (counts),  y-nodes = T_exp (temperatures),
    and the query point is C_q = p * N.

    Returns a dict keyed by target population p.
    """
    banner("QUESTION 1 - INTERPOLATED TEMPERATURES AT 30 %, 60 %, 90 %")

    monotonic = bool(np.all(np.diff(C_exp) > 0))
    print(f"\n  Counts strictly increasing (inverse interpolation well-posed): "
          f"{monotonic}")
    if not monotonic:
        raise RuntimeError("counts are not monotonic - T(C) is not single-valued")

    results = {}
    for p in TARGETS:
        Cq = p * NTOT                       # target count, e.g. 0.30 * 1000 = 300
        i = bracket(C_exp, Cq)              # bracketing interval

        T_lin = linear_interp(C_exp, T_exp, Cq)
        T_lag, sl = lagrange_local(C_exp, T_exp, Cq, npts=npts)
        L = lagrange_basis(C_exp[sl], Cq)   # cardinal weights, reused in Q5

        results[p] = dict(Cq=Cq, bracket=i, stencil=sl, weights=L,
                          T_lin=T_lin, T_lag=T_lag)

        print(f"\n  --- target {100*p:.0f} %   (C_q = {Cq:.0f} counts) ---")
        print(f"      bracketing nodes : T = [{T_exp[i]:.0f}, {T_exp[i+1]:.0f}] K"
              f"   <->  C = [{C_exp[i]:.0f}, {C_exp[i+1]:.0f}]")
        print(f"      (a) linear   (2-pt)  T = {T_lin:11.3f} K")
        print(f"          T = {T_exp[i]:.0f} + ({Cq:.0f}-{C_exp[i]:.0f})/"
              f"({C_exp[i+1]:.0f}-{C_exp[i]:.0f}) x "
              f"({T_exp[i+1]:.0f}-{T_exp[i]:.0f})")
        print(f"      (b) Lagrange ({npts}-pt)  T = {T_lag:11.3f} K")
        print(f"          stencil C = {C_exp[sl].astype(int).tolist()}")
        print(f"          weights L = {np.round(L, 5).tolist()}   "
              f"(sum = {L.sum():.10f})")

    return results


# ==============================================================================
# QUESTION 2
# "Calculate the corresponding temperatures theoretically using the Boltzmann
#  expression."
# ==============================================================================
def question_2_theory():
    """
    Evaluate the closed-form inversion T(p) derived in T_theory().

    Each value is back-substituted into P_theory() to confirm the algebra:
    P_theory(T_theory(p)) must return p to machine precision.
    """
    banner("QUESTION 2 - THEORETICAL TEMPERATURES FROM THE BOLTZMANN EXPRESSION")

    print("\n      T(p) = dE / ( kB * ln[ g_e (1-p) / (g_g p) ] )")
    print(f"      dE = {DE} eV,  g_g = {G_G:.0f},  g_e = {G_E:.0f},  "
          f"kB = {KB:.9e} eV/K")
    print(f"      saturation limit P_sat = g_e/(g_g+g_e) = {P_SAT:.6f}  "
          f"({NTOT*P_SAT:.1f} counts)\n")

    T_th = {}
    for p in TARGETS:
        T_th[p] = float(T_theory(p))
        arg = G_E * (1 - p) / (G_G * p)
        print(f"  p = {p:.2f} :  ln-argument = {arg:9.5f},  ln = {np.log(arg):8.5f}"
              f"   ->  T = {T_th[p]:11.3f} K")
        print(f"             back-check  P_theory(T) = "
              f"{float(P_theory(T_th[p])):.10f}")

    return T_th


# ==============================================================================
# QUESTION 3
# "Compare the interpolated and theoretical values."
# ==============================================================================
def question_3_compare(interp, T_th):
    """
    Tabulate absolute and relative deviations of each interpolant from theory.

        Delta      = T_interp - T_theory
        rel. error = Delta / T_theory
    """
    banner("QUESTION 3 - COMPARISON OF INTERPOLATED AND THEORETICAL VALUES")

    hdr = (f"\n  {'p':>5} {'T_theory':>11} {'T_linear':>11} {'Delta':>9} {'rel %':>8}"
           f" {'T_Lagrange':>12} {'Delta':>9} {'rel %':>8}")
    print(hdr)
    print("  " + rule("-", len(hdr) - 3))

    comp = {}
    for p in TARGETS:
        r = interp[p]
        e_lin = r["T_lin"] - T_th[p]
        e_lag = r["T_lag"] - T_th[p]
        comp[p] = dict(e_lin=e_lin, e_lag=e_lag,
                       r_lin=100 * e_lin / T_th[p], r_lag=100 * e_lag / T_th[p])
        print(f"  {p:5.2f} {T_th[p]:11.2f} {r['T_lin']:11.2f} {e_lin:+9.2f} "
              f"{comp[p]['r_lin']:+8.3f} {r['T_lag']:12.2f} {e_lag:+9.2f} "
              f"{comp[p]['r_lag']:+8.3f}")

    print("\n  Both schemes track theory closely at 30 % and 60 % and deviate")
    print("  strongly at 90 %.  Q5 shows this is a property of the problem")
    print("  rather than of the algorithms.")
    return comp


# ==============================================================================
# QUESTION 4
# "Determine which interpolation method gives the closer result."
# ==============================================================================
def question_4_which_closer(comp):
    """
    Declare a winner per level on |Delta|, and record the margin.

    The margin matters as much as the winner: if the two methods differ by far
    less than the measurement-noise floor computed in Q5, the "win" is not
    statistically meaningful and must not be reported as a ranking.
    """
    banner("QUESTION 4 - WHICH METHOD IS CLOSER?")

    print(f"\n  {'p':>5} {'|err| linear':>14} {'|err| Lagrange':>16} "
          f"{'closer':>10} {'margin (K)':>12}")
    print("  " + rule("-", 62))

    verdict = {}
    for p in TARGETS:
        a, b = abs(comp[p]["e_lin"]), abs(comp[p]["e_lag"])
        win = "linear" if a < b else "Lagrange"
        verdict[p] = dict(winner=win, margin=abs(a - b))
        print(f"  {p:5.2f} {a:14.2f} {b:16.2f} {win:>10} {abs(a-b):12.2f}")

    wins_lin = sum(v["winner"] == "linear" for v in verdict.values())
    print(f"\n  Raw tally on the experimental data:  linear {wins_lin} / "
          f"Lagrange {len(TARGETS) - wins_lin}")
    print("  The verdict is INCONSISTENT across the three levels.  Q5 separates")
    print("  method error from measurement error to explain why.")
    return verdict


# ==============================================================================
# QUESTION 5
# "Comment on the accuracy of the two methods for the three population levels."
# ==============================================================================
def question_5_accuracy(interp, T_th, comp, verdict):
    """
    Six pieces of evidence, in the order in which they build the argument:

      5.1  residuals of the data against the Boltzmann law  -> noise amplitude
      5.2  conditioning dT/dC and the resulting noise floor -> why 90 % is hard
      5.3  noise-free control run                           -> pure method error
      5.4  stencil geometry and cardinal weights            -> why the cubic
                                                               fails at 90 %
      5.5  effect of polynomial order, incl. the global polynomial
      5.6  physics-informed linearisation as a cross-check
    """
    banner("QUESTION 5 - ACCURACY OF THE TWO METHODS AT THE THREE LEVELS")

    # ---- 5.1  how noisy is the data? -----------------------------------------
    print("\n  [5.1] RESIDUALS OF THE DATA AGAINST THE BOLTZMANN LAW\n")
    C_clean = NTOT * P_theory(T_exp)      # counts with no measurement noise
    resid = C_exp - C_clean
    rms = float(np.sqrt(np.mean(resid ** 2)))

    print(f"      {'T (K)':>8} {'C_exp':>7} {'C_theory':>10} {'residual':>10}")
    for t, c, ct, r in zip(T_exp, C_exp, C_clean, resid):
        print(f"      {t:8.0f} {c:7.0f} {ct:10.2f} {r:+10.2f}")
    print(f"\n      RMS residual = {rms:.2f} counts,  "
          f"max |residual| = {np.max(np.abs(resid)):.2f} counts")
    print("      Residuals alternate in sign with no trend in T: measurement")
    print("      scatter of about +-0.4 % of N, not a systematic model error.")

    i90 = int(np.where(T_exp == 10000)[0][0])
    print(f"\n      CRITICAL POINT: at T = 10000 K theory gives "
          f"{C_clean[i90]:.2f} counts but the")
    print(f"      experiment reads {C_exp[i90]:.0f}.  The true 90 % crossing "
          f"({T_th[0.90]:.0f} K) lies BEYOND")
    print("      the 10000 K node, but the noisy data place it BEFORE.  The 90 %")
    print("      level is bracketed by the WRONG PAIR OF NODES, and no")
    print("      interpolation scheme can recover from that.")

    # ---- 5.2  conditioning ----------------------------------------------------
    print("\n  [5.2] CONDITIONING OF THE INVERSE PROBLEM\n")
    print("        dC/dT = N P(1-P) dE/(kB T^2)   ->   dT/dC = 1/(dC/dT)\n")
    print(f"      {'p':>5} {'T (K)':>10} {'dC/dT':>12} {'dT/dC (K/count)':>18} "
          f"{'noise floor (K)':>17}")
    floors = {}
    for p in TARGETS:
        s = float(dCdT_theory(T_th[p]))
        floors[p] = rms / s               # RMS noise x condition number
        print(f"      {p:5.2f} {T_th[p]:10.1f} {s:12.5f} {1/s:18.2f} "
              f"{floors[p]:17.1f}")
    amp = float(dCdT_theory(T_th[0.30])) / float(dCdT_theory(T_th[0.90]))
    print(f"\n      The inversion is {amp:.0f}x more ill-conditioned at 90 % than at")
    print("      30 %.  The last column is the temperature uncertainty that NO")
    print("      interpolation scheme can beat, given the scatter in the data.")

    # ---- 5.3  noise-free control ---------------------------------------------
    print("\n  [5.3] NOISE-FREE CONTROL - separating METHOD error from NOISE\n")
    print("      Repeat the whole Q1 inversion on synthetic counts C = N P_e(T)")
    print("      at the SAME temperature nodes.  Any error left is pure")
    print("      truncation error.\n")
    print(f"      {'p':>5} | {'noise-free':^26} | {'real data':^26}")
    print(f"      {'':>5} | {'linear':>12}{'Lagrange':>14} | "
          f"{'linear':>12}{'Lagrange':>14}")
    print("      " + rule("-", 66))
    control = {}
    for p in TARGETS:
        Cq = p * NTOT
        el_c = linear_interp(C_clean, T_exp, Cq) - T_th[p]
        eg_c = lagrange_local(C_clean, T_exp, Cq, NPTS_DEFAULT)[0] - T_th[p]
        control[p] = (el_c, eg_c)
        print(f"      {p:5.2f} | {el_c:+12.2f}{eg_c:+14.2f} | "
              f"{comp[p]['e_lin']:+12.2f}{comp[p]['e_lag']:+14.2f}")

    print("\n      Truncation error suppressed by going linear -> cubic:")
    for p in TARGETS:
        el_c, eg_c = control[p]
        print(f"        p = {p:.2f} :  {abs(el_c):8.3f} K -> {abs(eg_c):7.3f} K"
              f"   ({abs(el_c)/abs(eg_c):5.1f}x better)")

    print("\n      ERROR BUDGET (K):")
    print(f"      {'p':>5} {'truncation':>12} {'noise floor':>13} "
          f"{'observed lin':>14} {'observed Lag':>14}")
    for p in TARGETS:
        print(f"      {p:5.2f} {abs(control[p][0]):12.2f} {floors[p]:13.1f} "
              f"{abs(comp[p]['e_lin']):14.1f} {abs(comp[p]['e_lag']):14.1f}")
    print("\n      At every level the noise floor EXCEEDS the truncation error, so")
    print("      the Q4 winner is decided by noise, not by the merits of the two")
    print("      methods.  On clean data the cubic wins at all three levels.")

    # ---- 5.4  why the cubic fails at 90 % -------------------------------------
    print("\n  [5.4] STENCIL GEOMETRY AND CARDINAL WEIGHTS\n")
    print(f"      {'p':>5} {'stencil gaps (counts)':>26} {'max/min':>9} "
          f"{'max |L_k|':>11} {'sum |L_k|':>11}")
    for p in TARGETS:
        sl = interp[p]["stencil"]
        gaps = np.diff(C_exp[sl])
        L = interp[p]["weights"]
        print(f"      {p:5.2f} {str(gaps.astype(int).tolist()):>26} "
              f"{gaps.max()/gaps.min():9.2f} {np.max(np.abs(L)):11.4f} "
              f"{np.sum(np.abs(L)):11.4f}")
    print("\n      At 90 % the gaps are 4.5:1 uneven and one weight exceeds unity")
    print("      (L_2 = 1.1842, L_3 = -0.2158).  A weight outside [0,1] means the")
    print("      interpolant is locally EXTRAPOLATING from that node rather than")
    print("      averaging between nodes - the polynomial must bend sharply to")
    print("      reach 908 counts at 12000 K after 902 at 10000 K.  That is the")
    print("      oscillation visible in fig_q1_stencils.png.")

    print("\n      Curvature of the inverse function T(C):")
    h = 1e-4
    curv = {}
    for p in TARGETS:
        d2 = (float(T_theory(p + h)) - 2 * float(T_theory(p))
              + float(T_theory(p - h))) / h ** 2 / NTOT ** 2
        curv[p] = d2
        print(f"        p = {p:.2f} :  d2T/dC2 = {d2:10.5f} K/count^2")
    print(f"      Growth from 30 % to 90 %: factor {curv[0.90]/curv[0.30]:.0f}.")
    print("      Linear truncation error scales as h^2 T'', so this alone orders")
    print("      the three levels.")

    # ---- 5.5  effect of polynomial order --------------------------------------
    print("\n  [5.5] EFFECT OF LAGRANGE POLYNOMIAL ORDER\n")
    orders = [2, 3, 4, 5, 6, 8, 15]
    print(f"      {'n':>4} {'deg':>4} |" +
          "".join(f"{f'  p = {p:.2f}':>26}" for p in TARGETS))
    print(f"      {'':>4} {'':>4} |" +
          "".join(f"{'T (K)':>14}{'err (K)':>12}" for _ in TARGETS))
    print("      " + rule("-", 74))
    order_data = {p: [] for p in TARGETS}
    for npts in orders:
        row = f"      {npts:4d} {npts-1:4d} |"
        for p in TARGETS:
            Tv, _ = lagrange_local(C_exp, T_exp, p * NTOT, npts=npts)
            order_data[p].append(Tv)
            row += f"{Tv:14.2f}{Tv - T_th[p]:12.2f}"
        print(row)
    print("\n      n = 2 reproduces piecewise-linear; n = 15 is the global")
    print("      polynomial through all 15 points.  Errors plateau by n ~ 4 - the")
    print("      floor is data noise, not polynomial order - and the global")
    print("      polynomial returns an UNPHYSICAL NEGATIVE temperature at 30 %")
    print("      (Runge oscillation) while still passing exactly through every")
    print("      data point.")

    # ---- 5.6  physics-informed cross-check ------------------------------------
    print("\n  [5.6] PHYSICS-INFORMED LINEARISATION (zero truncation error)\n")
    lin_phys = {}
    for p in TARGETS:
        v = linearised_interp(T_exp, P_exp, p)
        lin_phys[p] = v
        print(f"      p = {p:.2f} :  T = {v:10.2f} K   "
              f"(err {v - T_th[p]:+8.2f} K, {100*(v-T_th[p])/T_th[p]:+6.3f} %)")
    print("\n      Not better than the cubic at 90 % - which is the point: once")
    print("      truncation error is removed entirely, only the noise floor")
    print("      remains.  The honest report is T_90% = (9.8 +- 0.5) x 10^3 K.")

    # ---- answer ----------------------------------------------------------------
    print("\n  " + rule("-"))
    print("  ANSWER TO Q5:")
    print("    * 30 % : both methods excellent (~0.2 %). Steep, well-conditioned")
    print("             region; the 0.64 K gap between them is far below the")
    print("             9.4 K noise floor, so they are statistically tied.")
    print("    * 60 % : both good (~0.8 %). Lagrange nominally closer, but again")
    print("             well inside the 18 K noise floor.")
    print("    * 90 % : both poor (3-6 %), and the cubic is genuinely worse.")
    print("             Near saturation the inversion is ill-conditioned")
    print("             (141.6 K/count), the bracket is wrong, and the uneven")
    print("             stencil drives the polynomial into oscillation.")

    return dict(resid=resid, rms=rms, floors=floors, control=control,
                orders=orders, order_data=order_data, curv=curv,
                lin_phys=lin_phys)


# ==============================================================================
# NOTE - "Use built-in functions for verification only."
# Every library interpolation / root-finding call in this file lives here.
# ==============================================================================
def verification(interp, T_th):
    """
    Check the three hand-written pieces against their library equivalents:

        linear_interp    vs  numpy.interp
        lagrange_interp  vs  scipy.interpolate.lagrange
        T_theory         vs  scipy.optimize.brentq on P_theory(T) - p = 0
    """
    banner("NOTE - VERIFICATION AGAINST BUILT-IN FUNCTIONS (checking only)")

    from scipy.interpolate import lagrange as sp_lagrange
    from scipy.optimize import brentq

    ok = True
    for p in TARGETS:
        Cq = p * NTOT
        r = interp[p]
        sl = r["stencil"]

        d1 = abs(float(np.interp(Cq, C_exp, T_exp)) - r["T_lin"])
        d2 = abs(float(sp_lagrange(C_exp[sl], T_exp[sl])(Cq)) - r["T_lag"])
        d3 = abs(brentq(lambda T: P_theory(T) - p, 100.0, 1.0e6, xtol=1e-10)
                 - T_th[p])
        ok &= (d1 < 1e-9) and (d2 < 1e-6) and (d3 < 1e-6)

        print(f"\n  p = {p:.2f}")
        print(f"      |linear_interp   - numpy.interp|          = {d1:.3e}")
        print(f"      |lagrange_interp - scipy.lagrange|        = {d2:.3e}")
        print(f"      |T_theory        - scipy.optimize.brentq| = {d3:.3e}")

    print(f"\n  ALL CHECKS PASSED: {ok}")
    return ok


# ==============================================================================
# FIGURES - each named after the question it supports
# ==============================================================================
def make_figures(interp, T_th, comp, diag):
    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 150, "savefig.dpi": 150, "axes.axisbelow": True,
        "mathtext.fontset": "dejavusans",
    })
    col = {"lin": "#1f77b4", "lag": "#d62728", "th": "#333333"}
    from matplotlib.lines import Line2D

    # ---- Q1: overview --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    Tg = np.logspace(np.log10(800), np.log10(2.0e4), 800)
    for a, xs in zip(ax, ("linear", "log")):
        a.plot(Tg, NTOT * P_theory(Tg), "-", color=col["th"], lw=1.6,
               label="Boltzmann law", zorder=2)
        a.axhline(NTOT * P_SAT, ls=":", color="grey", lw=1.2)
        a.plot(T_exp, C_exp, "o", ms=5.5, mfc="white", mec=col["lin"], mew=1.4,
               label="experiment", zorder=3)
        for p in TARGETS:
            a.axhline(p * NTOT, ls="--", lw=0.8, color="grey", alpha=0.8)
            a.plot(T_th[p], p * NTOT, "*", ms=13, color=col["th"], zorder=5)
            a.plot(interp[p]["T_lin"], p * NTOT, "s", ms=6, color=col["lin"], zorder=4)
            a.plot(interp[p]["T_lag"], p * NTOT, "^", ms=6, color=col["lag"], zorder=4)
        a.set_xscale(xs)
        a.set_xlabel("temperature $T$ (K)")
        a.set_ylabel("excited counts  $N P_e$")
        a.set_ylim(-30, 1000)
    ax[0].set_xlim(800, 1.55e4)
    ax[0].text(1.05e4, 975, r"$P_{\rm sat}=0.952$", fontsize=8, color="grey")
    ax[0].legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax[0].set_title("(a) linear $T$ axis", fontsize=10, loc="left")
    ax[1].set_title("(b) logarithmic $T$ axis", fontsize=10, loc="left")
    ax[1].legend(handles=[
        Line2D([], [], ls="", marker="*", ms=12, color=col["th"], label="theory (Q2)"),
        Line2D([], [], ls="", marker="s", ms=6, color=col["lin"], label="linear (Q1)"),
        Line2D([], [], ls="", marker="^", ms=6, color=col["lag"], label="Lagrange (Q1)"),
    ], loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig("fig_q1_overview.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Q1: local stencils --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    spans = {0.30: (1700, 2600), 0.60: (2400, 4200), 0.90: (6000, 13500)}
    for a, p in zip(axes, TARGETS):
        lo, hi = spans[p]
        Tz = np.linspace(lo, hi, 500)
        a.plot(Tz, NTOT * P_theory(Tz), "-", color=col["th"], lw=1.6, label="Boltzmann")
        m = (T_exp >= lo) & (T_exp <= hi)
        a.plot(T_exp[m], C_exp[m], "o", ms=6, mfc="white", mec="k", mew=1.2,
               label="data", zorder=4)
        i = interp[p]["bracket"]
        a.plot(T_exp[i:i+2], C_exp[i:i+2], "-", color=col["lin"], lw=2.2,
               alpha=0.85, label="linear segment", zorder=3)
        sl = interp[p]["stencil"]
        Cs, Ts = C_exp[sl], T_exp[sl]
        Cd = np.linspace(Cs.min(), Cs.max(), 400)
        a.plot([lagrange_interp(Cs, Ts, c) for c in Cd], Cd, "-", color=col["lag"],
               lw=1.6, alpha=0.9, label="Lagrange cubic", zorder=3)
        a.plot(Ts, Cs, "o", ms=9, mfc="none", mec=col["lag"], mew=1.3,
               label="stencil", zorder=2)
        a.axhline(p * NTOT, ls="--", lw=0.9, color="grey")
        a.plot(T_th[p], p * NTOT, "*", ms=15, color=col["th"], zorder=6)
        a.plot(interp[p]["T_lin"], p * NTOT, "s", ms=7, color=col["lin"], zorder=6)
        a.plot(interp[p]["T_lag"], p * NTOT, "^", ms=7, color=col["lag"], zorder=6)
        a.set_title(f"{100*p:.0f} %   $T_{{th}}$ = {T_th[p]:.1f} K\n"
                    f"$\\Delta_{{lin}}$ = {comp[p]['e_lin']:+.1f} K,  "
                    f"$\\Delta_{{Lag}}$ = {comp[p]['e_lag']:+.1f} K", fontsize=9.5)
        a.set_xlim(lo, hi)
        a.set_xlabel("$T$ (K)")
    axes[0].set_ylabel("excited counts")
    axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("fig_q1_stencils.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Q3/Q4: error comparison --------------------------------------------
    fig, a = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(TARGETS)); w = 0.34
    a.bar(x - w/2, [abs(comp[p]["e_lin"]) for p in TARGETS], w,
          color=col["lin"], label="linear", zorder=3)
    a.bar(x + w/2, [abs(comp[p]["e_lag"]) for p in TARGETS], w,
          color=col["lag"], label="Lagrange (4-pt)", zorder=3)
    for xi, p in zip(x, TARGETS):
        a.plot([xi - 0.46, xi + 0.46], [diag["floors"][p]] * 2, "k--", lw=1.5,
               zorder=4, label="noise floor" if xi == 0 else None)
    a.set_yscale("log")
    a.set_xticks(x, [f"{100*p:.0f} %" for p in TARGETS])
    a.set_xlabel("target population")
    a.set_ylabel("|error| vs theory (K)")
    a.set_title("Q3/Q4: deviation from theory, against the noise floor",
                fontsize=10, loc="left")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_q3_errors.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Q5: accuracy diagnostics -------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for p, c in zip(TARGETS, ["#1f77b4", "#ff7f0e", "#d62728"]):
        errs = [abs(v - T_th[p]) / T_th[p] * 100 for v in diag["order_data"][p]]
        ax[0].plot(diag["orders"], errs, "o-", color=c, ms=5, label=f"{100*p:.0f} %")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("stencil points $n$  (degree $n-1$)")
    ax[0].set_ylabel("|relative error| (%)")
    ax[0].set_title("(a) error vs polynomial order", fontsize=10, loc="left")
    ax[0].legend(fontsize=8, title="target", title_fontsize=8)

    Cd = np.linspace(C_exp.min(), C_exp.max(), 900)
    ax[1].plot(Cd, [lagrange_interp(C_exp, T_exp, c) for c in Cd], "-",
               color=col["lag"], lw=1.4, label="degree-14 global")
    ax[1].plot(Cd, np.interp(Cd, C_exp, T_exp), "-", color=col["lin"], lw=1.4,
               label="piecewise linear")
    ax[1].plot(C_exp, T_exp, "o", ms=5.5, mfc="white", mec="k", mew=1.2, label="nodes")
    ax[1].set_ylim(-8000, 30000)
    ax[1].axhspan(-8000, 0, color="red", alpha=0.07)
    ax[1].text(60, -5500, "unphysical $T<0$", color="red", fontsize=8)
    ax[1].set_xlabel("counts $C$"); ax[1].set_ylabel("$T$ (K)")
    ax[1].set_title("(b) Runge oscillation", fontsize=10, loc="left")
    ax[1].legend(fontsize=8, loc="upper left")

    Pg = np.linspace(0.02, 0.94, 600)
    ax[2].plot(Pg, 1.0 / dCdT_theory(T_theory(Pg)), "-", color="#333333", lw=1.8)
    ax[2].set_yscale("log")
    for p, c in zip(TARGETS, ["#1f77b4", "#ff7f0e", "#d62728"]):
        s = 1.0 / float(dCdT_theory(T_th[p]))
        ax[2].plot(p, s, "o", ms=8, color=c)
        ax[2].annotate(f"{100*p:.0f} %: {s:.1f}", (p, s), textcoords="offset points",
                       xytext=(-10, 8), fontsize=8.5, color=c, ha="right")
    ax[2].axvline(P_SAT, ls=":", color="grey")
    ax[2].set_xlabel("target population $p$")
    ax[2].set_ylabel(r"$|dT/dC|$ (K per count)")
    ax[2].set_title("(c) conditioning", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("fig_q5_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    print("\n  Figures written:")
    for f in ("fig_q1_overview.png", "fig_q1_stencils.png",
              "fig_q3_errors.png", "fig_q5_accuracy.png"):
        print(f"    {f}")


# ==============================================================================
# DRIVER - runs the five questions in the order the assignment asks them
# ==============================================================================
def main():
    print(rule("="))
    print("LAB 4 : INTERPOLATION TECHNIQUES")
    print("Boltzmann population inversion by linear and Lagrange interpolation")
    print(rule("="))
    print(f"  dE = {DE} eV   g_g = {G_G:.0f}   g_e = {G_E:.0f}   N = {NTOT}")
    print(f"  kB = {KB:.9e} eV/K")
    print(f"  Lagrange stencil: {NPTS_DEFAULT} points (cubic)")

    interp  = question_1_interpolate()                            # Q1
    T_th    = question_2_theory()                                 # Q2
    comp    = question_3_compare(interp, T_th)                    # Q3
    verdict = question_4_which_closer(comp)                       # Q4
    diag    = question_5_accuracy(interp, T_th, comp, verdict)    # Q5
    verification(interp, T_th)                                    # Note

    banner("FIGURES")
    make_figures(interp, T_th, comp, diag)
    print("\n" + rule("="))
    return interp, T_th, comp, verdict, diag


if __name__ == "__main__":
    main()
