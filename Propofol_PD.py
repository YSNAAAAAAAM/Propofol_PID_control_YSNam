# 치과마취의 이해1 과제
# Yoon-Seo Nam, 2025-06-02

import numpy as np

def pd_model(y, params):
    """
    Differential equations for the PD model.

    y: array-like, compartments [central, peripf1, peripf2, effect]
    params: dictionary containing all required parameters
    """
    A1, A2, A3, A4 = y
    V1, V2, V3 = params['V1'], params['V2'], params['V3']
    CL, Q2, Q3 = params['CL'], params['Q2'], params['Q3']

    K10 = CL / V1
    K12 = Q2 / V1
    K21 = Q2 / V2
    K13 = Q3 / V1
    K31 = Q3 / V3
    K123 = K10 + K12 + K13
    KE0 = params['KE0']

    dA1 = A2 * K21 + A3 * K31 - A1 * K123
    dA2 = A1 * K12 - A2 * K21
    dA3 = A1 * K13 - A3 * K31
    dA4 = KE0 * (A1 / V1 - A4)

    return [dA1, dA2, dA3, dA4]


def pd_model_control(y, control_value, params):
    """
    Differential equations for the PD model.

    y: array-like, compartments [central, peripf1, peripf2, effect]
    params: dictionary containing all required parameters
    """
    A1, A2, A3, A4 = y
    V1, V2, V3 = params['V1'], params['V2'], params['V3']
    CL, Q2, Q3 = params['CL'], params['Q2'], params['Q3']

    K10 = CL / V1
    K12 = Q2 / V1
    K21 = Q2 / V2
    K13 = Q3 / V1
    K31 = Q3 / V3
    K123 = K10 + K12 + K13
    KE0 = params['KE0']

    dA1 = A2 * K21 + A3 * K31 - A1 * K123 + control_value
    dA2 = A1 * K12 - A2 * K21
    dA3 = A1 * K13 - A3 * K31
    dA4 = KE0 * (A1 / V1 - A4)

    return [dA1, dA2, dA3, dA4]

def calculate_ipred(A4, params):
    """
    Calculates the predicted BIS value using the effect compartment amount A4.
    """
    E50 = params['E50']
    GAM = params['GAM']
    GAM1 = params['GAM1']
    EMAX = params['EMAX']
    RESD = params['RESD']

    CPLA = max(A4, 0)
    WGAM = 1 / (1 + np.exp(-30 * (CPLA - E50)))
    GAM0 = WGAM * GAM + (1 - WGAM) * GAM1
    PEFF = (CPLA ** GAM0) / (CPLA ** GAM0 + E50 ** GAM0)
    IPRED = EMAX * (1 - PEFF)

    return IPRED


def simulate_bis(t, y0, params, noise_std=1.0, seed=None):
    """
    Simulates BIS values over time using a numerical solver.
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(lambda t, y: pd_model(t, y, params), [t[0], t[-1]], y0, t_eval=t, method='LSODA')
    if not sol.success:
        raise RuntimeError("ODE solver failed.")

    A4_vals = sol.y[3]
    ipreds = np.array([calculate_ipred(A4, params) for A4 in A4_vals])

    if seed is not None:
        np.random.seed(seed)
    errors = np.random.normal(loc=0, scale=params['RESD'] * noise_std, size=len(ipreds))
    obs = ipreds + errors

    return t, obs, ipreds