# 치과마취의 이해1 과제
# Yoon-Seo Nam, 2025-05-31

import numpy as np

def pk_parameters(AGE, WGT, HGT, M1F2, PMA, TECH, A1V2, THETA, ETA):
    # HT2: height squared in m^2
    HT2 = (HGT / 100.0) ** 2

    # Maturation factors (Al-Sallami equations)
    MATM = 0.88 + (1 - 0.88) / (1 + (AGE / 13.4) ** (-12.7))
    MATF = 1.11 + (1 - 1.11) / (1 + (AGE / 7.1) ** (-1.1))
    MATR = 0.88 + (1 - 0.88) / (1 + (35. / 13.4) ** (-12.7))

    FFMM = MATM * 42.92 * HT2 * WGT / (30.93 * HT2 + WGT)
    FFMF = MATF * 37.99 * HT2 * WGT / (35.98 * HT2 + WGT)
    FFMR = MATR * 42.92 * 1.7**2 * 70. / (30.93 * 1.7**2 + 70.)
    FFM = FFMM * (2 - M1F2) + FFMF * (M1F2 - 1)
    NFFM = FFM / FFMR

    # Maturation for CL
    PMW = PMA * 52.0
    PMR = (35. + 40. / 52.) * 52.
    ME50 = np.exp(THETA[7])
    MGAM = np.exp(THETA[8])
    MCL = (PMW**MGAM) / (PMW**MGAM + ME50**MGAM)
    RCL = (PMR**MGAM) / (PMR**MGAM + ME50**MGAM)
    DCL = MCL / RCL

    # Maturation for Q3
    PMEW = AGE * 52. + 40.
    PMER = 35. * 52. + 40.
    QE50 = np.exp(THETA[13])
    MQ3 = PMEW / (PMEW + QE50)
    RQ3 = PMER / (PMER + QE50)
    DQ3 = MQ3 / RQ3

    # Age adjustments
    KV1 = 1.0
    KV2 = np.exp(THETA[9] * (AGE - 35.))
    KV3 = np.exp(THETA[12] * AGE * (TECH - 1))
    KCL = np.exp(THETA[10] * AGE * (TECH - 1))
    KQ2 = 1.0
    KQ3 = 1.0

    # V1
    VV50 = np.exp(THETA[11])
    CV1 = WGT / (WGT + VV50)
    RV1 = 70. / (70. + VV50)
    M1 = (CV1 / RV1) * KV1
    VCV1 = (A1V2 - 1) * (1 - CV1)
    V1 = np.exp(THETA[0] + ETA[0]) * M1 * (1 + VCV1 * np.exp(THETA[16]))

    # V2
    M2 = (WGT / 70.0)**1 * KV2
    V2 = np.exp(THETA[1] + ETA[1]) * M2

    # V3
    M3 = NFFM * KV3
    V3 = np.exp(THETA[2] + ETA[2]) * M3

    # CL
    M4 = (WGT / 70.0) ** 0.75 * KCL * DCL
    base_theta_cl = (2 - M1F2) * THETA[3] + (M1F2 - 1) * THETA[14]
    CL = np.exp(base_theta_cl + ETA[3]) * M4

    # Q2
    RV2 = np.exp(THETA[1])
    M5 = (V2 / RV2) ** 0.75 * KQ2
    KM5 = 1 + np.exp(THETA[15]) * (1 - MQ3)
    Q2 = np.exp(THETA[4] + ETA[4] + (A1V2 - 1) * THETA[17]) * M5 * KM5

    # Q3
    RV3 = np.exp(THETA[2])
    M6 = (V3 / RV3) ** 0.75 * KQ3 * DQ3
    Q3 = np.exp(THETA[5] + ETA[5]) * M6

    # Return all parameters
    return {
        "V1": V1,
        "V2": V2,
        "V3": V3,
        "CL": CL,
        "Q2": Q2,
        "Q3": Q3,
        "FFM": FFM,
        "NFFM": NFFM
    }