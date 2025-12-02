import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def disp(k, k0, n, ng):
    '''
    Calculates the material wavenumber K for a given vacuum wavenumber k

    Parameters:
    k: vacuum wavenumber
    k0: reference vacuum wavenumber
    n: medium refractive index
    ng: medium group index

    Returns:
    K: material wavenumber
    '''
    return n*k0 + ng*(k - k0)

@njit
def calculate_integral_exp(Dk, lower_bd, upper_bd):
    '''
    Calculates the numerical value of the integral
    I = int_{lower_bd}^{upper_bd} dz e^{i Dk z}

    Parameters:
    Dk: phase mismatch
    lower_bd: integration lower bound
    upper_bd: integration upper bound

    Returns:
    I: value of the integral
    '''
    I = (
        (upper_bd - lower_bd) *
        np.exp(0.5j*Dk*(upper_bd + lower_bd)) *
        np.sinc(0.5*Dk*(upper_bd - lower_bd)/np.pi)
    )
    return I

@njit
def calculate_integral_ppln(L, Dk, Lamb):
    '''
    Calculates the integral I for a periodically poled material,
    I = int_{-L/2}^{L/2} dz g(z) e^{i Dk z},
    where g(z) = +/- 1 flips sign every half-period, Lamb/2.

    Parameters:
    L: length of nonlinear material
    Dk: phase-mismatch
    Lamb: poling period

    Returns:
    I: value of the integral
    '''
    N_layers = int(2 * L // Lamb)
    I = 0
    for i in range(N_layers):
        lower_bd = -0.5*L + i*0.5*Lamb
        upper_bd = -0.5*L + (i+1)*0.5*Lamb
        I += (-1)**i * calculate_integral_exp(Dk, lower_bd, upper_bd)

    lower_bd = -0.5*L + N_layers*0.5*Lamb
    I += (-1)**N_layers * calculate_integral_exp(Dk, lower_bd, L/2)
    return I

@njit
def calculate_transmission(TM):
    '''
    Calculates the transmission coefficient given a 2x2 transfer matrix.

    Parameters:
    TM: 2x2 transfer matrix

    Returns:
    T: transmission coefficient
    '''
    T = np.abs( (TM[0,0]*TM[1,1] - TM[0,1]*TM[1,0])/TM[1,1] )**2
    return T

@njit
def asy_ampli_mat(M1, M2):
    '''
    Calculates the asymptotic amplitudes inside a Fabry-Perot cavity, given the
    transfer matrices for the mirrors

    Parameters:
    M1: 2x2 transfer matrix for mirror 1
    M2: 2x2 transfer matrix for mirror 2

    Returns:
    ampli_list: list of the asymptotic amplitudes
    '''
    e1_in_L = (
        (M1[0,0]*M1[1,1]*M2[1,1] - M1[0,1]*M1[1,0]*M2[1,1]) /
        (M1[0,1]*M2[1,0] + M1[1,1]*M2[1,1])
    )
    e2_in_L = (
        (- M1[0,0]*M1[1,1]*M2[1,0] + M1[0,1]*M1[1,0]*M2[1,0]) /
        (M1[0,1]*M2[1,0] + M1[1,1]*M2[1,1])
    )
    e1_out_R = (
        M1[0,0] /
        (M1[0,0]*M2[0,0] + M1[1,0]*M2[0,1])
    )
    e2_out_R = (
        M1[1,0] /
        (M1[0,0]*M2[0,0] + M1[1,0]*M2[0,1])
    )
    e1_out_L = (
        (- M1[0,0]*M1[1,1]*M2[0,1] + M1[0,1]*M1[1,0]*M2[0,1]) /
        (M1[0,0]*M2[0,0] + M1[1,0]*M2[0,1])
    )
    e2_out_L = (
        (M1[0,0]*M1[1,1]*M2[0,0] - M1[0,1]*M1[1,0]*M2[0,0]) /
        (M1[0,0]*M2[0,0] + M1[1,0]*M2[0,1])
    )
    ampli_list = [e1_in_L, e2_in_L, e1_out_R, e2_out_R, e1_out_L, e2_out_L]
    return ampli_list

@njit
def transfer_matrix(k, a, r):
    '''
    Calculates the transfer matrix for a mirror with reflection coefficient r,
    located at position z=a, and for a wave of wavenumber k.

    Parameters:
    k: wavenumber
    a: position of the mirror
    r: reflection amplitude

    Returns:
    T: 2x2 transfer matrix
    '''
    t = np.sqrt(1 - r**2)
    T = np.array([
        [1/t, r*np.exp(-2j*k*a)/t],
        [r*np.exp(2j*k*a)/t, 1/t]
    ], dtype=np.complex128)
    return T

@njit
def tm_fresnel(a, k1, k2):
    '''
    Calculates the transfer matrix for the displacement field of an interface
    between two media, 1 and 2, located at z=a
    
    Parameters:
    a: position of the interface
    k1: wavenumber at medium 1
    k2: wavenumber at medium 2

    Returns:
    T: 2x2 transfer matrix
    '''
    rho = k1 / k2
    
    W = 0.5 * np.exp(1j*(k1-k2)*a) * (1 + rho) / rho**2
    Z = 0.5 * np.exp(-1j*(k1+k2)*a) * (1 - rho) / rho**2
    
    T = np.array([
        [W, Z],
        [Z.conjugate(), W.conjugate()]
    ])
    
    return T

@njit
def tm_stack(z0, k1, k2, Lamb, N_layers):
    '''
    Calculates the transfer matrix for the displacement field of a Bragg
    reflector

    Parameters:
    z0: start position of the Bragg grating
    k1: wavenumber at medium 1
    k2: wavenumber at medium 2
    Lamb: Bragg grating period
    N_layers: number of medium 2 layers
    '''
    T = np.eye(2, dtype=np.complex128)

    for i in range(N_layers):
        T = tm_fresnel(z0 + i*Lamb, k1, k2) @ T
        T = tm_fresnel(z0 + (i+0.5)*Lamb, k2, k1) @ T

    return T

@njit
def calculate_J(k1, kp, L, n, ng, r1, r2):
    '''
    Calculates the spectral amplitude of the signal photon for SPDC in a Fabry-
    Perot cavity with flat-response mirrors.

    Parameters:
    k1: signal photon wavenumber
    kp: pump wavenumber
    L: cavity length
    n: list with refractive indices for the pump, signal and idler central
    wavenumbers, [n_p, n_s, n_i]
    ng: list with group indices for the pump, signal and idler central
    wavenumbers, [ng_p, ng_s, ng_i]
    r1: reflection amplitude for mirror 1
    r2: reflection amplitude for mirror 2

    Returns:
    J: spectral amplitudes for signal wavenumber k1 for different output channel
    configurations
        J[0]: signal exits at channel R, idler exits at R
        J[1]: signal exits at channel L, idler exits at L
        J[2]: signal exits at channel R, idler exits at L
        J[3]: signal exits at channel L, idler exits at R
    '''
    Kp = disp(kp, kp, n[0], ng[0])
    
    TM_L = transfer_matrix(Kp, -L/2, r1)
    TM_R = transfer_matrix(Kp, L/2, r2)
    e1_in_L_p, e2_in_L_p, e1_out_R_p, e2_out_R_p, e1_out_L_p, e2_out_L_p = \
    asy_ampli_mat(TM_L, TM_R)

    J = np.zeros((4, k1.size), dtype=np.complex128)
    for i in range(k1.size):
        K1 = disp(k1[i], kp/2, n[1], ng[1])
        k2 = kp - k1[i]
        K2 = disp(k2, kp/2, n[2], ng[2])
        Dk = Kp - K1 - K2
        
        TM_L = transfer_matrix(K1, -L/2, r1)
        TM_R = transfer_matrix(K1, L/2, r2)
        e1_in_L_1, e2_in_L_1, e1_out_R_1, e2_out_R_1, e1_out_L_1, e2_out_L_1 = \
        asy_ampli_mat(TM_L, TM_R)
        
        TM_L = transfer_matrix(K2, -L/2, r1)
        TM_R = transfer_matrix(K2, L/2, r2)
        e1_in_L_2, e2_in_L_2, e1_out_R_2, e2_out_R_2, e1_out_L_2, e2_out_L_2 = \
        asy_ampli_mat(TM_L, TM_R)
        
        I_RR = (
            e1_in_L_p*e1_out_R_1.conjugate()*e1_out_R_2.conjugate() +
            e2_in_L_p*e2_out_R_1.conjugate()*e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[0, i] = np.sqrt(kp*k1[i]*k2) * I_RR

        I_LL = (
            e1_in_L_p*e1_out_L_1.conjugate()*e1_out_L_2.conjugate() +
            e2_in_L_p*e2_out_L_1.conjugate()*e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[1, i] = np.sqrt(kp*k1[i]*k2) * I_LL

        I_RL = (
            e1_in_L_p*e1_out_R_1.conjugate()*e1_out_L_2.conjugate() +
            e2_in_L_p*e2_out_R_1.conjugate()*e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[2, i] = np.sqrt(kp*k1[i]*k2) * I_RL

        I_LR = (
            e1_in_L_p*e1_out_L_1.conjugate()*e1_out_R_2.conjugate() +
            e2_in_L_p*e2_out_L_1.conjugate()*e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[3, i] = np.sqrt(kp*k1[i]*k2) * I_LR

    return J

@njit
def calculate_J_counter(k1, kp, L, n, ng, r1, r2):
    '''
    Calculates the spectral amplitude of the signal photon for SPDC in a Fabry-
    Perot cavity with flat-response mirrors considering the possibility of
    generating counter-propagating photons

    Parameters:
    k1: signal photon wavenumber
    kp: pump wavenumber
    L: cavity length
    n: list with refractive indices for the pump, signal and idler central
    wavenumbers, [n_p, n_s, n_i]
    ng: list with group indices for the pump, signal and idler central
    wavenumbers, [ng_p, ng_s, ng_i]
    r1: reflection amplitude for mirror 1
    r2: reflection amplitude for mirror 2

    Returns:
    J: spectral amplitudes for signal wavenumber k1 for different output channel
    configurations
        J[0]: signal exits at channel R, idler exits at R
        J[1]: signal exits at channel L, idler exits at L
        J[2]: signal exits at channel R, idler exits at L
        J[3]: signal exits at channel L, idler exits at R
    '''
    Kp = disp(kp, kp, n[0], ng[0])

    TM_L = transfer_matrix(Kp, -L/2, r1)
    TM_R = transfer_matrix(Kp, L/2, r2)
    e1_in_L_p, e2_in_L_p, e1_out_R_p, e2_out_R_p, e1_out_L_p, e2_out_L_p = \
    asy_ampli_mat(TM_L, TM_R)

    J = np.zeros((4, k1.size), dtype=np.complex128)
    for i in range(k1.size):
        K1 = disp(k1[i], kp / 2, n[1], ng[1])
        k2 = kp - k1[i]
        K2 = disp(k2, kp / 2, n[2], ng[2])
        
        Dk_0 = Kp - K1 - K2
        Dk_1 = Kp + K1 - K2
        Dk_2 = Kp - K1 + K2
        Dk_3 = Kp + K1 + K2

        TM_L = transfer_matrix(K1, -L/2, r1)
        TM_R = transfer_matrix(K1, L/2, r2)
        e1_in_L_1, e2_in_L_1, e1_out_R_1, e2_out_R_1, e1_out_L_1, e2_out_L_1 = \
        asy_ampli_mat(TM_L, TM_R)
        
        TM_L = transfer_matrix(K2, -L/2, r1)
        TM_R = transfer_matrix(K2, L/2, r2)
        e1_in_L_2, e2_in_L_2, e1_out_R_2, e2_out_R_2, e1_out_L_2, e2_out_L_2 = \
        asy_ampli_mat(TM_L, TM_R)

        I_RR_0 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e2_out_R_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_0 * L / np.pi)

        I_RR_1 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e1_out_R_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_1 * L / np.pi)

        I_RR_2 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e2_out_R_2.conjugate() +
            e2_in_L_p * e2_out_R_1.conjugate() * e1_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_2 * L / np.pi)
        
        I_RR_3 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e2_out_R_2.conjugate() +
            e2_in_L_p * e1_out_R_1.conjugate() * e1_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_3 * L / np.pi)

        J[0, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_RR_0 + I_RR_1 + I_RR_2 + I_RR_3)
        )
        
        I_LL_0 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e2_out_L_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_0 * L / np.pi)

        I_LL_1 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e1_out_L_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_1 * L / np.pi)

        I_LL_2 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e2_out_L_2.conjugate() +
            e2_in_L_p * e2_out_L_1.conjugate() * e1_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_2 * L / np.pi)

        I_LL_3 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e2_out_L_2.conjugate() +
            e2_in_L_p * e1_out_L_1.conjugate() * e1_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_3 * L / np.pi)

        J[1, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_LL_0 + I_LL_1 + I_LL_2 + I_LL_3)
        )
        
        I_RL_0 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e2_out_R_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_0 * L / np.pi)

        I_RL_1 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e1_out_R_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_1 * L / np.pi)

        I_RL_2 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e2_out_L_2.conjugate() +
            e2_in_L_p * e2_out_R_1.conjugate() * e1_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_2 * L / np.pi)

        I_RL_3 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e2_out_L_2.conjugate() +
            e2_in_L_p * e1_out_R_1.conjugate() * e1_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_3 * L / np.pi)

        J[2, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_RL_0 + I_RL_1 + I_RL_2 + I_RL_3)
        )
        
        I_LR_0 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e2_out_L_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_0 * L / np.pi)

        I_LR_1 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e1_out_L_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_1 * L / np.pi)

        I_LR_2 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e2_out_R_2.conjugate() +
            e2_in_L_p * e2_out_L_1.conjugate() * e1_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_2 * L / np.pi)

        I_LR_3 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e2_out_R_2.conjugate() +
            e2_in_L_p * e1_out_L_1.conjugate() * e1_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk_3 * L / np.pi)

        J[3, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_LR_0 + I_LR_1 + I_LR_2 + I_LR_3)
        )
    
    return J

@njit
def calculate_J_bg(k1, kp, L, n, ng, bragg_params):
    '''
    Calculates the spectral amplitude of the signal photon for SPDC in a Fabry-
    Perot cavity with Bragg reflectors

    Parameters:
    k1: signal photon wavenumber
    kp: pump wavenumber
    L: cavity length
    n: list with refractive indices for the pump, signal and idler central
    wavenumbers, [n_p, n_s, n_i]
    ng: list with group indices for the pump, signal and idler central
    wavenumbers, [ng_p, ng_s, ng_i]
    bragg_params: list [n1, n2, Lamb, N]
        n1: refractive index of medium 1
        n2: refractive index of medium 2
        Lamb: Bragg grating period
        N: number of medium 2 layers

    Returns:
    J: spectral amplitudes for signal wavenumber k1 for different output channel
    configurations
        J[0]: signal exits at channel R, idler exits at R
        J[1]: signal exits at channel L, idler exits at L
        J[2]: signal exits at channel R, idler exits at L
        J[3]: signal exits at channel L, idler exits at R
    '''
    n1, n2, Lamb, N = bragg_params
    Kp = disp(kp, kp, n[0], ng[0])
    z0_L = - 0.5 * (L + (2*N - 1) * Lamb)
    z0_R = 0.5 * L
    TM_L = tm_stack(z0_L, n1*kp, n2*kp, Lamb, N)
    TM_R = tm_stack(z0_R, n1*kp, n2*kp, Lamb, N)
    e1_in_L_p, e2_in_L_p, e1_out_R_p, e2_out_R_p, e1_out_L_p, e2_out_L_p = \
    asy_ampli_mat(TM_L, TM_R)

    J = np.zeros((4, k1.size), dtype=np.complex128)
    for i in range(k1.size):
        K1 = disp(k1[i], kp/2, n[1], ng[1])
        k2 = kp - k1[i]
        K2 = disp(k2, kp/2, n[2], ng[2])
        Dk = Kp - K1 - K2
        
        TM_L = tm_stack(z0_L, n1*k1[i], n2*k1[i], Lamb, N)
        TM_R = tm_stack(z0_R, n1*k1[i], n2*k1[i], Lamb, N)
        e1_in_L_1, e2_in_L_1, e1_out_R_1, e2_out_R_1, e1_out_L_1, e2_out_L_1 = \
        asy_ampli_mat(TM_L, TM_R)
        
        TM_L = tm_stack(z0_L, n1*k2, n2*k2, Lamb, N)
        TM_R = tm_stack(z0_R, n1*k2, n2*k2, Lamb, N)
        e1_in_L_2, e2_in_L_2, e1_out_R_2, e2_out_R_2, e1_out_L_2, e2_out_L_2 = \
        asy_ampli_mat(TM_L, TM_R)
        
        I_RR = (
            e1_in_L_p*e1_out_R_1.conjugate()*e1_out_R_2.conjugate() +
            e2_in_L_p*e2_out_R_1.conjugate()*e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[0, i] = np.sqrt(kp*k1[i]*k2) * I_RR

        I_LL = (
            e1_in_L_p*e1_out_L_1.conjugate()*e1_out_L_2.conjugate() +
            e2_in_L_p*e2_out_L_1.conjugate()*e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[1, i] = np.sqrt(kp*k1[i]*k2) * I_LL

        I_RL = (
            e1_in_L_p*e1_out_R_1.conjugate()*e1_out_L_2.conjugate() +
            e2_in_L_p*e2_out_R_1.conjugate()*e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[2, i] = np.sqrt(kp*k1[i]*k2) * I_RL

        I_LR = (
            e1_in_L_p*e1_out_L_1.conjugate()*e1_out_R_2.conjugate() +
            e2_in_L_p*e2_out_L_1.conjugate()*e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[3, i] = np.sqrt(kp*k1[i]*k2) * I_LR

    return J

@njit
def calculate_J_ppln(k1, kp, L, n, ng, r1, r2, Lamb):
    '''
    Calculates the spectral amplitude of the signal photon for SPDC in a Fabry-
    Perot cavity with flat-response mirrors and a periodically-poled material

    Parameters:
    k1: signal photon wavenumber
    kp: pump wavenumber
    L: cavity length
    n: list with refractive indices for the pump, signal and idler central
    wavenumbers, [n_p, n_s, n_i]
    ng: list with group indices for the pump, signal and idler central
    wavenumbers, [ng_p, ng_s, ng_i]
    r1: reflection amplitude for mirror 1
    r2: reflection amplitude for mirror 2
    Lamb: poling period

    Returns:
    J: spectral amplitudes for signal wavenumber k1 for different output channel
    configurations
        J[0]: signal exits at channel R, idler exits at R
        J[1]: signal exits at channel L, idler exits at L
        J[2]: signal exits at channel R, idler exits at L
        J[3]: signal exits at channel L, idler exits at R
    '''
    Kp = disp(kp, kp, n[0], ng[0])
    TM_L = transfer_matrix(Kp, -L/2, r1)
    TM_R = transfer_matrix(Kp, L/2, r2)
    e1_in_L_p, e2_in_L_p, e1_out_R_p, e2_out_R_p, e1_out_L_p, e2_out_L_p = \
    asy_ampli_mat(TM_L, TM_R)

    J = np.zeros((4, k1.size), dtype=np.complex128)
    for i in range(k1.size):
        K1 = disp(k1[i], kp/2, n[1], ng[1])
        k2 = kp - k1[i]
        K2 = disp(k2, kp/2, n[2], ng[2])
        Dk = Kp - K1 - K2
        
        TM_L = transfer_matrix(K1, -L/2, r1)
        TM_R = transfer_matrix(K1, L/2, r2)
        e1_in_L_1, e2_in_L_1, e1_out_R_1, e2_out_R_1, e1_out_L_1, e2_out_L_1 = \
        asy_ampli_mat(TM_L, TM_R)
        
        TM_L = transfer_matrix(K2, -L/2, r1)
        TM_R = transfer_matrix(K2, L/2, r2)
        e1_in_L_2, e2_in_L_2, e1_out_R_2, e2_out_R_2, e1_out_L_2, e2_out_L_2 = \
        asy_ampli_mat(TM_L, TM_R)

        I_RR = (
            e1_in_L_p * e1_out_R_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk, Lamb) +
            e2_in_L_p * e2_out_R_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk, Lamb)
        )
        J[0, i] = np.sqrt(kp*k1[i]*k2) * I_RR

        I_LL = (
            e1_in_L_p * e1_out_L_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk, Lamb) +
            e2_in_L_p * e2_out_L_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk, Lamb)
        )
        J[1, i] = np.sqrt(kp*k1[i]*k2) * I_LL

        I_RL = (
            e1_in_L_p * e1_out_R_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk, Lamb) +
            e2_in_L_p * e2_out_R_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk, Lamb)
        )
        J[2, i] = np.sqrt(kp*k1[i]*k2) * I_RL

        I_LR = (
            e1_in_L_p * e1_out_L_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk, Lamb) +
            e2_in_L_p * e2_out_L_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk, Lamb)
        )
        J[3, i] = np.sqrt(kp*k1[i]*k2) * I_LR

    return J

@njit
def calculate_J_ppln_counter(k1, kp, L, n, ng, r1, r2, Lamb):
    '''
    Calculates the spectral amplitude of the signal photon for SPDC in a Fabry-
    Perot cavity with flat-response mirrors and a periodically-poled material
    considering the possibility of generating counter-propagating photons

    Parameters:
    k1: signal photon wavenumber
    kp: pump wavenumber
    L: cavity length
    n: list with refractive indices for the pump, signal and idler central
    wavenumbers, [n_p, n_s, n_i]
    ng: list with group indices for the pump, signal and idler central
    wavenumbers, [ng_p, ng_s, ng_i]
    r1: reflection amplitude for mirror 1
    r2: reflection amplitude for mirror 2
    Lamb: poling period

    Returns:
    J: spectral amplitudes for signal wavenumber k1 for different output channel
    configurations
        J[0]: signal exits at channel R, idler exits at R
        J[1]: signal exits at channel L, idler exits at L
        J[2]: signal exits at channel R, idler exits at L
        J[3]: signal exits at channel L, idler exits at R
    '''
    Kp = disp(kp, kp, n[0], ng[0])

    TM_L = transfer_matrix(Kp, -L/2, r1)
    TM_R = transfer_matrix(Kp, L/2, r2)
    e1_in_L_p, e2_in_L_p, e1_out_R_p, e2_out_R_p, e1_out_L_p, e2_out_L_p = \
    asy_ampli_mat(TM_L, TM_R)

    J = np.zeros((4, k1.size), dtype=np.complex128)
    for i in range(k1.size):
        K1 = disp(k1[i], kp / 2, n[1], ng[1])
        k2 = kp - k1[i]
        K2 = disp(k2, kp / 2, n[2], ng[2])
        
        Dk_0 = Kp - K1 - K2
        Dk_1 = Kp + K1 - K2
        Dk_2 = Kp - K1 + K2
        Dk_3 = Kp + K1 + K2

        TM_L = transfer_matrix(K1, -L/2, r1)
        TM_R = transfer_matrix(K1, L/2, r2)
        e1_in_L_1, e2_in_L_1, e1_out_R_1, e2_out_R_1, e1_out_L_1, e2_out_L_1 = \
        asy_ampli_mat(TM_L, TM_R)
        
        TM_L = transfer_matrix(K2, -L/2, r1)
        TM_R = transfer_matrix(K2, L/2, r2)
        e1_in_L_2, e2_in_L_2, e1_out_R_2, e2_out_R_2, e1_out_L_2, e2_out_L_2 = \
        asy_ampli_mat(TM_L, TM_R)

        I_RR_0 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_0, Lamb) +
            e2_in_L_p * e2_out_R_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_0, Lamb)
        )

        I_RR_1 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_1, Lamb) +
            e2_in_L_p * e1_out_R_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_1, Lamb)
        )

        I_RR_2 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_2, Lamb) +
            e2_in_L_p * e2_out_R_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_2, Lamb)
        )
        
        I_RR_3 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_3, Lamb) +
            e2_in_L_p * e1_out_R_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_3, Lamb)
        )

        J[0, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_RR_0 + I_RR_1 + I_RR_2 + I_RR_3)
        )
        
        I_LL_0 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_0, Lamb) +
            e2_in_L_p * e2_out_L_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_0, Lamb)
        )

        I_LL_1 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_1, Lamb) +
            e2_in_L_p * e1_out_L_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_1, Lamb)
        )

        I_LL_2 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_2, Lamb) +
            e2_in_L_p * e2_out_L_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_2, Lamb)
        )

        I_LL_3 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_3, Lamb) +
            e2_in_L_p * e1_out_L_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_3, Lamb)
        )

        J[1, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_LL_0 + I_LL_1 + I_LL_2 + I_LL_3)
        )
        
        I_RL_0 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_0, Lamb) +
            e2_in_L_p * e2_out_R_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_0, Lamb)
        )

        I_RL_1 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_1, Lamb) +
            e2_in_L_p * e1_out_R_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_1, Lamb)
        )

        I_RL_2 = (
            e1_in_L_p * e1_out_R_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_2, Lamb) +
            e2_in_L_p * e2_out_R_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_2, Lamb)
        )

        I_RL_3 = (
            e1_in_L_p * e2_out_R_1.conjugate() * e2_out_L_2.conjugate() *
            calculate_integral_ppln(L, Dk_3, Lamb) +
            e2_in_L_p * e1_out_R_1.conjugate() * e1_out_L_2.conjugate() *
            calculate_integral_ppln(L, -Dk_3, Lamb)
        )

        J[2, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_RL_0 + I_RL_1 + I_RL_2 + I_RL_3)
        )
        
        I_LR_0 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_0, Lamb) +
            e2_in_L_p * e2_out_L_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_0, Lamb)
        )

        I_LR_1 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_1, Lamb) +
            e2_in_L_p * e1_out_L_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_1, Lamb)
        )

        I_LR_2 = (
            e1_in_L_p * e1_out_L_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_2, Lamb) +
            e2_in_L_p * e2_out_L_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_2, Lamb)
        )

        I_LR_3 = (
            e1_in_L_p * e2_out_L_1.conjugate() * e2_out_R_2.conjugate() *
            calculate_integral_ppln(L, Dk_3, Lamb) +
            e2_in_L_p * e1_out_L_1.conjugate() * e1_out_R_2.conjugate() *
            calculate_integral_ppln(L, -Dk_3, Lamb)
        )

        J[3, i] = (
            np.sqrt( kp * k1[i] * k2 ) *
            (I_LR_0 + I_LR_1 + I_LR_2 + I_LR_3)
        )
    
    return J

@njit
def calculate_J_sfwm(k1, kp, L, n, ng, r1, r2):
    '''
    Calculates the spectral amplitude of the signal photon for SFWM in a Fabry-
    Perot cavity with flat-response mirrors.

    Parameters:
    k1: signal photon wavenumber
    kp: pump wavenumber
    L: cavity length
    n: list with refractive indices for the pump, signal and idler central
    wavenumbers, [n_p, n_s, n_i]
    ng: list with group indices for the pump, signal and idler central
    wavenumbers, [ng_p, ng_s, ng_i]
    r1: reflection amplitude for mirror 1
    r2: reflection amplitude for mirror 2

    Returns:
    J: spectral amplitudes for signal wavenumber k1 for different output channel
    configurations
        J[0]: signal exits at channel R, idler exits at R
        J[1]: signal exits at channel L, idler exits at L
        J[2]: signal exits at channel R, idler exits at L
        J[3]: signal exits at channel L, idler exits at R
    '''
    Kp = disp(kp, kp, n[0], ng[0])
    
    TM_L = transfer_matrix(Kp, -L/2, r1)
    TM_R = transfer_matrix(Kp, L/2, r2)
    e1_in_L_p, e2_in_L_p, e1_out_R_p, e2_out_R_p, e1_out_L_p, e2_out_L_p = \
    asy_ampli_mat(TM_L, TM_R)

    J = np.zeros((4, k1.size), dtype=np.complex128)
    for i in range(k1.size):
        K1 = disp(k1[i], kp/2, n[1], ng[1])
        k2 = 2*kp - k1[i]
        K2 = disp(k2, 3*kp/2, n[2], ng[2])
        Dk = 2*Kp - K1 - K2
        
        TM_L = transfer_matrix(K1, -L/2, r1)
        TM_R = transfer_matrix(K1, L/2, r2)
        e1_in_L_1, e2_in_L_1, e1_out_R_1, e2_out_R_1, e1_out_L_1, e2_out_L_1 = \
        asy_ampli_mat(TM_L, TM_R)
        
        TM_L = transfer_matrix(K2, -L/2, r1)
        TM_R = transfer_matrix(K2, L/2, r2)
        e1_in_L_2, e2_in_L_2, e1_out_R_2, e2_out_R_2, e1_out_L_2, e2_out_L_2 = \
        asy_ampli_mat(TM_L, TM_R)
        
        I_RR = (
            e1_in_L_p * e1_in_L_p * e1_out_R_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_R_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[0, i] = np.sqrt(kp * kp * k1[i] * k2) * I_RR

        I_LL = (
            e1_in_L_p * e1_in_L_p * e1_out_L_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_L_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[1, i] = np.sqrt(kp * kp * k1[i] * k2) * I_LL

        I_RL = (
            e1_in_L_p * e1_in_L_p * e1_out_R_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_R_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[2, i] = np.sqrt(kp * kp * k1[i] * k2) * I_RL

        I_LR = (
            e1_in_L_p * e1_in_L_p * e1_out_L_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_L_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5*Dk*L/np.pi)
        J[3, i] = np.sqrt(kp * kp * k1[i] * k2) * I_LR

    return J

@njit
def calculate_J_sfwm_bg(k1, kp, L, n, ng, bragg_params):
    '''
    Calculates the spectral amplitude of the signal photon for SFWM in a Fabry-
    Perot cavity with Bragg reflectors

    Parameters:
    k1: signal photon wavenumber
    kp: pump wavenumber
    L: cavity length
    n: list with refractive indices for the pump, signal and idler central
    wavenumbers, [n_p, n_s, n_i]
    ng: list with group indices for the pump, signal and idler central
    wavenumbers, [ng_p, ng_s, ng_i]
    bragg_params: list [n1, n2, Lamb, N]
        n1: refractive index of medium 1
        n2: refractive index of medium 2
        Lamb: Bragg grating period
        N: number of medium 2 layers

    Returns:
    J: spectral amplitudes for signal wavenumber k1 for different output channel
    configurations
        J[0]: signal exits at channel R, idler exits at R
        J[1]: signal exits at channel L, idler exits at L
        J[2]: signal exits at channel R, idler exits at L
        J[3]: signal exits at channel L, idler exits at R
    '''
    n1, n2, Lamb, N = bragg_params
    Kp = disp(kp, kp, n[0], ng[0])

    z0_L = - 0.5 * (L + (2*N - 1) * Lamb)
    z0_R = 0.5 * L
    TM_L = tm_stack(z0_L, n1*kp, n2*kp, Lamb, N)
    TM_R = tm_stack(z0_R, n1*kp, n2*kp, Lamb, N)
    e1_in_L_p, e2_in_L_p, e1_out_R_p, e2_out_R_p, e1_out_L_p, e2_out_L_p = \
    asy_ampli_mat(TM_L, TM_R)

    J = np.zeros((4, k1.size), dtype=np.complex128)
    for i in range(k1.size):
        K1 = disp(k1[i], kp/2, n[1], ng[1])
        k2 = 2*kp - k1[i]
        K2 = disp(k2, 3*kp/2, n[2], ng[2])
        
        Dk = 2*Kp - K1 - K2

        TM_L = tm_stack(z0_L, n1*k1[i], n2*k1[i], Lamb, N)
        TM_R = tm_stack(z0_R, n1*k1[i], n2*k1[i], Lamb, N)
        e1_in_L_1, e2_in_L_1, e1_out_R_1, e2_out_R_1, e1_out_L_1, e2_out_L_1 = \
        asy_ampli_mat(TM_L, TM_R)
        
        TM_L = tm_stack(z0_L, n1*k2, n2*k2, Lamb, N)
        TM_R = tm_stack(z0_R, n1*k2, n2*k2, Lamb, N)
        e1_in_L_2, e2_in_L_2, e1_out_R_2, e2_out_R_2, e1_out_L_2, e2_out_L_2 = \
        asy_ampli_mat(TM_L, TM_R)

        Iz_RR = (
            e1_in_L_p * e1_in_L_p * e1_out_R_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_R_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk * L / np.pi)

        J[0, i] = np.sqrt(kp * kp * k1[i] * k2) * Iz_RR

        Iz_LL = (
            e1_in_L_p * e1_in_L_p * e1_out_L_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_L_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk * L / np.pi)

        J[1, i] = np.sqrt(kp * kp * k1[i] * k2) * Iz_LL
        
        Iz_RL = (
            e1_in_L_p * e1_in_L_p * e1_out_R_1.conjugate() * e1_out_L_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_R_1.conjugate() * e2_out_L_2.conjugate()
        ) * L * np.sinc(0.5 * Dk * L / np.pi)

        J[2, i] = np.sqrt(kp * kp * k1[i] * k2) * Iz_RL

        Iz_LR = (
            e1_in_L_p * e1_in_L_p * e1_out_L_1.conjugate() * e1_out_R_2.conjugate() +
            e2_in_L_p * e2_in_L_p * e2_out_L_1.conjugate() * e2_out_R_2.conjugate()
        ) * L * np.sinc(0.5 * Dk * L / np.pi)

        J[3, i] = np.sqrt(kp * kp * k1[i] * k2) * Iz_LR
    
    return J