import math as m


def stage (G, H, psi_t):

    tau = 0.5 #Степень реактивности
    U_k = m.sqrt(H/psi_t)
    #U_k = 250 m/s
    U_k = 250
    H = psi_t * U_k**2
    H_tot = 3.5 * R * T_in * (pi**0.286 - 1)


    pass

G = 100 #kg/s
P_in = 100000#Pa
P_out = 500000#Pa
H = 1.375 * 10**4
psi_t = 0.25
stage(G,H,psi_t)
R = 287
T_in = 303 #K
