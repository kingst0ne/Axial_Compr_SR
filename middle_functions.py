import math as m
import CoolProp as cp

def stage (G, H, psi_t, pi_c, R):


    pass


n = 5200
G = 100 #kg/s
P_in = 100000#Pa
P_out = 500000#Pa
H = 1.375 * 10**4
psi_t = 0.25

R = 287
T_in = 303 #K
pi_c = P_out/P_in

tau = 0.5  # Степень реактивности

H_tot = 3.5 * R * T_in * (pi_c ** 0.286 - 1)

phi = 0.55
ro = cp.CoolProp.PropsSI('D', 'T', T_in, 'P', P_in, 'Air')
V = G/ro

U_k = (V * m.pi * n**2 / (phi * 900 * 0.75))**(1/3)
U_k = round(U_k)

H = psi_t * U_k ** 2
stage(G,H,psi_t, pi_c, R)