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

psi_t1 = 0.29
C_a = 0.6* U_k
H = psi_t * U_k ** 2

z = H_tot/H
D_k = 60*U_k/(n*m.pi)
F_1 = D_k**2 * m.pi*0.75/4
d = 0.5
K_h = 0.99
nu_ad = 0.84

DATA = {}
r1_mid = []
z = 1
for i in range(int(z)):
    r1_mid = m.sqrt(0.5*(1+d**2))
    h_lop = D_k*(1 - d)*0.5
    C1_U_mid = U_k*r1_mid*(1 - tau - (psi_t/(2*r1_mid**2)))
    T1_mid = T_in - (C_a**2 - C1_U_mid**2)/2010
    P1 = 0.98 * P_in * (T1_mid/T_in)**3.5
    ro_1 = P1/(R*T1_mid)
    F_1 = G/(ro_1*C_a)
    D_k = m.sqrt(4*F_1/(m.pi * (1 - 0.5**2)))
    D_k = round(D_k)
    D_vt = d*D_k
    U_mid = r1_mid * U_k
    W1_mid = m.sqrt(C_a**2 + (U_mid-C1_U_mid)**2)
    a1_mid = 20.1 * m.sqrt(T1_mid)
    M1_mid = W1_mid/a1_mid
    H_t = psi_t*U_k**2
    L_z = H_t*K_h
    h_ad = L_z * nu_ad
    tau_ = 0.5
    r4_mid_new = r1_mid
    r4_mid = 0
    while abs(r4_mid_new - r4_mid) > 0.00001:
        r4_mid = r4_mid_new
        C4_U_mid = U_k*r4_mid*(1 - tau_ - psi_t1/(2* r4_mid**2))

        C_2 = C4_U_mid**2 - C1_U_mid**2
        T_4 = T1_mid + L_z/1004.5 - C_2/2010
        L_ad = h_ad - C_2/2

        pi_i = (1 + L_ad/(1004.5*T1_mid))**3.5
        P4 = pi_i*P1
        ro_4 = P4/(R*T_4)
        F4 = G/(ro_4*C_a)
        D_4 = m.sqrt(1 - 4 * F4 / (m.pi * (D_k**2)))
        r4_mid_new = m.sqrt((1 + D_4**2)/2)

    alpha_1 = m.degrees(m.atan((U_mid - C1_U_mid)/C_a))
    C2_U_mid = U_k*r4_mid*(1 - tau_ + psi_t1/(2* r4_mid**2))
    alpha_2 = m.degrees(m.atan((U_mid - C2_U_mid)/C_a))
    eps_rk_1 = alpha_1 - alpha_2

def sechenie (r, K1, K2):
    CC_1a = m.sqrt(phi**2 + 2*(1-tau_)*psi_t1* m.log(r/r1_mid) + 2*(1 - tau_)**2 * (r1_mid**2 - r**2))
    CC_2a = m.sqrt(phi ** 2 - 2 * (1 - tau_) * psi_t1 * m.log(r / r1_mid) + 2 * (1 - tau_) ** 2 * (r1_mid ** 2 - r ** 2))
    C_11a = C_a + K1 * (CC_2a - CC_1a)
    C_22a = C_a + K2 * (CC_2a - CC_1a)
    phi_1a = C_11a/U_k
    phi_2a = C_22a/U_k

    alpha_1sech = m.degrees(m.atan(1/phi_1a * (r*tau_ + psi_t/2*r)))
    alpha_2sech = m.degrees(m.atan(1 / phi_2a * (r * tau_ + psi_t / 2 * r)))
    alpha_3 = m.atan(r/phi_2a)

DATA['r1_mid'] = r1_mid


stage(G,H,psi_t, pi_c, R)