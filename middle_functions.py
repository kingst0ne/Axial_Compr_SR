import math as m
import CoolProp as cp

#TODO! основная идея которую я предлагаю проста:
#1) ниже ты описываешь класс gasdynamic, который рассчитывает все параметры для одной ступени.
#2) для расчета всего компрессора тебе нужно будет создать несколько экземпляров этого класса (по количеству ступеней).
#3) для первой ступени ты задаешь все исходные данные касающиеся общих параметров компрессора (например обороты) и
# параметры касающиеся конкретно этой ступени (например параметыр на входе, какой-нибудь коэффициен напора...)
#4) TODO! далее тебе нужно будет в самом конце реализвать метод (назовем его к примеру def get_results(self)),
# этот метод должен будет возвращать все данные, используемые в качестве исходных данных для последующей ступени
#5) т.е. примерно(!) это будет так:
    # stage1 = gasdynamic()
    # stage1.set_input_data(исходные данные)
    # stage1_results=stage1.get_results()
    # stage2 = gasdynamic()
    # stage2.set_input_data(stage1_results)
    # stage2_results=stage2.get_results()
    # и т.д.
    # и дальше отдельно нужно будет подумать над обработкой общих результатов расчета всего компрессора

def b_t(stage): #TODO! эту функцию надо будет потом удалить - там внизу кода увидешь, я написал про это. Все верно про костыль - так делать не стоит
    #костыль костылевый
    b = [0.96, 1, 1.15, 1.25, 1.25, 1.35, 1.35, 1.55, 1.2, 1.2]
    return b[stage-1]

def h_b(stage): #TODO! эта функция нигде не использована
    h = [2.5, 2.3, 2.1, 1.95, 1.8, 1.6, 1.5, 1.5, 1.5, 1.5]
    return h[stage-1]

#класс для расчета газодинамических параметров в одной(!) ступени. Для расчета всего компрессора нужно
# использовать несколько экземпляров этого класса,
#передавая результаты расчета из одной ступени (из одного экземпляра класса)
# в последующую ступень (в последюущий экземпляр класса)

class define_number_of_stages:
    def __init__(self, n, G, P_in, P_out, T_in, H, psi_t, tau):
        self.z = 1
        self.pi_k = P_out/P_in
        R = 287 #universal gas constant -понять можно ли через кулпроп вызвать, или Мишину гд
        H_tot = 3.5 * R * T_in * (self.pi_k ** 0.286 - 1)

        phi = 0.55

        #плотность воздуха на входе
        ro = cp.CoolProp.PropsSI('D', 'T', T_in, 'P', P_in, 'Air')
        #объемный расход
        V = G / ro
        #окружная скорость на колесе
        U_k = (V * m.pi * n ** 2 / (phi * 900 * 0.75)) ** (1 / 3)
        #задаем что меридиональная скорость примерно 60% от окружной
        C_a = 0.6 * U_k
        #так как считается что внешний диаметр компрессора постоянен, то U_k будет постоянна на протяжении всей
        # проточной части. Таким образом можно расчитать средний напор на ступень
        H = psi_t * U_k ** 2
        #Количество ступеней:
        z = H_tot / H
        #Окружной диаметр тогда:
        D_k = 60 * U_k / (n * m.pi)
        #Необходимая площадь
        F_1 = D_k ** 2 * m.pi * 0.75 / 4
        # По логике нужно округлить Z, шоб пересчитать окружную скорость и диаметр, но похер





class gasdynamic:

    # конструктор класса, служит для инициализации экземпляра класса, внутри него реализован основной алгоритм расчета параметров однйо ступени
    #TODO! расписать физический смылс параметров(см. ниже):
    #n - обороты, какая размерность?
    #G - расход на входе, кг/с
    #P_in - давление на входе в ступень, Па
    #P_out - давление на выходе из ступени, Па
    #H - энтропия (напор), Дж?
    #psi_t - коэф напора
    #T_in - температура на входе, К
    #tau - коэф реактивности
    #d - относительный втулочный диаметр, D_k/D_вт
    #R - газовая постоянная для воздуха
    #nu_ad - адиабатический КПД - по хорошему должен уточняться в процессе расчета, но это говнокод
    #K_h - какой-то ебучий коэф, хз откуда он но без него не считается
    # D_k - диаметр колеса,  [м]
    # d - относительный втулочный диаметр, D_k/D_вт
    # U_k - окружная скорость, [м/с]
    # tau - коэффициент реактивности
    # psi_t - коэффициент напора
    # T_in - входная температура, К
    # C_a - меридиональная скорость

    def __init__(self, n, G, P_in, P_out, H, psi_t, T_in, tau, d, R, U_k, K_h, nu_ad,):
        #Данные которые нужны для построения профиля
        self.r = 1
        self.betta1 = 1
        self.betta2 = 1
        self.hord = 1
        self.angle_v = 1


        D_k = 60 * U_k / (n * m.pi)
        F_1 = D_k ** 2 * m.pi * 0.75 / 4
        C_a = 0.6 * U_k

    #метод stupen_compressor
    #нужен для дальнейшего расчета параметров ступени TODO! по-моему все-таки нет смысла выделять его в отдельный метод, можно было писать прямо в одну кучу внутри __init__
    # TODO! расписать физический смылс параметров метода ниже:


        r1_mid = m.sqrt(0.5 * (1 + d ** 2))
        h_lop = D_k * (1 - d) * 0.5
        C1_U_mid = U_k * r1_mid * (1 - tau - (psi_t / (2 * r1_mid ** 2)))
        T1_mid = T_in - (C_a ** 2 - C1_U_mid ** 2) / 2010
        P1 = 0.98 * P_in * (T1_mid / T_in) ** 3.5
        ro_1 = P1 / (R * T1_mid)
        F_1 = G / (ro_1 * C_a)
        D_k = m.sqrt(4 * F_1 / (m.pi * (1 - 0.5 ** 2)))
        D_k = round(D_k)
        D_vt = d * D_k
        U_mid = r1_mid * U_k
        W1_mid = m.sqrt(C_a ** 2 + (U_mid - C1_U_mid) ** 2)
        a1_mid = 20.1 * m.sqrt(T1_mid)
        M1_mid = W1_mid / a1_mid
        H_t = psi_t * U_k ** 2
        L_z = H_t * K_h
        h_ad = L_z * nu_ad
        tau_ = 0.5
        r4_mid_new = r1_mid
        r4_mid = 0
        #Цикл сведения радиуса колеса
        while abs(r4_mid_new - r4_mid) > 0.00001:
            r4_mid = r4_mid_new
            C4_U_mid = U_k * r4_mid * (1 - tau_ - psi_t / (2 * r4_mid ** 2))

            C_2 = C4_U_mid ** 2 - C1_U_mid ** 2
            T_4 = T1_mid + L_z / 1004.5 - C_2 / 2010
            L_ad = h_ad - C_2 / 2

            pi_i = (1 + L_ad / (1004.5 * T1_mid)) ** 3.5
            P4 = pi_i * P1
            ro_4 = P4 / (R * T_4)
            F4 = G / (ro_4 * C_a)
            D_4 = m.sqrt(1 - 4 * F4 / (m.pi * (D_k ** 2)))
            r4_mid_new = m.sqrt((1 + D_4 ** 2) / 2)

        alpha_1 = m.degrees(m.atan((U_mid - C1_U_mid) / C_a))
        C2_U_mid = U_k * r4_mid * (1 - tau_ + psi_t / (2 * r4_mid ** 2))
        alpha_2 = m.degrees(m.atan((U_mid - C2_U_mid) / C_a))
        eps_rk_1 = alpha_1 - alpha_2
        return

    #sechenie_tau_const - один из законов профилирования по высоте
    # физический смысл в том, что на протяжении всей высоты лопатки коэф реактивности - tau остается неизменным
    # TODO! расписать физический смылс параметров метода ниже:
    #r- радиус конкретной высоты, на которой ведется расчет, [m]
    #K1 - какой-то коэфициент, на данный момент зафиксирован
    #K2 -какой-то коэфициент, на данный момент зафиксирован
    # phi -
    # tau_ -
    # psi_t1 -
    # r1_mid -
    # C_a -
    # U_k -
    # psi_t -
    def sechenie_tau_const(self, r, K1, K2, phi, tau_, psi_t1, r1_mid, C_a, U_k, psi_t):
        CC_1a = m.sqrt(
            phi ** 2 + 2 * (1 - tau_) * psi_t1 * m.log(r / r1_mid) + 2 * (1 - tau_) ** 2 * (r1_mid ** 2 - r ** 2))
        CC_2a = m.sqrt(
            phi ** 2 - 2 * (1 - tau_) * psi_t1 * m.log(r / r1_mid) + 2 * (1 - tau_) ** 2 * (r1_mid ** 2 - r ** 2))
        C_11a = C_a + K1 * (CC_2a - CC_1a)
        C_22a = C_a + K2 * (CC_2a - CC_1a)
        phi_1a = C_11a / U_k
        phi_2a = C_22a / U_k

        alpha_1sech = m.degrees(m.atan(1 / phi_1a * (r * tau_ + psi_t / (2 * r))))
        alpha_2sech = m.degrees(m.atan(1 / phi_2a * (r * tau_ - psi_t / (2 * r))))
        alpha_3sech = m.degrees(m.atan(r / phi_2a - m.tan(m.radians(alpha_2sech))))
        alpha_4sech = m.degrees(m.atan(r / phi_1a - m.tan(m.radians(alpha_1sech))))
        eps_rk_1 = alpha_1sech - alpha_2sech
        eps_rk_2 = alpha_3sech - alpha_4sech
        return

    #def sechenie_CuR_const(r): #TODO! нужно дореализовать этот метод тоже, он у тебя ниже есть вместе со всеми нужными комментариями: что делает этот метод, какие исходные данные, что должен возвращать
        #...

    # hord_ - ?
    # TODO! в чем суть метода, какие исходные данные, что должен возвразать? (как я понимаю он помоему как раз и должен возвращать ключевые параметры для построения профиля?)
    #r -
    # alpha_1sech -
    # alpha_2sech -
    # eps_rk_1 -
    # ins_ang - угол атаки?


    def hord_(self, r, stage, alpha_1sech, alpha_2sech, eps_rk_1, ins_ang): #TODO! номер ступени при данной
        # архитектуре алгоритма здесь не нужен, удОли!)
        alpha_1air = alpha_1sech - ins_ang
        m_ = 0.23 * (0.45 ** 2) + 0.002 * alpha_2sech
        # !!!TODO расписать нормально b/t - все верно, читай в следующей строке
        tetta_prof = (eps_rk_1 - ins_ang) / (1 - m_ * (m.sqrt(1 / b_t(stage))))
        #TODO! так с b_t(stage) категорически не стоит делать. Удали эту функцию в начале файла.
        # И сделай передачу параметра b_t через список аргументов к методу  hord_(...тут...)
        alpha_2air = (alpha_2sech - ins_ang)
        return

test = gasdynamic()






'''
n = 5200
G = 100 #kg/s
P_in = 100000#Pa
P_out = 500000#Pa
H = 1.375 * 10**4
psi_t = 0.25

R = 287
T_in = 303 #K
tau = 0.5  # Степень реактивности


pi_c = P_out/P_in
H_tot = 3.5 * R * T_in * (pi_c ** 0.286 - 1)

phi = 0.55
ro = cp.CoolProp.PropsSI('D', 'T', T_in, 'P', P_in, 'Air')
V = G/ro

U_k = (V * m.pi * n**2 / (phi * 900 * 0.75))**(1/3)
U_k = round(U_k)

psi_t1 = 0.29
C_a = 0.6 * U_k
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

def profiling(r, stage, alpha_1sech, alpha_2sech, eps_rk_1, ins_ang):
    alpha_1air = alpha_1sech - ins_ang
    m_ = 0.23*(0.45**2)  + 0.002*alpha_2sech
    #!!!TODO расписать нормально b/t
    tetta_prof = (eps_rk_1 - ins_ang)/(1 - m_*(m.sqrt(1/b_t(stage))))
    alpha_2air = (alpha_2sech - ins_ang)
    return

def sechenie_tau_const (r, K1, K2):
    CC_1a = m.sqrt(phi**2 + 2*(1-tau_)*psi_t1* m.log(r/r1_mid) + 2*(1 - tau_)**2 * (r1_mid**2 - r**2))
    CC_2a = m.sqrt(phi ** 2 - 2 * (1 - tau_) * psi_t1 * m.log(r / r1_mid) + 2 * (1 - tau_) ** 2 * (r1_mid ** 2 - r ** 2))
    C_11a = C_a + K1 * (CC_2a - CC_1a)
    C_22a = C_a + K2 * (CC_2a - CC_1a)
    phi_1a = C_11a/U_k
    phi_2a = C_22a/U_k

    alpha_1sech = m.degrees(m.atan(1/phi_1a * (r*tau_ + psi_t/(2*r))))
    alpha_2sech = m.degrees(m.atan(1 / phi_2a * (r * tau_ - psi_t /( 2 * r))))
    alpha_3sech = m.degrees(m.atan(r/phi_2a - m.tan(m.radians(alpha_2sech))))
    alpha_4sech = m.degrees(m.atan(r/phi_1a - m.tan(m.radians(alpha_1sech))))
    eps_rk_1 = alpha_1sech - alpha_2sech
    eps_rk_2 = alpha_3sech - alpha_4sech
    profiling(r, 1, alpha_1sech, alpha_2sech, eps_rk_1, ins_ang=2)
    return

def sechenie_CuR_const(r):
    tau_sech = 1 - (1 -tau_)*(r1_mid/r)**2
    C_11a = C_a/U_k
    C_22a = C_a/U_k - 0.008
    alpha_1sech = m.degrees(m.atan(1 / C_11a * (r * tau_sech + psi_t / (2 * r))))
    alpha_2sech = m.degrees(m.atan(1 / C_22a * (r * tau_sech - psi_t / (2 * r))))
    alpha_3sech = m.degrees(m.atan(r / C_11a - m.tan(m.radians(alpha_2sech))))
    alpha_4sech = m.degrees(m.atan(r / C_22a - m.tan(m.radians(alpha_1sech))))
    eps_rk_1 = alpha_1sech - alpha_2sech
    eps_rk_2 = alpha_3sech - alpha_4sech
    profiling(r, 1, alpha_1sech, alpha_2sech, ins_ang=2)

    return





sechenie_tau_const(r = 0.55, K1= 2.8, K2 = 7.2)
sechenie_CuR_const(r = 0.55)



DATA['r1_mid'] = r1_mid


#stage_(G,H,psi_t, pi_c, R)
'''



def b_t(stage):
    #костыль костылевый
    b = [0.96, 1, 1.15, 1.25, 1.25, 1.35, 1.35, 1.55, 1.2, 1.2]
    return b[stage-1]

def h_b(stage):
    h = [2.5, 2.3, 2.1, 1.95, 1.8, 1.6, 1.5, 1.5, 1.5, 1.5]
    return h[stage-1]

class define_number_of_stages:
    def __init__(self, n, G, P_in, P_out, T_in, H, psi_t, tau):
        self.z = 1
        self.pi_k = P_out/P_in
        R = 287 #universal gas constant -понять можно ли через кулпроп вызвать, или Мишину гд
        H_tot = 3.5 * R * T_in * (self.pi_k ** 0.286 - 1)

        phi = 0.55

        #плотность воздуха на входе
        ro = cp.CoolProp.PropsSI('D', 'T', T_in, 'P', P_in, 'Air')
        #объемный расход
        V = G / ro
        #окружная скорость на колесе
        U_k = (V * m.pi * n ** 2 / (phi * 900 * 0.75)) ** (1 / 3)
        #задаем что меридиональная скорость примерно 60% от окружной
        C_a = 0.6 * U_k
        #так как считается что внешний диаметр компрессора постоянен, то U_k будет постоянна на протяжении всей
        # проточной части. Таким образом можно расчитать средний напор на ступень
        H = psi_t * U_k ** 2
        #Количество ступеней:
        z = H_tot / H
        #Окружной диаметр тогда:
        D_k = 60 * U_k / (n * m.pi)
        #Необходимая площадь
        F_1 = D_k ** 2 * m.pi * 0.75 / 4
        # По логике нужно округлить Z, шоб пересчитать окружную скорость и диаметр, но похер







class stages:
    def __init__(self):
        n = 5200
        G = 100  # kg/s
        P_in = 100000  # Pa
        P_out = 500000  # Pa
        H = 1.375 * 10 ** 4
        psi_t = 0.25

        R = 287
        T_in = 303  # K
        tau = 0.5  # Степень реактивности

        pi_c = P_out / P_in
        H_tot = 3.5 * R * T_in * (pi_c ** 0.286 - 1)

        phi = 0.55
        ro = cp.CoolProp.PropsSI('D', 'T', T_in, 'P', P_in, 'Air')
        V = G / ro

        U_k = (V * m.pi * n ** 2 / (phi * 900 * 0.75)) ** (1 / 3)
        U_k = round(U_k)

        psi_t1 = 0.29
        C_a = 0.6 * U_k
        H = psi_t * U_k ** 2

        z = H_tot / H
        D_k = 60 * U_k / (n * m.pi)
        F_1 = D_k ** 2 * m.pi * 0.75 / 4
        d = 0.5
        K_h = 0.99
        nu_ad = 0.84
        pass

class gasdynamic:
    def __init__(self, n, G, P_in, P_out, H, psi_t, T_in, tau, d, R):
        #Данные которые нужны для построения профиля
        self.r = 1
        self.betta1 = 1
        self.betta2 = 1
        self.hord = 1
        self.angle_v = 1

        pi_c = P_out / P_in
        H_tot = 3.5 * R * T_in * (pi_c ** 0.286 - 1)
        phi = 0.55
        ro = cp.CoolProp.PropsSI('D', 'T', T_in, 'P', P_in, 'Air')
        V = G / ro
        U_k = (V * m.pi * n ** 2 / (phi * 900 * 0.75)) ** (1 / 3)
        U_k = round(U_k)
        psi_t1 = 0.29
        C_a = 0.6 * U_k
        H = psi_t * U_k ** 2
        z = H_tot / H
        D_k = 60 * U_k / (n * m.pi)
        F_1 = D_k ** 2 * m.pi * 0.75 / 4
        K_h = 0.99
        nu_ad = 0.84
        DATA = {}
        r1_mid = []
        z = 1

    def stupen_compressor(self, D_k, d, U_k, tau, psi_t, T_in, C_a, P_in, R, G, K_h, nu_ad, psi_t1):
        r1_mid = m.sqrt(0.5 * (1 + d ** 2))
        h_lop = D_k * (1 - d) * 0.5
        C1_U_mid = U_k * r1_mid * (1 - tau - (psi_t / (2 * r1_mid ** 2)))
        T1_mid = T_in - (C_a ** 2 - C1_U_mid ** 2) / 2010
        P1 = 0.98 * P_in * (T1_mid / T_in) ** 3.5
        ro_1 = P1 / (R * T1_mid)
        F_1 = G / (ro_1 * C_a)
        D_k = m.sqrt(4 * F_1 / (m.pi * (1 - 0.5 ** 2)))
        D_k = round(D_k)
        D_vt = d * D_k
        U_mid = r1_mid * U_k
        W1_mid = m.sqrt(C_a ** 2 + (U_mid - C1_U_mid) ** 2)
        a1_mid = 20.1 * m.sqrt(T1_mid)
        M1_mid = W1_mid / a1_mid
        H_t = psi_t * U_k ** 2
        L_z = H_t * K_h
        h_ad = L_z * nu_ad
        tau_ = 0.5
        r4_mid_new = r1_mid
        r4_mid = 0
        while abs(r4_mid_new - r4_mid) > 0.00001:
            r4_mid = r4_mid_new
            C4_U_mid = U_k * r4_mid * (1 - tau_ - psi_t1 / (2 * r4_mid ** 2))

            C_2 = C4_U_mid ** 2 - C1_U_mid ** 2
            T_4 = T1_mid + L_z / 1004.5 - C_2 / 2010
            L_ad = h_ad - C_2 / 2

            pi_i = (1 + L_ad / (1004.5 * T1_mid)) ** 3.5
            P4 = pi_i * P1
            ro_4 = P4 / (R * T_4)
            F4 = G / (ro_4 * C_a)
            D_4 = m.sqrt(1 - 4 * F4 / (m.pi * (D_k ** 2)))
            r4_mid_new = m.sqrt((1 + D_4 ** 2) / 2)

        alpha_1 = m.degrees(m.atan((U_mid - C1_U_mid) / C_a))
        C2_U_mid = U_k * r4_mid * (1 - tau_ + psi_t1 / (2 * r4_mid ** 2))
        alpha_2 = m.degrees(m.atan((U_mid - C2_U_mid) / C_a))
        eps_rk_1 = alpha_1 - alpha_2
        return

    def sechenie_tau_const(self, r, K1, K2, phi, tau_, psi_t1, r1_mid, C_a, U_k, psi_t):
        CC_1a = m.sqrt(
            phi ** 2 + 2 * (1 - tau_) * psi_t1 * m.log(r / r1_mid) + 2 * (1 - tau_) ** 2 * (r1_mid ** 2 - r ** 2))
        CC_2a = m.sqrt(
            phi ** 2 - 2 * (1 - tau_) * psi_t1 * m.log(r / r1_mid) + 2 * (1 - tau_) ** 2 * (r1_mid ** 2 - r ** 2))
        C_11a = C_a + K1 * (CC_2a - CC_1a)
        C_22a = C_a + K2 * (CC_2a - CC_1a)
        phi_1a = C_11a / U_k
        phi_2a = C_22a / U_k

        alpha_1sech = m.degrees(m.atan(1 / phi_1a * (r * tau_ + psi_t / (2 * r))))
        alpha_2sech = m.degrees(m.atan(1 / phi_2a * (r * tau_ - psi_t / (2 * r))))
        alpha_3sech = m.degrees(m.atan(r / phi_2a - m.tan(m.radians(alpha_2sech))))
        alpha_4sech = m.degrees(m.atan(r / phi_1a - m.tan(m.radians(alpha_1sech))))
        eps_rk_1 = alpha_1sech - alpha_2sech
        eps_rk_2 = alpha_3sech - alpha_4sech
        return

    def hord_(self, r, stage, alpha_1sech, alpha_2sech, eps_rk_1, ins_ang):
        alpha_1air = alpha_1sech - ins_ang
        m_ = 0.23 * (0.45 ** 2) + 0.002 * alpha_2sech
        # !!!TODO расписать нормально b/t
        tetta_prof = (eps_rk_1 - ins_ang) / (1 - m_ * (m.sqrt(1 / b_t(stage))))
        alpha_2air = (alpha_2sech - ins_ang)
        return

test = gasdynamic()






'''
n = 5200
G = 100 #kg/s
P_in = 100000#Pa
P_out = 500000#Pa
H = 1.375 * 10**4
psi_t = 0.25

R = 287
T_in = 303 #K
tau = 0.5  # Степень реактивности


pi_c = P_out/P_in
H_tot = 3.5 * R * T_in * (pi_c ** 0.286 - 1)

phi = 0.55
ro = cp.CoolProp.PropsSI('D', 'T', T_in, 'P', P_in, 'Air')
V = G/ro

U_k = (V * m.pi * n**2 / (phi * 900 * 0.75))**(1/3)
U_k = round(U_k)

psi_t1 = 0.29
C_a = 0.6 * U_k
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

def profiling(r, stage, alpha_1sech, alpha_2sech, eps_rk_1, ins_ang):
    alpha_1air = alpha_1sech - ins_ang
    m_ = 0.23*(0.45**2)  + 0.002*alpha_2sech
    #!!!TODO расписать нормально b/t
    tetta_prof = (eps_rk_1 - ins_ang)/(1 - m_*(m.sqrt(1/b_t(stage))))
    alpha_2air = (alpha_2sech - ins_ang)
    return

def sechenie_tau_const (r, K1, K2):
    CC_1a = m.sqrt(phi**2 + 2*(1-tau_)*psi_t1* m.log(r/r1_mid) + 2*(1 - tau_)**2 * (r1_mid**2 - r**2))
    CC_2a = m.sqrt(phi ** 2 - 2 * (1 - tau_) * psi_t1 * m.log(r / r1_mid) + 2 * (1 - tau_) ** 2 * (r1_mid ** 2 - r ** 2))
    C_11a = C_a + K1 * (CC_2a - CC_1a)
    C_22a = C_a + K2 * (CC_2a - CC_1a)
    phi_1a = C_11a/U_k
    phi_2a = C_22a/U_k

    alpha_1sech = m.degrees(m.atan(1/phi_1a * (r*tau_ + psi_t/(2*r))))
    alpha_2sech = m.degrees(m.atan(1 / phi_2a * (r * tau_ - psi_t /( 2 * r))))
    alpha_3sech = m.degrees(m.atan(r/phi_2a - m.tan(m.radians(alpha_2sech))))
    alpha_4sech = m.degrees(m.atan(r/phi_1a - m.tan(m.radians(alpha_1sech))))
    eps_rk_1 = alpha_1sech - alpha_2sech
    eps_rk_2 = alpha_3sech - alpha_4sech
    profiling(r, 1, alpha_1sech, alpha_2sech, eps_rk_1, ins_ang=2)
    return

def sechenie_CuR_const(r):
    tau_sech = 1 - (1 -tau_)*(r1_mid/r)**2
    C_11a = C_a/U_k
    C_22a = C_a/U_k - 0.008
    alpha_1sech = m.degrees(m.atan(1 / C_11a * (r * tau_sech + psi_t / (2 * r))))
    alpha_2sech = m.degrees(m.atan(1 / C_22a * (r * tau_sech - psi_t / (2 * r))))
    alpha_3sech = m.degrees(m.atan(r / C_11a - m.tan(m.radians(alpha_2sech))))
    alpha_4sech = m.degrees(m.atan(r / C_22a - m.tan(m.radians(alpha_1sech))))
    eps_rk_1 = alpha_1sech - alpha_2sech
    eps_rk_2 = alpha_3sech - alpha_4sech
    profiling(r, 1, alpha_1sech, alpha_2sech, ins_ang=2)

    return





sechenie_tau_const(r = 0.55, K1= 2.8, K2 = 7.2)
sechenie_CuR_const(r = 0.55)



DATA['r1_mid'] = r1_mid


#stage_(G,H,psi_t, pi_c, R)
'''