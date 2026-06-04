"""
Модуль расчета приближенного решения прямой задачи внутренней баллистики.

@Автор: <Барышев Савва>
Назначение: Определение оптимальных проектных параметров АО
Версия: 1.0

Module for approximate solution for the direct problem of internal ballistics.

@author: <Savva Baryshev>
Purpose: Cannon system parameters determination.
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import copy
from ODE_solvers import *
# Настройка отображения с запятой в качестве десятичного разделителя
import locale

try:
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
except:
    pass


class Cannon_System_Parameters:
    # Класс входных проектных параметров АО для тестового задания

    def __init__(self):
        # Перезапись параметров из файла с исходными данными для дальнейшего анализа:
        ds = pd.read_csv('lab_data.csv', skiprows=[10]).set_index('var')
        # Чтение файла "lab_data.csv":
        self.__dict__.update(ds["value"].to_dict())

        self.v_pm = 770 # дульная скорость снаряда, м/с
        self.p_a_max = 290e6 # допустимое максимальное давление в канале ствола, Па
        self.f = 1e6 # сила условного пороха, Дж/кг
        self.hi = 1.25 #

        self.dksi = 1e-3

        self.Delta = np.arange(500, 876, 25)  # диапазон варьирования плотности заряжания, кг/м³

        if self.__dict__.get('kappa_2', 0) == 0:
            # Набор параметров для трубчатого пороха
            self.B_t = np.arange(0.75, 4.6, 0.25) # диапазон варьирования параметра условий заряжания (параметра Дроздова) для трубчатого пороха
            self.eta_e_t = np.arange(0.25, 0.76, 0.05) # диапазон варьирования параметра относительного положения снаряда в конце горения для трубчатого пороха

            # Многомерная сетка параметров для трубчатого пороха
            self.Delta_m, self.B_m, self.eta_e_m = np.meshgrid(
                self.Delta, self.B_t, self.eta_e_t, indexing='ij'
            )
            self.powder_type = 'tubular'  # индикатор трубчатого пороха

        else:
            # Набор параметров для семиканального пороха
            self.B_s = np.arange(0.5, 2.1, 0.1) # диапазон варьирования параметра условий заряжания (параметра Дроздова) для семиканального пороха
            self.eta_e_s = np.arange(0.5, 1.1, 0.05) # диапазон варьирования параметра относительного положения снаряда в конце горения для трубчатого пороха

            # Многомерная сетка параметров для семиканального пороха
            self.Delta_m, self.B_m, self.eta_e_m = np.meshgrid(
                self.Delta, self.B_s, self.eta_e_s, indexing='ij'
            )
            self.powder_type = 'seven_core'  # индикатор семиканального пороха



class Math_Model:
    # класс, реализующий математическую модель приближенного решения ПЗВБ

    def __init__(self, parameters):
        self.dx_dksi = None
        self.p = parameters

    # Предварительный период вычисления (параметры газоприхода в начале пиродинамического периода):

    def dzeta(self):
        # метод вычисления относительной массы воспламенительного состава:
        numerator = ((1 / self.p.Delta_m) - (1 / self.p.delta))
        denominator = (self.p.f / self.p.p_ign + self.p.b)
        return numerator / denominator

    def psi_s(self):
        # метод вычисления относительной массы сгоревшей части заряда в момент распада порохового зерна:
        return self.p.kappa_1*(1 + self.p.lambda_1)

    def psi_0(self):
        # метод вычисления относительной массы сгоревшей части заряда в момент форсирования:
        numerator = self.p.p_0 * (1/self.p.Delta_m - 1/self.p.delta - self.p.b*self.dzeta()) - self.p.f*self.dzeta()
        denominator = self.p.f - self.p.p_0*(1/self.p.delta - self.p.b)
        return numerator / denominator

    def z_0(self):
        # метод вычисления толщины сгоревшего свода порохового элемента в момент форсирования:
        numerator = 2*self.psi_0()
        denominator = self.p.kappa_1*(1 + (1 + (4*self.p.lambda_1*self.psi_0()/self.p.kappa_1))**(1/2))

        return numerator / denominator

    def sigma_0(self):
        # метод вычисления относительной площади поверхности горения в момент форсирования:
        return 1 + 2*self.p.lambda_1*self.z_0()

    def ksi_s(self):
        # метод вычисления толщины сгоревшего свода порохового элемента от начала движения снаряда, до момента распада зерна:
        return 1 - self.z_0()

    # Пиродинамический период:
    # ξ - шаг интегрирования
    # x[0] - Lambda, то есть приведённый путь снаряда по каналу ствола

    def ODE(self, ksi, x):
        # метод записи интегрального уравнения пиродинамического периода:
        self.dx_dksi = np.zeros(1)
        self.dx_dksi[0] = ksi*self.p.f*self.p.Delta_m*self.p.B_m/self.p_m(ksi, x) # производная
        return self.dx_dksi

    def eta_r(self, ksi):
        # метод вычисления термического КПД:
        return (self.p.k - 1)/2 *self.p.B_m*ksi**2

    def p_m(self, ksi, x):
        # метод вычисления среднего баллистического давления:
        numerator = self.psi_ksi(ksi)+ self.dzeta() - self.eta_r(ksi)
        denominator = (1 + x[0])/self.p.Delta_m - 1/self.p.delta + ((1 - self.p.b*self.p.delta)/self.p.delta)*self.psi_ksi(ksi) - self.p.b*self.dzeta()
        return self.p.f*numerator/denominator

    def psi_ksi(self, ksi):
        if ksi < self.ksi_s():
            psi_ksi = self.psi_0() + self.p.kappa_1 * self.sigma_0() * ksi + self.p.kappa_1 * self.p.lambda_1 * ksi ** 2
        else:
            psi_ksi = self.psi_s() + self.p.kappa_2 * (ksi - self.ksi_s()) * (1 + self.p.lambda_2 * (ksi - self.ksi_s()))

        return psi_ksi

    def init_conditions(self):
        # метод возврата списка начальных условий:
        return [0]

    def end_conditions(self, ksi, x):
        # метод конечных условий, реализующий вычисление толщины сгоревшего свода порохового элемента от начала движения снаряда, до момента сгорания зерна:
        return self.p.z_e - self.z_0() - ksi

    def report(self, ksi, x):
        return [self.p.B_m, self.p.Delta_m, self.p_m(ksi, x), self.eta_r(ksi), self.psi_ksi(ksi)]

    # Период адиабатического расширения:

    def eta_r_adiabatic(self, Lambda, Lambda_e, eta_r_e):
        # метод вычисления термического КПД для адиабатического периода:
        numerator = 1 - self.p.b*self.p.Delta_m*(1 + self.dzeta()) + Lambda_e
        denominator = 1 - self.p.b*self.p.Delta_m*(1 + self.dzeta()) + Lambda
        return 1 + self.dzeta() - (1 + self.dzeta() - eta_r_e)*(numerator/denominator)**(self.p.k - 1)

    def p_m_adiabatic(self, Lambda, Lambda_e, p_m_e):
        # метод вычисления среднего баллистического давления для адиабатического периода:
        numerator = 1 - self.p.b*self.p.Delta_m*(1 + self.dzeta()) + Lambda_e
        denominator = 1 - self.p.b*self.p.Delta_m*(1 + self.dzeta()) + Lambda
        return p_m_e*(numerator/denominator)**self.p.k

    # Привлечение конструктивных параметров:

    def omega_q(self, eta_r_m):
        # метод вычисления относительной массы порохового заряда:
        numerator = self.p.K
        denominator = (2/(self.p.k - 1)) * (eta_r_m*self.p.f/self.p.v_pm**2) - (1 + self.dzeta())/3
        return numerator / denominator

    def W_0(self, eta_r_m):
        # метод вычисления объема зарядной каморы:
        numerator = self.omega_q(eta_r_m)*self.p.q
        denominator = self.p.Delta_m
        return numerator / denominator

    def l_m(self, Lambda_m, eta_r_m):
        # метод вычисления длины ведущей части канала ствола:
        S = self.p.n_s*math.pi*self.p.d**2/4
        l_0 = self.W_0(eta_r_m)/S
        return Lambda_m*l_0

    def I_e(self, eta_r_m):
        # метод вычисления импульса пороха:
        phi = self.p.K + 1 / 3 * self.omega_q(eta_r_m)*self.p.q * (1 + self.dzeta()) / self.p.q
        numerator = (self.p.f*self.omega_q(eta_r_m)*self.p.q**2*phi*self.p.B_m)
        denominator = self.p.n_s * math.pi * self.p.d ** 2 / 4
        return numerator / denominator


class Simulation:
    # Класс получения приближенных решений прямой задачи внутренней баллистики:

    def __init__(self, model: Math_Model):
        self.m = model

        # Получение значений параметров на пиростатическом периоде:
        result_pyrodynamic = RungeKutta4(self.m.ODE, self.m.init_conditions(), self.m.end_conditions, self.m.report, self.m.p.dksi, 0, 10000)

        pd.set_option('display.precision', 5)

        self.df_pyrodynamic = pd.DataFrame(result_pyrodynamic, columns=['ksi', 'Lambda', 'B', 'Delta', 'p_m', 'eta_r', 'psi_ksi'])

        # Получение значений параметров в конце адиабатического периода:
        powder_burnout = self.df_pyrodynamic.iloc[-1] # условие момента полного выгорания порохового зерна
        Lambda_e = powder_burnout['Lambda'] # значение приведённого пути снаряда по каналу ствола в момент полного выгорания порохового зерна
        p_m_e = powder_burnout['p_m'] # значение среднего баллистического давление в момент полного выгорания порохового зерна
        eta_r_e = powder_burnout['eta_r'] # значение параметра относительного положения снаряда в канале ствола в момент полного выгорания порохового зерна
        ksi_e = powder_burnout['ksi'] # значение толщины сгоревшего свода порохового элемента в момент полного выгорания порохового зерна
        psi_e = powder_burnout['psi_ksi'] # значение массы сгоревшей части заряда в момент полного выгорания порохового зерна

        Lambda_m = Lambda_e/self.m.p.eta_e_m # приведённый путь снаряда по каналу ствола в момент вылета снаряда из канала ствола

        # Проверка на полное выгорание пороха внутри ствола:
        if Lambda_m > Lambda_e:
            Lambda_values = np.arange(Lambda_e, Lambda_m, self.m.p.dksi)

            results_adiabatic = []
            for Lambda in Lambda_values:
                p_m_adiabatic = self.m.p_m_adiabatic(Lambda, Lambda_e, p_m_e)
                eta_r_adiabatic = self.m.eta_r_adiabatic(Lambda, Lambda_e, eta_r_e)

                results_adiabatic.append([ksi_e, Lambda, self.m.p.B_m, self.m.p.Delta_m, p_m_adiabatic, eta_r_adiabatic, psi_e])

            self.df_adiabatic = pd.DataFrame(results_adiabatic,  columns=['ksi', 'Lambda', 'B', 'Delta', 'p_m', 'eta_r', 'psi_ksi'])

            # Объединение таблиц для пиродинамического и адиабатического периодов в одну:
            self.df_full = pd.concat([self.df_pyrodynamic, self.df_adiabatic], ignore_index=True)

        else:
            print("Не произошло полного выгорания порохового зерна внутри канала ствола")
            self.df_adiabatic = None
            self.df_full = self.df_pyrodynamic

        # Получение значений параметров в конце адиабатического периода:
        muzzle = self.df_full.iloc[-1] # условие момента выхода снаряда из канала ствола
        # Lambda_m = muzzle['Lambda'] # значение приведённого пути снаряда по каналу ствола в момент выхода снаряда из канала ствола
        # p_m_m = muzzle['p_m'] # значение среднего баллистического давление в момент выхода снаряда из канала ствола
        # eta_r_m = muzzle['eta_r'] # значение параметра относительного положения снаряда в канале ствола в момент выхода снаряда из канала ствола
        # ksi_m = muzzle['ksi'] # значение толщины сгоревшего свода порохового элемента в момент выхода снаряда из канала ствола
        # psi_m = muzzle['psi_ksi'] # значение массы сгоревшей части заряда в момент выхода снаряда из канала ствола

        Lambda_m = Lambda_e/self.m.p.eta_e_m # значение приведённого пути снаряда по каналу ствола в момент выхода снаряда из канала ствола
        p_m_m = self.m.p_m_adiabatic(Lambda_m, Lambda_e, p_m_e) # значение среднего баллистического давление в момент выхода снаряда из канала ствола
        eta_r_m = self.m.eta_r_adiabatic(Lambda_m, Lambda_e,eta_r_e) # значение параметра относительного положения снаряда в канале ствола в момент выхода снаряда из канала ствола
        ksi_m = muzzle['ksi'] # значение толщины сгоревшего свода порохового элемента в момент выхода снаряда из канала ствола
        psi_m = muzzle['psi_ksi'] # значение массы сгоревшей части заряда в момент выхода снаряда из канала ствола

        # Сохранение в атрибуты
        self.Lambda_e = Lambda_e
        self.p_m_e = p_m_e
        self.eta_r_e = eta_r_e
        self.ksi_e = ksi_e
        self.psi_e = psi_e

        self.Lambda_m = Lambda_m
        self.p_m_m = p_m_m
        self.eta_r_m = eta_r_m
        self.ksi_m = ksi_m
        self.psi_m = psi_m


class Pressure_Rate_Table:
    # класс составления таблицы соответствия (Δ, B) = p_max

    def __init__(self, params: Cannon_System_Parameters):
        self.params = params

    def build_table_fast(self):
        # Построение таблицы:

        # Получение размеров сеток
        n_Delta = len(self.params.Delta)
        n_B = len(self.params.B_s) if hasattr(self.params, 'B_s') else len(self.params.B_t)

        print(f"Размер сетки: Δ = {n_Delta} значений, B = {n_B} значений")
        print(f"Диапазон Δ: {self.params.Delta[0]} - {self.params.Delta[-1]}")
        print(
            f"Диапазон B: {self.params.B_s[0] if hasattr(self.params, 'B_s') else self.params.B_t[0]} - {self.params.B_s[-1] if hasattr(self.params, 'B_s') else self.params.B_t[-1]}")

        # Создание пустой таблицы
        pressure_matrix = np.zeros((n_B, n_Delta))

        # Счетчики успешных и неудачных расчетов
        success_count = 0
        fail_count = 0

        # Перебор индексов
        for i in range(n_Delta):  # по Delta
            for j in range(n_B):  # по B
                try:
                    # Берем значения из существующих сеток
                    Delta_val = self.params.Delta_m[i, j, 0] if self.params.Delta_m.ndim == 3 else self.params.Delta_m[
                        i, j]
                    B_val = self.params.B_m[i, j, 0] if self.params.B_m.ndim == 3 else self.params.B_m[i, j]
                    eta_e_val = self.params.eta_e_m[i, j, 0] if self.params.eta_e_m.ndim == 3 else self.params.eta_e_m[
                        i, j]

                    # Приводим к скалярным значениям
                    Delta_val = float(Delta_val)
                    B_val = float(B_val)
                    eta_e_val = float(eta_e_val)

                    params_copy = copy.deepcopy(self.params)
                    params_copy.Delta_m = Delta_val
                    params_copy.B_m = B_val
                    params_copy.eta_e_m = eta_e_val

                    # Расчет с использованием копии
                    model = Math_Model(params_copy)
                    sim = Simulation(model)

                    if sim.df_full is not None and len(sim.df_full) > 0:
                        p_max = sim.df_full['p_m'].max() / 1e6
                        pressure_matrix[j, i] = p_max
                        success_count += 1
                        print(f"i={i}, j={j}: Δ={Delta_val:.0f}, B={B_val:.2f}, p_max={p_max:.2f} МПа")
                    else:
                        pressure_matrix[j, i] = np.nan
                        fail_count += 1
                        print(f"i={i}, j={j}: Δ={Delta_val:.0f}, B={B_val:.2f} -> ОШИБКА (нет данных)")

                except Exception as e:
                    pressure_matrix[j, i] = np.nan
                    fail_count += 1
                    print(f"i={i}, j={j}: ОШИБКА: {str(e)[:100]}")

        print(f"\nРасчет завершен. Успешно: {success_count}, Ошибок: {fail_count}")

        B_index = self.params.B_s if hasattr(self.params, 'B_s') else self.params.B_t
        Delta_columns = self.params.Delta

        table = pd.DataFrame(
            pressure_matrix,
            index=np.round(B_index, 3),
            columns=np.round(Delta_columns, 0)
        )

        # Удаляем строки и столбцы, где все значения NaN
        table = table.dropna(how='all', axis=0).dropna(how='all', axis=1)

        return table

    def display_table(self, table):
        # Вывод таблицы в консоль:
        print("\n" + "=" * 80)
        print("ТАБЛИЦА МАКСИМАЛЬНЫХ ДАВЛЕНИЙ p_m_max (МПа)")
        print("=" * 80)

        # Форматируем вывод с двумя знаками после запятой
        formatted_table = table.round(2)
        print(formatted_table)
        print("=" * 80)

        # Дополнительно выводим информацию о размерах
        print(f"\nРазмер таблицы: {len(table.index)} строк (B) x {len(table.columns)} столбцов (Δ)")

    def save_table(self, table, filename='Pressure_Rate_Table.csv'):
        # Сохранение таблицы в CSV файл

        os.makedirs('results/Direct_Problem', exist_ok=True)

        filepath = os.path.join('results/Direct_Problem', filename)
        table.to_csv(filepath, index=True, encoding='utf-8-sig')

        table_comma = table.copy()

        for col in table_comma.columns:
            table_comma[col] = table_comma[col].apply(lambda x: f"{x:.6f}".replace('.', ',') if pd.notna(x) else "")

        table_comma.index = table_comma.index.map(lambda x: f"{x:.6f}".replace('.', ','))

        filepath_comma = os.path.join('results/Direct_Problem', filename)
        table_comma.to_csv(filepath_comma, index=True, encoding='utf-8-sig')

        print(f"Таблица сохранена в файл: {filepath}")
        print(f"Таблица с запятой сохранена в файл: {filepath_comma}")



class Required_Pressure_Rate_Table:
    # класс составления таблицы соответствия (Δ, B*) = p_max*

    def __init__(self, params: Cannon_System_Parameters, pressure_table: pd.DataFrame):
        self.params = params
        self.pressure_table = pressure_table

    def find_bracketing_B(self, Delta_val, target_pmax, B_values, p_max_values):

        for idx in range(len(B_values) - 1):
            p_left = p_max_values[idx]
            p_right = p_max_values[idx + 1]

            # Проверка, что target находится между p_left и p_right
            if p_left >= target_pmax >= p_right:
                return B_values[idx], B_values[idx + 1], p_left, p_right

            elif p_left <= target_pmax <= p_right:
                return B_values[idx], B_values[idx + 1], p_left, p_right

        # target вне диапазона
        if p_max_values[0] < target_pmax:
            # Давление при минимальном B уже меньше целевого
            return None, None, None, None
        elif p_max_values[-1] > target_pmax:
            # Давление при максимальном B еще больше целевого
            return None, None, None, None

        return None, None, None, None

    def interpolate_B(self, B_left, B_right, p_left, p_right, target_pmax):
        # p_left > p_right:
        t = (target_pmax - p_left) / (p_right - p_left)
        B_target = B_left + t * (B_right - B_left)
        return B_target

    def get_pmax_for_B(self, Delta_val, B_val, eta_e_val=0.7):

        params_copy = copy.deepcopy(self.params)
        params_copy.Delta_m = Delta_val
        params_copy.B_m = B_val
        params_copy.eta_e_m = eta_e_val

        model = Math_Model(params_copy)
        sim = Simulation(model)

        if sim.df_full is not None:
            return sim.df_full['p_m'].max(), sim
        else:
            return 0, None

    def refine_B(self, Delta_val, B_initial, target_pmax, eta_e_val=0.7, tol=1e-4, max_iter=10):

        B_current = B_initial
        B_step = 0.05

        for _ in range(max_iter):
            p_current, sim = self.get_pmax_for_B(Delta_val, B_current, eta_e_val)

            if abs(p_current - target_pmax) < tol:
                return B_current, sim

            if p_current > target_pmax:
                # Давление слишком высокое - нужно увеличить B
                B_current += B_step
            else:
                # Давление слишком низкое - нужно уменьшить B
                B_current -= B_step

            B_step *= 0.5  # уменьшаем шаг

        return B_current, sim

    def build_table(self):

        n_Delta = len(self.params.Delta)

        # Получаем значения B и Delta из таблицы
        B_values = self.pressure_table.index.values  # значения B
        Delta_values = self.pressure_table.columns.values  # значения Delta

        # Инициализация массивов для результатов
        B_a_values = np.zeros(n_Delta)
        p_max_values = np.zeros(n_Delta)
        p_m_e_values = np.zeros(n_Delta)
        eta_r_e_values = np.zeros(n_Delta)
        Lambda_e_values = np.zeros(n_Delta)
        dzeta_values = np.zeros(n_Delta)

        target_pmax = float(self.params.p_a_max) / 1e6  # 290 МПа (в МПа, как в таблице)

        print("\n" + "=" * 100)
        print("ПОДБОР B_a ДЛЯ КАЖДОГО Δ (p_max = 290 МПа)")
        print("=" * 100)

        for i, Delta_val in enumerate(Delta_values):
            print(f"\nΔ = {Delta_val} кг/м³:")

            # Получаем строку давлений для данного Delta (из таблицы)
            p_max_row = self.pressure_table[Delta_val].values

            # Находим интервал, содержащий target_pmax
            B_left, B_right, p_left, p_right = self.find_bracketing_B(
                Delta_val, target_pmax, B_values, p_max_row
            )

            if B_left is None:
                print(f"    ВНИМАНИЕ: целевое давление {target_pmax} МПа вне диапазона")
                print(f"    p_max при B_min={B_values[0]:.2f}: {p_max_row[0]:.2f} МПа")
                print(f"    p_max при B_max={B_values[-1]:.2f}: {p_max_row[-1]:.2f} МПа")
                B_a_values[i] = np.nan
                p_max_values[i] = np.nan
                p_m_e_values[i] = np.nan
                eta_r_e_values[i] = np.nan
                Lambda_e_values[i] = np.nan
                dzeta_values[i] = np.nan
                continue

            print(f"    Интервал: B∈[{B_left:.2f}, {B_right:.2f}], p∈[{p_left:.2f}, {p_right:.2f}] МПа")

            # Первичная интерполяция
            B_initial = self.interpolate_B(B_left, B_right, p_left, p_right, target_pmax)
            print(f"    Начальное приближение B = {B_initial:.6f}")

            # Уточнение методом половинного деления
            B_final, sim = self.refine_B(Delta_val, B_initial, target_pmax * 1e6, tol=1e-4)

            # Сохраняем результаты
            B_a_values[i] = B_final
            p_max_values[i] = target_pmax

            if sim is not None:
                p_m_e_values[i] = sim.p_m_e / 1e6 if hasattr(sim, 'p_m_e') else 0
                eta_r_e_values[i] = sim.eta_r_e if hasattr(sim, 'eta_r_e') else 0
                Lambda_e_values[i] = sim.Lambda_e if hasattr(sim, 'Lambda_e') else 0
                dzeta_values[i] = sim.m.dzeta() if hasattr(sim.m, 'dzeta') else 0

            print(f"    => B_a = {B_final:.6f}, p_max = {target_pmax:.2f} МПа")

        # Создание DataFrame
        result_dict = {
            'Δ, кг/м³': Delta_values,
            'B_a': B_a_values,
            'p_max, МПа': p_max_values,
            'p_m_e, МПа': p_m_e_values,
            'η_r_e': eta_r_e_values,
            'Λ_e': Lambda_e_values,
            'ζ': dzeta_values
        }

        table = pd.DataFrame(result_dict)

        # Округляем числа
        for col in ['B_a', 'p_max, МПа', 'p_m_e, МПа', 'η_r_e', 'Λ_e', 'ζ']:
            if col in table.columns:
                table[col] = table[col].round(6)

        return table

    def display_table(self, table):
        # Вывод таблицы в консоль
        print("\n" + "=" * 100)
        print("ТАБЛИЦА ТРЕБУЕМЫХ ПАРАМЕТРОВ (p_max = 290 МПа)")
        print("=" * 100)
        print(table.to_string(index=False))
        print("=" * 100)

    def save_table(self, table, filename='Required_Pressure_Rate_Table.csv'):
        # Сохранение таблицы в CSV файл

        os.makedirs('results/Direct_Problem', exist_ok=True)

        filepath = os.path.join('results/Direct_Problem', filename)
        table.to_csv(filepath, index=False, encoding='utf-8-sig')

        # Дополнительное сохранение с запятой в качестве десятичного разделителя
        table_comma = table.copy()
        for col in table_comma.columns:
            if table_comma[col].dtype == 'float64':
                table_comma[col] = table_comma[col].apply(lambda x: f"{x:.6f}".replace('.', ','))

        filepath_comma = os.path.join('results/Direct_Problem', filename)
        table_comma.to_csv(filepath_comma, index=False, encoding='utf-8-sig')

        print(f"Таблица сохранена в файл: {filepath}")
        print(f"Таблица с запятой сохранена в файл: {filepath_comma}")


class Plotter_Pressure_Rate_Table:
    # класс вывода диаграммы баллистических решений в координатах (Δ, B) = p_max

    def __init__(self, params: Cannon_System_Parameters, pressure_table: pd.DataFrame):

        self.params = params
        self.pressure_table = pressure_table

    def plot_scatter(self, save=False, filename='Pressure_Scatter.png'):
        # Построение точечной диаграммы зависимости p_max от Δ и B

        # Настройка шрифтов
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

        # Размер фигуры
        fig, ax = plt.subplots(figsize=(20, 10), facecolor=(1, 1, 1))

        # Параметры маркеров
        m = 's'  # маркер - квадрат
        ms = 128  # размер маркера
        mec = 'k'  # цвет обводки маркера

        # Пороговое давление (МПа)
        p_threshold = self.params.p_a_max / 1e6  # 290 МПа

        # Преобразуем таблицу в формат для scatter
        B_values_all = []
        Delta_values_all = []
        p_max_values = []

        B_values_filtered = []
        Delta_values_filtered = []
        p_max_values_filtered = []

        for i, Delta_val in enumerate(self.pressure_table.columns):
            for j, B_val in enumerate(self.pressure_table.index):
                p_val = self.pressure_table.iloc[j, i]
                if pd.notna(p_val):
                    B_values_all.append(B_val)
                    Delta_values_all.append(Delta_val)
                    p_max_values.append(p_val)

                    # Только значения, где p_max <= p_a_max
                    if p_val <= p_threshold:
                        B_values_filtered.append(B_val)
                        Delta_values_filtered.append(Delta_val)
                        p_max_values_filtered.append(p_val)

        B_values_all = np.array(B_values_all)
        Delta_values_all = np.array(Delta_values_all)
        B_values_filtered = np.array(B_values_filtered)
        Delta_values_filtered = np.array(Delta_values_filtered)
        p_max_values_filtered = np.array(p_max_values_filtered)

        # Все точки серым цветом (включая те, что выше порога)
        ax.scatter(x=B_values_all, y=Delta_values_all, c='lightgray', marker=m, s=ms, edgecolor=mec, alpha=0.5)

        # Только допустимые точки с цветовой дифференциацией по давлению
        if len(p_max_values_filtered) > 0:
            sc = ax.scatter(x=B_values_filtered, y=Delta_values_filtered, c=p_max_values_filtered,
                            cmap=plt.cm.RdYlBu.reversed(), marker=m, s=ms, edgecolor=mec, vmin=0, vmax=p_threshold)

            # Цветовая шкала
            cb = plt.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label('$p_{max}$, МПа', fontsize=20, fontname='Times New Roman')
            cb.ax.tick_params(labelsize=16)

        # Настройка осей
        ax.set_xlabel('$B$', fontsize=20, fontname='Times New Roman')
        ax.set_ylabel('$\\Delta$, кг/м$^3$', fontsize=20, fontname='Times New Roman')
        ax.set_title('Максимальное давление $p_{max}$, МПа', fontsize=22, fontname='Times New Roman')

        # Толщина осей 2 пункта
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.tick_params(axis='both', width=2, length=6, labelsize=16)

        plt.tight_layout()

        if save:
            os.makedirs('results/Direct_Problem', exist_ok=True)
            filepath = os.path.join('results/Direct_Problem', filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"График сохранен в файл: {filepath}")

        plt.show()
        return fig, ax


# ============================================================================
# ОСНОВНОЙ КОД
# ============================================================================

if __name__ == "__main__":
    # Создаем объект параметров
    params = Cannon_System_Parameters()

    # ============================================================================
    # 1. ПОСТРОЕНИЕ ТАБЛИЦЫ ДАВЛЕНИЙ (Pressure_Rate_Table)
    # ============================================================================

    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ТАБЛИЦЫ МАКСИМАЛЬНЫХ ДАВЛЕНИЙ")
    print("=" * 80)

    prt = Pressure_Rate_Table(params)
    pressure_table = prt.build_table_fast()
    prt.display_table(pressure_table)
    prt.save_table(pressure_table, 'Pressure_Rate_Table.csv')

    # ============================================================================
    # 2. ПОСТРОЕНИЕ ТАБЛИЦЫ ТРЕБУЕМЫХ ПАРАМЕТРОВ (Required_Pressure_Rate_Table)
    # ============================================================================

    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ТАБЛИЦЫ ТРЕБУЕМЫХ ПАРАМЕТРОВ")
    print("=" * 80)

    rprt = Required_Pressure_Rate_Table(params, pressure_table)
    required_table = rprt.build_table()
    rprt.display_table(required_table)
    rprt.save_table(required_table, 'Required_Pressure_Rate_Table.csv')

    # ============================================================================
    # 3. ПОСТРОЕНИЕ ТОЧЕЧНОЙ ДИАГРАММЫ (Plotter_Pressure_Rate_Table)
    # ============================================================================

    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ТОЧЕЧНОЙ ДИАГРАММЫ ДАВЛЕНИЙ")
    print("=" * 80)

    plotter = Plotter_Pressure_Rate_Table(params, pressure_table)
    plotter.plot_scatter(save=True, filename='Pressure_Scatter.png')

    print("\nРабота программы завершена.")