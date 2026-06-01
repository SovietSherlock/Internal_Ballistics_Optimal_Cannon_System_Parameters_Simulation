"""
Тестовый модуль расчета приближенного решения прямой задачи внутренней баллистики.

@Автор: <Барышев Савва>
Назначение: Определение оптимальных проектных параметров АО
Версия: 1.0

Test module for approximate solution for the direct problem of internal ballistics.

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


class TEST_Cannon_System_Parameters:
    # Класс входных проектных параметров АО для тестового задания

    def __init__(self):
        # Перезапись параметров из файла с исходными данными для дальнейшего анализа:
        ds = pd.read_csv('test_data.csv', skiprows=[10]).set_index('var')
        # Чтение файла "test_data.csv":
        self.__dict__.update(ds["value"].to_dict())

        self.v_pm = 770 # дульная скорость снаряда, м/с
        self.p_a_max = 290e6 # допустимое максимальное давление в канале ствола, Па
        self.f = 1e6 # сила условного пороха, Дж/кг
        self.hi = 1.25 #

        self.dksi = 1e-3

        self.Delta = np.arange(500, 875, 25)  # диапазон варьирования плотности заряжания, кг/м³

        if self.__dict__.get('kappa_2', 0) == 0:
            # Набор параметров для трубчатого пороха
            self.B_t = np.arange(0.75, 4.5, 0.25) # диапазон варьирования параметра условий заряжания (параметра Дроздова) для трубчатого пороха
            self.eta_e_t = np.arange(0.25, 0.75, 0.05) # диапазон варьирования параметра относительного положения снаряда в конце горения для трубчатого пороха

            # Многомерная сетка параметров для трубчатого пороха
            self.Delta_m, self.B_m, self.eta_e_m = np.meshgrid(
                self.Delta, self.B_t, self.eta_e_t, indexing='ij'
            )
            self.powder_type = 'tubular'  # индикатор трубчатого пороха

        else:
            # Набор параметров для семиканального пороха
            self.B_s = np.arange(0.5, 2.0, 0.1) # диапазон варьирования параметра условий заряжания (параметра Дроздова) для семиканального пороха
            self.eta_e_s = np.arange(0.5, 1.0, 0.05) # диапазон варьирования параметра относительного положения снаряда в конце горения для трубчатого пороха

            # Многомерная сетка параметров для семиканального пороха
            self.Delta_m, self.B_m, self.eta_e_m = np.meshgrid(
                self.Delta, self.B_s, self.eta_e_s, indexing='ij'
            )
            self.powder_type = 'seven_core'  # индикатор семиканального пороха

    def derived_parameters(self):
        # функция вычисления параметров, зависящих от начальных величин:

        return

class Math_Model:
    # класс, реализующий математическую модель приближенного решения ПЗВБ

    def __init__(self, parameters):
        self.dx_dksi = None
        self.p = parameters
        self.p.derived_parameters()

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
        # метод вычисления относительной массы сгоревшей части заряда в пиродинамическрм периоде:
        before_decay = self.psi_0() + self.p.kappa_1*self.sigma_0()*ksi + self.p.kappa_1*self.p.lambda_1*ksi**2
        if self.p.kappa_2 != 0:
            after_decay = self.psi_s() + self.p.kappa_2*(ksi - self.ksi_s())*(1 + self.p.lambda_2*(ksi - self.ksi_s()))
        else: after_decay = 0
        return before_decay + after_decay

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


class Simulation:
    # Класс получения приближенных решений прямой задачи внутренней баллистики:

    def __init__(self, model: Math_Model):
        self.m = model

        # Получение значений параметров на пиростатическом периоде:
        result_pyrodynamic = RungeKutta4(self.m.ODE, self.m.init_conditions(), self.m.end_conditions, self.m.report, self.m.p.dksi, 0, 10000)

        pd.set_option('display.precision', 5)

        self.df_pyrodynamic = pd.DataFrame(result_pyrodynamic, columns=['ksi', 'Lambda', 'B', 'Delta', 'p_m', 'eta_r', 'psi_ksi'])

        # Получение значений параметров на адиабатическом периоде:
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


class PressureTable:
    def __init__(self, params: TEST_Cannon_System_Parameters):
        self.params = params

    def build_table_fast(self):
        """Быстрое построение таблицы (используя существующие сетки)"""

        # Получаем размеры сеток
        n_Delta = len(self.params.Delta)
        n_B = len(self.params.B_m[0, :, 0])

        # Создаем пустую таблицу
        pressure_matrix = np.zeros((n_B, n_Delta))

        # Перебираем все индексы
        for i in range(n_Delta):  # по Delta
            for j in range(n_B):  # по B
                # Берем значения из существующих сеток
                Delta_val = self.params.Delta_m[i, j, 0]
                B_val = self.params.B_m[i, j, 0]
                eta_e_val = self.params.eta_e_m[i, j, 0]

                # ВАЖНО: создаем КОПИЮ параметров для каждого расчета
                import copy
                params_copy = copy.deepcopy(self.params)

                # Обновляем параметры в копии
                params_copy.Delta_m = Delta_val
                params_copy.B_m = B_val
                params_copy.eta_e_m = eta_e_val

                # Расчет с использованием копии
                model = Math_Model(params_copy)
                sim = Simulation(model)

                if sim.df_full is not None:
                    pressure_matrix[j, i] = sim.df_full['p_m'].max() / 1e6
                else:
                    pressure_matrix[j, i] = np.nan

                print(f"i={i}, j={j}: Δ={Delta_val}, B={B_val:.2f}, p_max={pressure_matrix[j, i]:.2f} МПа")

        # Создаем DataFrame
        table = pd.DataFrame(
            pressure_matrix,
            index=np.round(self.params.B_m[0, :, 0], 3),
            columns=np.round(self.params.Delta, 0)
        )

        return table

    def display_table(self, table):
        """Красивый вывод таблицы в консоль"""
        print("\n" + "=" * 80)
        print("ТАБЛИЦА МАКСИМАЛЬНЫХ ДАВЛЕНИЙ p_m_max (МПа)")
        print("=" * 80)
        print(table.round(2))
        print("=" * 80)

    def save_table(self, table, filename):
        """Сохранение таблицы в CSV файл"""
        table.to_csv(filename)
        print(f"Таблица сохранена в файл: {filename}")

    def plot_heatmap(self, table):
        """Построение тепловой карты"""
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(table.values, cmap='hot', aspect='auto', origin='upper')

        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_xticklabels(table.columns)
        ax.set_yticklabels(table.index)

        ax.set_xlabel('Плотность заряжания Δ, кг/м³', fontsize=12)
        ax.set_ylabel('Параметр Дроздова B', fontsize=12)
        ax.set_title('Максимальное давление p_max, МПа', fontsize=14)

        plt.colorbar(im, ax=ax, label='Давление, МПа')
        plt.tight_layout()
        plt.show()

# Создаем объект параметров
params = TEST_Cannon_System_Parameters()

# Создаем объект для построения таблицы
pt = PressureTable(params)

# Строим таблицу
table = pt.build_table_fast()

# Выводим в консоль
pt.display_table(table)  # display_table, а не display_table_fast

# Сохраняем в файл
pt.save_table(table, 'p_max_table.csv')

# Строим тепловую карту
pt.plot_heatmap(table)