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
from Runge_Kutta4 import *


class TEST_Cannon_System_Parameters:
    # Класс входных проектных параметров АО для тестового задания

    def __init__(self):
        # Перезапись параметров из файла с исходными данными для дальнейшего анализа:
        ds = pd.read_csv('test_data.csv.csv', skiprows=[10]).set_index('var').to_dict()
        # Чтение файла "test_data.csv":
        self.__dict__.update(ds["value"].to_dict())

        self.v_pm = None # дульная скорость снаряда, м/с
        self.p_a_max = None # допустимое максимальное давление в канале ствола, Па

        self.Delta = np.arange(500, 875, 25)  # диапазон варьирования плотности заряжания, кг/м³

        if self.__dict__.get('kappa_2', 0) == 0:
            # Набор параметров для трубчатого пороха
            self.B_t = np.arange(0.75, 4.5, 0.25)
            self.eta_e_t = np.arange(0.25, 0.75, 0.05)

            # Многомерная сетка параметров для трубчатого пороха
            self.Delta_m, self.B_m, self.eta_e_m = np.meshgrid(
                self.Delta, self.B_t, self.eta_e_t, indexing='ij'
            )
            self.powder_type = 'tubular'  # индикатор трубчатого пороха

        else:
            # Набор параметров для семиканального пороха
            self.B_s = np.arange(0.5, 2.0, 0.1)
            self.eta_e_s = np.arange(0.5, 1.0, 0.05)

            # Многомерная сетка параметров для семиканального пороха
            self.Delta_m, self.B_m, self.eta_e_m = np.meshgrid(
                self.Delta, self.B_s, self.eta_e_s, indexing='ij'
            )
            self.powder_type = 'seven_core'  # индикатор семиканального пороха

        # self.Delta = np.arange(500, 875, 25) # диапазон варьирования плотности заряжания, кг/м³
        # self.B_t = np.arange(0.75, 4.5, 0.25) # диапазон варьирования параметра условий заряжания (параметра Дроздова) для трубчатого пороха
        # self.B_s = np.arange(0.5, 2.0, 0.1) # диапазон варьирования параметра условий заряжания (параметра Дроздова) для семиканального пороха
        # self.eta_e_t = np.arange(0.25, 0.75, 0.05) # диапазон варьирования параметра относительного положения снаряда в конце горения для трубчатого пороха
        # self.eta_e_s = np.arange(0.5, 1.0, 0.05) # диапазон варьирования параметра относительного положения снаряда в конце горения для трубчатого пороха
        #
        # self.Delta_t, self.B_t_m, self.eta_e_t_m = np.meshgrid(self.Delta, self.B_t, self.eta_e_t) # многомерные сетки параметров для трубчатого пороха
        # self.Delta_s, self.B_s_m, self.eta_e_s_m = np.meshgrid(self.Delta, self.B_s, self.eta_e_s) # многомерные сетки параметров для семиканального пороха

    def derived_parameters(self):
        # функция вычисления параметров, зависящих от начальных величин:

        return

class Math_Model:
    # класс, реализующий математическую модель приближенного решения ПЗВБ

    def __init__(self, parameters):
        self.p = parameters
        self.p.derived_parameters()

    # Предварительный период вычисления (параметры газоприхода в начале пиродинамического периода):

    def dzeta(self):
        # метод вычисления относительной массы воспламенительного состава:
        numerator = ((1 / self.p.Delta_m) - (1 / self.p.delta))
        denominator = (self.p.f_n / self.p.p_ign + self.p.b)
        return numerator / denominator

    def psi_s(self):
        # метод вычисления значения функции газоприхода в момент распада порохового зерна (для семиканального пороха):
        return self.p.kappa_1*(1 + self.p.lambda_1)

    def psi_0(self):
        numerator = self.p.p_0/self.p.f * (1/self.p.Delta_m - 1/8 - self.p.b*self.dzeta()) - self.dzeta()
        denominator = 1 - (self.p.p_0/self.p.f)*((1-self.p.b*self.p.delta)/self.p.delta)
        return numerator / denominator