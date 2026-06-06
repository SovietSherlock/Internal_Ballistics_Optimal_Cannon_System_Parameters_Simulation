"""
Модуль расчета приближенного решения обратной задачи внутренней баллистики.

@Автор: <Барышев Савва>
Назначение: Определение оптимальных проектных параметров АО
Версия: 1.0

Module for approximate solution for the inverse problem of internal ballistics.

@author: <Savva Baryshev>
Purpose: Cannon system parameters determination.
Version: 1.0
"""
from os import rename

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import copy
from ODE_solvers import *
# Настройка отображения с запятой в качестве десятичного разделителя
import locale

from SRC.Direct_Problem import Cannon_System_Parameters, Math_Model

try:
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
except:
    pass

class Init_Parameters:
    # класс входных параметров для решения обратной задачи внутренней баллистики

    def __init__(self):
        self.CSP = Cannon_System_Parameters
        self.MMDP = Math_Model

        ds = pd.read_csv('Required_Pressure_Rate_Table.csv', decimal=',')

        rename_parameters = {
            'Δ, кг/м³': 'delta',
            'B_a': 'B_a',
            'p_max, МПа': 'p_max',
            'p_m_e, МПа': 'p_m_e',
            'η_r_e': 'eta_r_e',
            'Λ_e': 'Lambda_e',
            'ζ': 'zeta'
        }
        ds = ds.rename(columns=rename_parameters)

        for column in ds.columns:
            self.__dict__[column] = ds[column].to_numpy()
