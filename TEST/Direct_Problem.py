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
        ds = pd.read_csv('СМ6-62 Барышев СА 2026-27-05.csv', skiprows=[10]).set_index('var').to_dict()
        # Чтение файла "test_data.csv":
        # 1. Открытие csv файла с начальными данными, игнорирование 11 строки (строка с обозначением пороха),
        # 2. Преобразование индексов строк в обозначение параметров таблицы
        # 3. Преобразование таблицы в словарь ds = {'value': {'параметр_1': значение_1, 'параметр_2': значение_2, ...}}
        vals = ds['value']
        # извлечение значений по ключу 'value' в отдельный словарь vals = {'параметр_1': значение_1, 'параметр_2': значение_2, ...}
        # Основные исходные данные:
        self.d = vals['d'] # калибр орудия, м
        self.q = vals['q']
        self.v_pm = None
