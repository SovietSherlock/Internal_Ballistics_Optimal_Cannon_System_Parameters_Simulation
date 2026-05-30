"""
Модуль численного решения обыкновенных дифференциальных уравнений
методом Рунге-Кутты 4-го порядка

@Автор: <Барышев Савва>
Назначение: Расчет траектории ЛА на пассивном баллистическом участке
Версия: 1.0

Module for numerical technique for solving Ordinary Differential Equations (ODEs)
using 4th Order Runge-Kutta method

@author: <Savva Baryshev>
Purpose: Calculation of the aircraft trajectory in the passive ballistic section
Version: 1.0
"""

import numpy as np

def Runge_Kutta4(ODE_system, init_conditions, end_conditions, record, dt, t_0 = 0, max_steps = 100000):
    # функция, реализующая метод численного интегрирования системы обыкновенных дифференциальных уравнений, метод Рунге-Кутты 4-го порядка
    # ODE_system - система ОДУ математической модели, принимаемая на вход алгоритмом Runge_Kutta4
    # init_conditions - начальные условия переменных
    # end_conditions - конечные условия для переменных
    # record - запись выходных данных
    # dt - шаг интегрирования по независимой переменной

    t = np.zeros(max_steps)
    m = len(init_conditions) + 1
    n = len(record(t_0, init_conditions))
    result = np.zeros((max_steps, m+n))

    i = 0
    y = init_conditions
    t[i] = t_0
    result[i, 0] = t[i]
    result[i, 1: m] = y
    result[i, m: m + n] = record(t[i], y)

    while i+1<max_steps:
        k_1 = ODE_system(t[i], y)
        k_2 = ODE_system(t[i] + 0.5 * dt, y + k_1 * 0.5 * dt)
        k_3 = ODE_system(t[i] + 0.5 * dt, y + k_2 * 0.5 * dt)
        k_4 = ODE_system(t[i] + dt, y + k_3 * dt)
        i += 1
        y += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t[i] = t[i - 1] + dt
        result[i, 0] = t[i]
        result[i, 1: m] = y
        result[i, m: m + n] = record(t[i], y)
        if end_conditions(y) is True:
            return result[0: i + 1, :]
