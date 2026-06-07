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

from SRC.Direct_Problem import Cannon_System_Parameters, Math_Model, Simulation

try:
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
except:
    pass

class Initial_Parameters:
    # класс входных параметров для решения обратной задачи внутренней баллистики

    def __init__(self):

        ds = pd.read_csv('results/Direct_Problem/Required_Pressure_Rate_Table.csv', decimal=',')

        rename_parameters = {
            'Δ, кг/м³': 'Delta',
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



class Ballistics_Solutions_Tables:
    # класс вывода таблиц баллистических решений для сочетаний (Δ, B*)

    def __init__(self):
        # Загрузка данных через Initial_Parameters
        self.IP = Initial_Parameters()

        # Параметры системы
        self.params = Cannon_System_Parameters()

        # Данные из Initial_Parameters
        self.Delta_values = self.IP.Delta  # значения Δ
        self.B_a_values = self.IP.B_a  # значения B_a

        # Определение eta_e_values в зависимости от типа пороха (по значению kappa_2)
        if hasattr(self.params, 'kappa_2') and self.params.kappa_2 != 0:
            # Семиканальный порох
            self.eta_e_values = self.params.eta_e_s
        else:
            # Трубчатый порох
            self.eta_e_values = self.params.eta_e_t

        # Размеры сетки
        n_Delta = len(self.Delta_values)
        n_eta = len(self.eta_e_values)

        # Инициализация таблиц
        self.omega_q_table = np.zeros((n_eta, n_Delta))
        self.l_m_d_table = np.zeros((n_eta, n_Delta))
        self.W_0_d3_table = np.zeros((n_eta, n_Delta))
        self.W_b_d3_table = np.zeros((n_eta, n_Delta))
        self.I_e_table = np.zeros((n_eta, n_Delta))
        self.C_Sl_table = np.zeros((n_eta, n_Delta))

        # Заполнение таблиц
        self._build_tables()

        # Поиск экстремумов
        self.min_W_b_d3 = np.min(self.W_b_d3_table[~np.isnan(self.W_b_d3_table)])
        self.max_C_Sl = np.max(self.C_Sl_table[~np.isnan(self.C_Sl_table)])
        self.min_W_b_idx = np.unravel_index(np.nanargmin(self.W_b_d3_table), self.W_b_d3_table.shape)
        self.max_C_Sl_idx = np.unravel_index(np.nanargmax(self.C_Sl_table), self.C_Sl_table.shape)

    def _build_tables(self):
        """Заполнение всех таблиц"""
        n_Delta = len(self.Delta_values)
        n_eta = len(self.eta_e_values)

        print("\n" + "=" * 80)
        print("РАСЧЕТ ТАБЛИЦ БАЛЛИСТИЧЕСКИХ РЕШЕНИЙ")
        print("=" * 80)

        for i, Delta_val in enumerate(self.Delta_values):
            B_a_val = self.B_a_values[i]

            for j, eta_e_val in enumerate(self.eta_e_values):
                try:
                    # Создание копии параметров
                    params_copy = copy.deepcopy(self.params)
                    params_copy.Delta_m = Delta_val
                    params_copy.B_m = B_a_val
                    params_copy.eta_e_m = eta_e_val

                    # Расчет
                    model = Math_Model(params_copy)
                    sim = Simulation(model)

                    # Получение значений в момент вылета
                    Lambda_m = sim.Lambda_m
                    eta_r_m = sim.eta_r_m

                    # Вычисление параметров
                    omega_q = model.omega_q(eta_r_m)
                    W_0 = model.W_0(eta_r_m)
                    l_m = model.l_m(Lambda_m, eta_r_m)
                    I_e = model.I_e(eta_r_m)
                    W_b = model.W_b(Lambda_m, eta_r_m)
                    C_Sl = model.C_Sl(Lambda_m, eta_r_m)

                    # Заполнение таблиц
                    self.omega_q_table[j, i] = omega_q
                    self.l_m_d_table[j, i] = l_m / self.params.d
                    self.W_0_d3_table[j, i] = W_0 / (self.params.d ** 3)
                    self.W_b_d3_table[j, i] = W_b / (self.params.d ** 3)
                    self.I_e_table[j, i] = I_e / 1e6  # в МДж/кг
                    self.C_Sl_table[j, i] = C_Sl

                    print(f"Δ={Delta_val:.0f}, B_a={B_a_val:.6f}, η_e={eta_e_val:.2f}: "
                          f"ω/q={omega_q:.4f}, W_b/d³={W_b / self.params.d ** 3:.4f}, C_Sl={C_Sl:.4f}")

                except Exception as e:
                    print(f"Ошибка для Δ={Delta_val}, η_e={eta_e_val}: {str(e)[:80]}")
                    self.omega_q_table[j, i] = np.nan
                    self.l_m_d_table[j, i] = np.nan
                    self.W_0_d3_table[j, i] = np.nan
                    self.W_b_d3_table[j, i] = np.nan
                    self.I_e_table[j, i] = np.nan
                    self.C_Sl_table[j, i] = np.nan

    def _create_dataframe(self, data, name):
        """Создание DataFrame из данных"""
        df = pd.DataFrame(
            data,
            index=np.round(self.eta_e_values, 3),
            columns=np.round(self.Delta_values, 0)
        )
        df.index.name = 'η_e \\ Δ'
        return df

    def display_omega_q_table(self):
        """Вывод таблицы ω/q"""
        df = self._create_dataframe(self.omega_q_table, 'omega_q')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА ω/q (относительная масса порохового заряда)")
        print("=" * 80)
        print(df.round(4))
        print("=" * 80)
        return df

    def display_l_m_d_table(self):
        """Вывод таблицы l_m/d"""
        df = self._create_dataframe(self.l_m_d_table, 'l_m/d')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА l_m/d (длина ствола в калибрах)")
        print("=" * 80)
        print(df.round(2))
        print("=" * 80)
        return df

    def display_W_0_d3_table(self):
        """Вывод таблицы W_0/d³"""
        df = self._create_dataframe(self.W_0_d3_table, 'W_0/d³')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА W_0/d³ (объем каморы в калибрах)")
        print("=" * 80)
        print(df.round(4))
        print("=" * 80)
        return df

    def display_W_b_d3_table(self):
        """Вывод таблицы W_b/d³ с отмеченным минимумом"""
        df = self._create_dataframe(self.W_b_d3_table, 'W_b/d³')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА W_b/d³ (суммарный объем канала ствола в калибрах)")
        print("=" * 80)

        df_display = df.copy().round(4)
        min_row, min_col = self.min_W_b_idx
        min_val = df_display.iloc[min_row, min_col]

        print("η_e \\ Δ", end="")
        for col in df_display.columns:
            print(f"{col:10.0f}", end="")
        print()
        print("-" * (10 + 10 * len(df_display.columns)))

        for idx in df_display.index:
            print(f"{idx:.2f}", end="")
            for col in df_display.columns:
                val = df_display.loc[idx, col]
                if pd.notna(val):
                    if abs(val - min_val) < 1e-6 and df_display.index.get_loc(
                            idx) == min_row and df_display.columns.get_loc(col) == min_col:
                        print(f"{val:9.4f}*", end=" ")
                    else:
                        print(f"{val:10.4f}", end=" ")
                else:
                    print(f"{'   -   '}", end=" ")
            print()

        print("=" * 80)
        print(f"Минимальное значение W_b/d³ = {min_val:.4f} "
              f"(Δ={df.columns[min_col]:.0f}, η_e={df.index[min_row]:.2f})")
        print("=" * 80)
        return df

    def display_I_e_table(self):
        """Вывод таблицы I_e (импульс пороха)"""
        df = self._create_dataframe(self.I_e_table, 'I_e, МДж/кг')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА I_e (импульс пороха, МДж/кг)")
        print("=" * 80)
        print(df.round(2))
        print("=" * 80)
        return df

    def display_C_Sl_table(self):
        """Вывод таблицы C_Sl с отмеченным максимумом"""
        df = self._create_dataframe(self.C_Sl_table, 'C_Sl')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА C_Sl (критерий Слухоцкого)")
        print("=" * 80)

        df_display = df.copy().round(4)
        max_row, max_col = self.max_C_Sl_idx
        max_val = df_display.iloc[max_row, max_col]

        print("η_e \\ Δ", end="")
        for col in df_display.columns:
            print(f"{col:10.0f}", end="")
        print()
        print("-" * (10 + 10 * len(df_display.columns)))

        for idx in df_display.index:
            print(f"{idx:.2f}", end="")
            for col in df_display.columns:
                val = df_display.loc[idx, col]
                if pd.notna(val):
                    if abs(val - max_val) < 1e-6 and df_display.index.get_loc(
                            idx) == max_row and df_display.columns.get_loc(col) == max_col:
                        print(f"{val:9.4f}*", end=" ")
                    else:
                        print(f"{val:10.4f}", end=" ")
                else:
                    print(f"{'   -   '}", end=" ")
            print()

        print("=" * 80)
        print(f"Максимальное значение C_Sl = {max_val:.4f} "
              f"(Δ={df.columns[max_col]:.0f}, η_e={df.index[max_row]:.2f})")
        print("=" * 80)
        return df

    def save_all_tables(self):
        """Сохранение всех таблиц в CSV файлы с запятой как десятичным разделителем"""
        os.makedirs('results/Inverse_Problem', exist_ok=True)

        tables = {
            'omega_q_table.csv': (self.omega_q_table, 'ω/q'),
            'l_m_d_table.csv': (self.l_m_d_table, 'l_m/d'),
            'W_0_d3_table.csv': (self.W_0_d3_table, 'W_0/d³'),
            'W_b_d3_table.csv': (self.W_b_d3_table, 'W_b/d³'),
            'I_e_table.csv': (self.I_e_table, 'I_e, МДж/кг'),
            'C_Sl_table.csv': (self.C_Sl_table, 'C_Sl')
        }

        for filename, (data, name) in tables.items():
            df = self._create_dataframe(data, name)

            # Форматирование с запятой
            df_comma = df.copy()
            for col in df_comma.columns:
                df_comma[col] = df_comma[col].apply(lambda x: f"{x:.6f}".replace('.', ',') if pd.notna(x) else "")
            df_comma.index = df_comma.index.map(lambda x: f"{x:.2f}".replace('.', ','))

            # Сохранение
            filepath = os.path.join('results/Inverse_Problem', filename)
            df_comma.to_csv(filepath, index=True, encoding='utf-8-sig')

            print(f"Таблица сохранена: {filepath}")


# ============================================================================
# ОСНОВНОЙ КОД
# ============================================================================

if __name__ == "__main__":
    bst = Ballistics_Solutions_Tables()

    # Вывод всех таблиц
    bst.display_omega_q_table()
    bst.display_l_m_d_table()
    bst.display_W_0_d3_table()
    bst.display_W_b_d3_table()
    bst.display_I_e_table()
    bst.display_C_Sl_table()

    # Сохранение всех таблиц
    bst.save_all_tables()

    print("\nРабота программы завершена.")