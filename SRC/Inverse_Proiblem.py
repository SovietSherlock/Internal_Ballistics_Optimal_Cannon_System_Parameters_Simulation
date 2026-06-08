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

from pyballistics import get_db_powder

from ODE_solvers import *
import pyballistics

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
        # Заполнение всех таблиц:
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
        # Создание DataFrame:
        df = pd.DataFrame(
            data,
            index=np.round(self.eta_e_values, 3),
            columns=np.round(self.Delta_values, 0)
        )
        df.index.name = 'η_e \\ Δ'
        return df

    def display_omega_q_table(self):
        # Вывод таблицы ω/q:
        df = self._create_dataframe(self.omega_q_table, 'omega_q')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА ω/q (относительная масса порохового заряда)")
        print("=" * 80)
        print(df.round(4))
        print("=" * 80)
        return df

    def display_l_m_d_table(self):
        # Вывод таблицы l_m/d:
        df = self._create_dataframe(self.l_m_d_table, 'l_m/d')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА l_m/d (длина ствола в калибрах)")
        print("=" * 80)
        print(df.round(2))
        print("=" * 80)
        return df

    def display_W_0_d3_table(self):
        # Вывод таблицы W_0/d³:
        df = self._create_dataframe(self.W_0_d3_table, 'W_0/d³')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА W_0/d³ (объем каморы в калибрах)")
        print("=" * 80)
        print(df.round(4))
        print("=" * 80)
        return df

    def display_W_b_d3_table(self):
        # Вывод таблицы W_b/d³ с отмеченным минимумом:
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
        # Вывод таблицы I_e:
        df = self._create_dataframe(self.I_e_table, 'I_e, МДж/кг')
        print("\n" + "=" * 80)
        print("ТАБЛИЦА I_e (импульс пороха, МДж/кг)")
        print("=" * 80)
        print(df.round(2))
        print("=" * 80)
        return df

    def display_C_Sl_table(self):
        # Вывод таблицы C_Sl с отмеченным максимумом:
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
        # Сохранение всех таблиц в CSV файлы:
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

    def plot_optimization_diagram(self, save=False, filename='Optimization_Diagram.png'):
        # Построение диаграммы баллистических решений в координатах (ω/q, Δ)
        # с отмеченными точками минимума W_b/d³ и максимума C_Sl

        # Настройка шрифтов
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

        # Размер фигуры 10x10 дюймов
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=(1, 1, 1))

        # Параметры маркеров
        m = 'o'  # маркер - кружок
        ms = 256  # размер маркера
        mec = 'k'  # цвет обводки маркера

        # Сбор данных для всех точек
        omega_q_all = []
        Delta_all = []
        C_Sl_all = []
        W_b_all = []

        for i, Delta_val in enumerate(self.Delta_values):
            for j, eta_val in enumerate(self.eta_e_values):
                omega_q_val = self.omega_q_table[j, i]
                W_b_val = self.W_b_d3_table[j, i]
                C_Sl_val = self.C_Sl_table[j, i]

                if pd.notna(omega_q_val) and pd.notna(Delta_val):
                    omega_q_all.append(omega_q_val)
                    Delta_all.append(Delta_val)
                    C_Sl_all.append(C_Sl_val)
                    W_b_all.append(W_b_val)

        omega_q_all = np.array(omega_q_all)
        Delta_all = np.array(Delta_all)
        C_Sl_all = np.array(C_Sl_all)
        W_b_all = np.array(W_b_all)

        # Все точки с цветовой дифференциацией по C_Sl
        sc = ax.scatter(omega_q_all, Delta_all, c=C_Sl_all,
                        cmap=plt.cm.RdYlBu.reversed(), marker=m,
                        s=ms, edgecolor=mec, vmin=0)

        # Нахождение и отметка точки с максимальным C_Sl
        max_C_Sl_idx = np.nanargmax(C_Sl_all)
        omega_q_max_C = omega_q_all[max_C_Sl_idx]
        Delta_max_C = Delta_all[max_C_Sl_idx]
        max_C_val = C_Sl_all[max_C_Sl_idx]

        ax.scatter(omega_q_max_C, Delta_max_C, marker='*', s=256,
                   c='gold', edgecolor='k', linewidth=1,
                   label=f'max $C_{{Sl}}$ = {max_C_val:.4f}')

        # Нахождение и отметка точки с минимальным W_b/d³
        min_W_b_idx = np.nanargmin(W_b_all)
        omega_q_min_W = omega_q_all[min_W_b_idx]
        Delta_min_W = Delta_all[min_W_b_idx]
        min_W_val = W_b_all[min_W_b_idx]

        ax.scatter(omega_q_min_W, Delta_min_W, marker='*', s=256,
                   c='lime', edgecolor='k', linewidth=1,
                   label=f'min $W_b/d^3$ = {min_W_val:.4f}')

        # Цветовая шкала
        cb = plt.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label('$C_{Sl}$', fontsize=20, fontname='Times New Roman')
        cb.ax.tick_params(labelsize=16)

        # Настройка осей
        ax.set_xlabel('$\\omega/q$', fontsize=20, fontname='Times New Roman')
        ax.set_ylabel('$\\Delta$, кг/м$^3$', fontsize=20, fontname='Times New Roman')
        ax.set_title('Баллистические решения обратной задачи', fontsize=22, fontname='Times New Roman')

        # Толщина осей 2 пункта
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.tick_params(axis='both', width=2, length=6, labelsize=16)

        # Легенда
        ax.legend(loc='best', fontsize=14, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        if save:
            os.makedirs('results/Inverse_Problem', exist_ok=True)
            filepath = os.path.join('results/Inverse_Problem', filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Диаграмма сохранена в файл: {filepath}")

        plt.show()
        return fig, ax


class Individual_Ballistics_Solutions:
    # класс определения баллистических решений для индивидуальных марок порохов

    def __init__(self):
        self.powder_list = ['18/1 тр БП', 'НДТ-3 19/1', 'ДГ-3 20/1', 'НДТ-2 19/1', '180/57', '152/57 БП', '152/57 Ш']
        self.powders_data = []
        self._get_all_powders_data()

        # Параметры системы
        self.params = Cannon_System_Parameters()

        # Загрузка таблицы Required_Pressure_Rate_Table для получения B_a
        self.df_required = pd.read_csv('results/Direct_Problem/Required_Pressure_Rate_Table.csv', encoding='utf-8-sig')

        # Результаты расчетов для каждого пороха
        self.powders_results = []
        self._calculate_all_powders()

    def get_powder_parameters(self, powder_name: str) -> dict:
        """
        Получение параметров индивидуальной марки пороха из библиотеки pyballistics

        Параметры:
        powder_name: str - название пороха

        Возвращает:
        dict - словарь с параметрами пороха
        """
        try:
            powder_data = pyballistics.get_db_powder(powder_name)

            # Извлекаем нужные параметры
            parameters = {
                'name': powder_data.get('name', powder_name),
                'I_e': powder_data.get('I_e', 0),
                'f': powder_data.get('f', 0),
                'k': powder_data.get('k', 0),
                'b': powder_data.get('b', 0),
                'delta': powder_data.get('delta', 0),
                'z_e': powder_data.get('z_e', 0),
                'kappa_1': powder_data.get('kappa_1', 0),
                'lambda_1': powder_data.get('lambda_1', 0),
                'kappa_2': powder_data.get('kappa_2', 0),
                'lambda_2': powder_data.get('lambda_2', 0),
            }
            return parameters

        except Exception as e:
            print(f"Ошибка при получении параметров пороха '{powder_name}': {e}")
            return None

    def _get_all_powders_data(self):
        """Получение данных для всех порохов из списка"""
        print("\n" + "=" * 80)
        print("ЗАГРУЗКА ПАРАМЕТРОВ ПОРОХОВ ИЗ БИБЛИОТЕКИ pyballistics")
        print("=" * 80)

        # Правильные названия порохов в библиотеке pyballistics
        powder_names_in_db = {
            '18/1 тр БП': '18/1 тр БП',
            'НДТ-3 19/1': 'НДТ-3 19/1',
            'ДГ-3 20/1': 'ДГ-3 20/1',
            'НДТ-2 19/1': 'НДТ-2 19/1',
            '180/57': '180/57',
            '152/57 БП': '152/57 БП',
            '152/57 Ш': '152/57 Ш'
        }

        for powder_name in self.powder_list:
            db_name = powder_names_in_db.get(powder_name, powder_name)
            params = self.get_powder_parameters(db_name)
            if params is not None:
                self.powders_data.append(params)
                print(f"\nПорох: {params['name']}")
                print(f"  I_e = {params['I_e']:.2f} Па·с")
                print(f"  f   = {params['f']:.2f} Дж/кг")
                print(f"  k   = {params['k']:.4f}")
                print(f"  b   = {params['b']:.6f} м³/кг")
                print(f"  δ   = {params['delta']:.2f} кг/м³")
                print(f"  z_e = {params['z_e']:.4f}")
                print(f"  κ₁  = {params['kappa_1']:.4f}")
                print(f"  λ₁  = {params['lambda_1']:.4f}")
                print(f"  κ₂  = {params['kappa_2']:.4f}")
                print(f"  λ₂  = {params['lambda_2']:.4f}")
            else:
                print(f"\nНе удалось загрузить параметры для пороха: {powder_name}")

        print("\n" + "=" * 80)
        print(f"Загружено {len(self.powders_data)} порохов из {len(self.powder_list)}")
        print("=" * 80)

    def display_powders_table(self):
        """Вывод таблицы параметров всех порохов из pyballistics"""
        if not self.powders_data:
            print("Нет данных о порохах")
            return None

        df = pd.DataFrame(self.powders_data)

        # Выбираем только нужные колонки
        columns_to_display = ['name', 'I_e', 'f', 'k', 'b', 'delta', 'z_e', 'kappa_1', 'lambda_1', 'kappa_2',
                              'lambda_2']
        df_display = df[columns_to_display].copy()

        print("\n" + "=" * 100)
        print("ТАБЛИЦА ПАРАМЕТРОВ ПОРОХОВ ИЗ БИБЛИОТЕКИ pyballistics")
        print("=" * 100)

        # Форматирование для вывода
        df_display['I_e'] = (df_display['I_e']/1e6).apply(lambda x: f"{x:.3f}")
        df_display['f'] = (df_display['f']/1e6).apply(lambda x: f"{x:.3f}")
        df_display['b'] = (df_display['b']*1e3).apply(lambda x: f"{x:.3f}")
        df_display['delta'] = df_display['delta'].apply(lambda x: f"{x:.0f}")
        df_display['z_e'] = df_display['z_e'].apply(lambda x: f"{x:.3f}")
        df_display['kappa_1'] = df_display['kappa_1'].apply(lambda x: f"{x:.3f}")
        df_display['lambda_1'] = df_display['lambda_1'].apply(lambda x: f"{x:.3f}")
        df_display['kappa_2'] = df_display['kappa_2'].apply(lambda x: f"{x:.3f}")
        df_display['lambda_2'] = df_display['lambda_2'].apply(lambda x: f"{x:.3f}")

        # Переименовываем колонки для красивого вывода
        df_display = df_display.rename(columns={
            'name': 'Марка пороха',
            'I_e': 'I_e, Па·с',
            'f': 'f, Дж/кг',
            'k': 'k',
            'b': 'b, дм³/кг',
            'delta': 'δ, кг/м³',
            'z_e': 'z_e',
            'kappa_1': 'κ₁',
            'lambda_1': 'λ₁',
            'kappa_2': 'κ₂',
            'lambda_2': 'λ₂'
        })

        print(df_display.to_string(index=False))
        print("=" * 100)

        return df_display

    def save_powders_table(self, filename='Powder_Data.csv'):
        """Сохранение таблицы параметров порохов из pyballistics в CSV файл"""
        if not self.powders_data:
            print("Нет данных для сохранения")
            return

        os.makedirs('results/Inverse_Problem', exist_ok=True)

        # Создаем DataFrame только с нужными параметрами
        df = pd.DataFrame(self.powders_data)

        # Выбираем только нужные колонки
        columns_to_save = ['name', 'I_e', 'f', 'k', 'b', 'delta', 'z_e', 'kappa_1', 'lambda_1', 'kappa_2', 'lambda_2']
        df = df[columns_to_save]

        filepath = os.path.join('results/Inverse_Problem', filename)

        # Сохранение с запятой как десятичным разделителем
        df_comma = df.copy()
        for col in df_comma.columns:
            if col != 'name' and df_comma[col].dtype in ['float64', 'int64']:
                df_comma[col] = df_comma[col].apply(lambda x: f"{x:.6f}".replace('.', ',') if pd.notna(x) else "")

        df_comma.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nТаблица параметров порохов сохранена: {filepath}")

    def get_B_a_for_Delta(self, Delta_val: float) -> float:
        """Получение B_a для заданного Delta из таблицы Required_Pressure_Rate_Table"""
        # Преобразуем колонку в float для корректного сравнения
        Delta_col = self.df_required['Δ, кг/м³'].astype(float)

        # Находим индекс с точным совпадением
        mask = Delta_col == Delta_val
        if mask.any():
            B_a_str = self.df_required.loc[mask, 'B_a'].iloc[0]
        else:
            # Если точного совпадения нет, находим ближайшее
            idx = (Delta_col - Delta_val).abs().argmin()
            B_a_str = self.df_required['B_a'].iloc[idx]
            print(f"  Для Δ={Delta_val} используется значение B_a из Δ={Delta_col.iloc[idx]:.0f}")

        if isinstance(B_a_str, str):
            B_a_str = B_a_str.replace(',', '.')
        return float(B_a_str)

    def calculate_single_powder(self, powder_params: dict, Delta_val: float, eta_e_val: float = 0.7) -> dict:
        """
        Расчет баллистического решения для конкретного пороха

        Параметры:
        powder_params: dict - параметры пороха
        Delta_val: float - плотность заряжания
        eta_e_val: float - относительное положение снаряда

        Возвращает:
        dict - результаты расчета
        """
        # Получаем B_a для заданного Delta
        B_a_val = self.get_B_a_for_Delta(Delta_val)
        if B_a_val is None:
            return None

        # Создаем копию параметров системы
        params_copy = copy.deepcopy(self.params)

        # Заменяем параметры на параметры выбранного пороха
        params_copy.I_e = powder_params['I_e']
        params_copy.f = powder_params['f']
        params_copy.k = powder_params['k']
        params_copy.b = powder_params['b']
        params_copy.delta = powder_params['delta']
        params_copy.z_e = powder_params['z_e']
        params_copy.kappa_1 = powder_params['kappa_1']
        params_copy.lambda_1 = powder_params['lambda_1']
        params_copy.kappa_2 = powder_params['kappa_2']
        params_copy.lambda_2 = powder_params['lambda_2']

        # Задаем плотность заряжания
        params_copy.Delta_m = Delta_val
        params_copy.B_m = B_a_val
        params_copy.eta_e_m = eta_e_val

        # Выполняем расчет
        model = Math_Model(params_copy)
        sim = Simulation(model)

        # Результаты
        result = {
            'powder_name': powder_params['name'],
            'Delta': Delta_val,
            'B_a': B_a_val,
            'eta_e': eta_e_val,
            'Lambda_m': sim.Lambda_m,
            'p_m_m': sim.p_m_m / 1e6,
            'eta_r_m': sim.eta_r_m,
            'omega_q': model.omega_q(sim.eta_r_m),
            'W_0': model.W_0(sim.eta_r_m),
            'l_m': model.l_m(sim.Lambda_m, sim.eta_r_m),
            'I_e_calc': model.I_e(sim.eta_r_m) / 1e6,
            'W_b': model.W_b(sim.Lambda_m, sim.eta_r_m),
            'C_Sl': model.C_Sl(sim.Lambda_m, sim.eta_r_m)
        }

        return result

    def _calculate_all_powders(self):
        """Расчет для всех порохов при различных Delta и eta_e"""
        print("\n" + "=" * 80)
        print("РАСЧЕТ БАЛЛИСТИЧЕСКИХ РЕШЕНИЙ ДЛЯ ИНДИВИДУАЛЬНЫХ ПОРОХОВ")
        print("=" * 80)

        # Получаем уникальные значения Delta из многомерного массива (первый столбец, первая строка)
        Delta_values = self.params.Delta_m[:, 0, 0]  # уникальные значения Δ

        # Получаем уникальные значения eta_e из многомерного массива (первая строка, первый столбец по eta_e)
        if hasattr(self.params, 'kappa_2') and self.params.kappa_2 != 0:
            # Для семиканального пороха
            eta_e_values = self.params.eta_e_m[0, 0, :]  # уникальные значения η_e
        else:
            # Для трубчатого пороха
            eta_e_values = self.params.eta_e_m[0, 0, :]  # уникальные значения η_e

        print(f"Диапазон Δ: {Delta_values[0]:.0f} - {Delta_values[-1]:.0f} ({len(Delta_values)} значений)")
        print(f"Диапазон η_e: {eta_e_values[0]:.2f} - {eta_e_values[-1]:.2f} ({len(eta_e_values)} значений)")

        for powder in self.powders_data:
            powder_results = []
            print(f"\n--- Расчет для пороха: {powder['name']} ---")

            for Delta_val in Delta_values:
                for eta_e_val in eta_e_values:
                    result = self.calculate_single_powder(powder, Delta_val, eta_e_val)
                    if result is not None:
                        powder_results.append(result)
                        print(
                            f"  Δ={Delta_val:.0f}, η_e={eta_e_val:.2f}: ω/q={result['omega_q']:.4f}, C_Sl={result['C_Sl']:.4f}")

            if powder_results:
                self.powders_results.append({
                    'powder': powder['name'],
                    'results': powder_results
                })

    def plot_individual_optimization_diagram(self, powder_name: str, save=False, filename=None):
        """
        Построение диаграммы баллистических решений для индивидуального пороха
        в координатах (ω/q, Δ) с отмеченной точкой максимума C_Sl
        """
        # Находим результаты для указанного пороха
        powder_result = None
        for pr in self.powders_results:
            if pr['powder'] == powder_name:
                powder_result = pr
                break

        if powder_result is None:
            print(f"Не найдены результаты для пороха '{powder_name}'")
            return None

        # Настройка шрифтов
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

        # Размер фигуры 12x10 дюймов
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=(1, 1, 1))

        # Параметры маркеров
        m = 'o'
        ms = 256
        mec = 'k'

        # Сбор данных
        omega_q_all = []
        Delta_all = []
        C_Sl_all = []

        for result in powder_result['results']:
            omega_q_all.append(result['omega_q'])
            Delta_all.append(result['Delta'])
            C_Sl_all.append(result['C_Sl'])

        omega_q_all = np.array(omega_q_all)
        Delta_all = np.array(Delta_all)
        C_Sl_all = np.array(C_Sl_all)

        # Все точки с цветовой дифференциацией по C_Sl
        sc = ax.scatter(omega_q_all, Delta_all, c=C_Sl_all,
                        cmap=plt.cm.RdYlBu.reversed(), marker=m,
                        s=ms, edgecolor=mec, vmin=0)

        # Нахождение и отметка точки с максимальным C_Sl
        max_C_Sl_idx = np.nanargmax(C_Sl_all)
        omega_q_max_C = omega_q_all[max_C_Sl_idx]
        Delta_max_C = Delta_all[max_C_Sl_idx]
        max_C_val = C_Sl_all[max_C_Sl_idx]

        ax.scatter(omega_q_max_C, Delta_max_C, marker='*', s=256,
                   c='gold', edgecolor='k', linewidth=1,
                   label=f'max $C_{{Sl}}$ = {max_C_val:.4f}')

        # Цветовая шкала
        cb = plt.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label('$C_{Sl}$', fontsize=20, fontname='Times New Roman')
        cb.ax.tick_params(labelsize=16)

        # Настройка осей
        ax.set_xlabel('$\\omega/q$', fontsize=20, fontname='Times New Roman')
        ax.set_ylabel('$\\Delta$, кг/м$^3$', fontsize=20, fontname='Times New Roman')
        ax.set_title(f'Баллистические решения\n для пороха {powder_name}', fontsize=22, fontname='Times New Roman')

        # Толщина осей 2 пункта
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.tick_params(axis='both', width=2, length=6, labelsize=16)

        # Легенда
        ax.legend(loc='best', fontsize=14, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        if save:
            os.makedirs('results/Inverse_Problem', exist_ok=True)
            if filename is None:
                safe_name = powder_name.replace('/', '_').replace(' ', '_')
                filename = f'Optimization_Diagram_{safe_name}.png'
            filepath = os.path.join('results/Inverse_Problem', filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Диаграмма для пороха '{powder_name}' сохранена: {filepath}")

        plt.show()
        return fig, ax

    def plot_all_individual_diagrams(self, save=False):
        """Построение диаграмм для всех индивидуальных порохов"""
        for powder in self.powders_data:
            self.plot_individual_optimization_diagram(powder['name'], save=save)

    def display_max_C_Sl_table(self):
        """Вывод таблицы для каждой марки пороха значений параметров в точке максимума C_Sl"""
        print("\n" + "=" * 100)
        print("ТАБЛИЦА ПАРАМЕТРОВ В ТОЧКЕ МАКСИМУМА C_Sl ДЛЯ КАЖДОЙ МАРКИ ПОРОХА")
        print("=" * 100)

        # Список для сбора данных
        data = []

        for powder_result in self.powders_results:
            powder_name = powder_result['powder']
            results = powder_result['results']

            # Находим результат с максимальным C_Sl
            max_C_Sl_val = -np.inf
            best_result = None

            for idx, result in enumerate(results):
                if result['C_Sl'] > max_C_Sl_val:
                    max_C_Sl_val = result['C_Sl']
                    best_result = result

            if best_result is not None:
                # Вычисляем реальные значения W_0 и l_m
                d = self.params.d  # калибр, м
                W_0_real = best_result['W_0']  # уже в м³ (из метода W_0)
                l_m_real = best_result['l_m']  # уже в м (из метода l_m)

                # p_max должно быть равно self.params.p_a_max / 1e6 (290 МПа)
                p_max_target = self.params.p_a_max / 1e6

                data.append({
                    'Марка пороха': powder_name,
                    'Δ, кг/м³': best_result['Delta'],
                    'η_e': best_result['eta_e'],
                    'B_a': best_result['B_a'],
                    'ω/q': best_result['omega_q'],
                    'W_0, м³': W_0_real,
                    'l_m, м': l_m_real,
                    'p_max, МПа': p_max_target,
                    'v_pm, м/с': self.params.v_pm,
                    'C_Sl': best_result['C_Sl']
                })

                print(f"\n--- {powder_name} ---")
                print(f"  Δ       = {best_result['Delta']:.0f} кг/м³")
                print(f"  η_e     = {best_result['eta_e']:.2f}")
                print(f"  B_a     = {best_result['B_a']:.6f}")
                print(f"  ω/q     = {best_result['omega_q']:.4f}")
                print(f"  W_0     = {W_0_real:.6f} м³")
                print(f"  l_m     = {l_m_real:.4f} м")
                print(f"  p_max   = {p_max_target:.2f} МПа")
                print(f"  v_pm    = {self.params.v_pm:.0f} м/с")
                print(f"  C_Sl    = {best_result['C_Sl']:.4f}")

        # Создаем DataFrame
        df = pd.DataFrame(data)

        print("\n" + "=" * 100)
        print("СВОДНАЯ ТАБЛИЦА ПАРАМЕТРОВ В ТОЧКЕ МАКСИМУМА C_Sl")
        print("=" * 100)

        # Форматируем вывод
        df_display = df.copy()
        for col in df_display.columns:
            if col not in ['Марка пороха']:
                if col in ['Δ, кг/м³', 'η_e', 'B_a', 'ω/q', 'C_Sl']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
                elif col in ['W_0, м³']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.6f}")
                elif col in ['l_m, м']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
                elif col in ['p_max, МПа', 'v_pm, м/с']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.0f}")

        print(df_display.to_string(index=False))
        print("=" * 100)

        return df

    def save_max_C_Sl_table(self, filename='Max_C_Sl_Parameters.csv'):
        """Сохранение таблицы параметров в точке максимума C_Sl в CSV файл"""
        if not self.powders_results:
            print("Нет данных для сохранения")
            return

        os.makedirs('results/Inverse_Problem', exist_ok=True)

        # Собираем данные
        data = []
        for powder_result in self.powders_results:
            powder_name = powder_result['powder']
            results = powder_result['results']

            # Находим результат с максимальным C_Sl
            max_C_Sl_val = -np.inf
            best_result = None

            for result in results:
                if result['C_Sl'] > max_C_Sl_val:
                    max_C_Sl_val = result['C_Sl']
                    best_result = result

            if best_result is not None:
                # Вычисляем реальные значения W_0 и l_m
                W_0_real = best_result['W_0']  # уже в м³
                l_m_real = best_result['l_m']  # уже в м
                p_max_target = self.params.p_a_max / 1e6

                data.append({
                    'Марка пороха': powder_name,
                    'Δ, кг/м³': best_result['Delta'],
                    'η_e': best_result['eta_e'],
                    'B_a': best_result['B_a'],
                    'ω/q': best_result['omega_q'],
                    'W_0, м³': W_0_real,
                    'l_m, м': l_m_real,
                    'p_max, МПа': p_max_target,
                    'v_pm, м/с': self.params.v_pm,
                    'C_Sl': best_result['C_Sl']
                })

        df = pd.DataFrame(data)
        filepath = os.path.join('results/Inverse_Problem', filename)

        # Сохранение с запятой как десятичным разделителем
        df_comma = df.copy()
        for col in df_comma.columns:
            if col != 'Марка пороха':
                if col in ['W_0, м³']:
                    df_comma[col] = df_comma[col].apply(lambda x: f"{x:.6f}".replace('.', ',') if pd.notna(x) else "")
                elif col in ['l_m, м']:
                    df_comma[col] = df_comma[col].apply(lambda x: f"{x:.4f}".replace('.', ',') if pd.notna(x) else "")
                else:
                    df_comma[col] = df_comma[col].apply(lambda x: f"{x:.6f}".replace('.', ',') if pd.notna(x) else "")

        df_comma.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nТаблица параметров в точке максимума C_Sl сохранена: {filepath}")


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

    # Построение диаграммы оптимизации
    bst.plot_optimization_diagram(save=True, filename='Optimization_Diagram.png')

    print("\n" + "=" * 80)
    print("РАСЧЕТ ДЛЯ ИНДИВИДУАЛЬНЫХ МАРОК ПОРОХОВ")
    print("=" * 80)

    ibs = Individual_Ballistics_Solutions()
    ibs.display_powders_table()
    ibs.save_powders_table()

    # Вывод таблицы параметров в точке максимума C_Sl
    ibs.display_max_C_Sl_table()
    ibs.save_max_C_Sl_table()

    # Построение диаграмм для всех порохов
    ibs.plot_all_individual_diagrams(save=True)

    print("\nРабота программы завершена.")