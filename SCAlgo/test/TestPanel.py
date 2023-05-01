# Demo/Test UI for Data Analysis Modules
# Author: Shihchuan Kao (Kevin Kao)
# Contact: kaoshihchuan@gmail.com

import tkinter as tk
import tkinter.font as tkfont
import tkinter.messagebox as tkmsg

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
# import SCAlgo.src.LogistFit as LogistFit
import SCAlgo.src.DataGenerator as DataGen
import SCAlgo.src.LogisticFit as LogisticFit


class Plots:

    __instance__ = None

    def __init__(self):

        self.canvas = None
        self.fig = plt.figure(figsize=(8, 7))

        self.ax1 = self.fig.add_subplot()
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        # self.ax1 = plt.subplot2grid((2, 1), (0, 0))
        # self.ax2 = plt.subplot2grid((2, 1), (1, 0))
        # self.ax2.set_xlabel('Time (sec)')
        # self.ax2.set_ylabel('Travel distance (m)')

        Plots.__instance__ = self

    def config(self, root_tk):

        # ==== Scan profile display =====
        self.canvas = FigureCanvasTkAgg(self.fig, master=root_tk)  # A tk.DrawingArea.
        self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=15, columnspan=11, sticky='NSWE')
        # self.fig.tight_layout(w_pad=0.8, h_pad=0.0)

        # ##############    TOOLBAR    ###############
        toolbar_frame = tk.Frame(master=root_tk)
        toolbar_frame.grid(row=0, column=0, columnspan=6)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def __del__(self):
        Plots.__instance__ = None

    @staticmethod
    def get_instance():

        if Plots.__instance__ is None:
            Plots()

        return Plots.__instance__


class GuiWindow:
    __instance__ = None

    n_1d_data_label_key = 'n_1d_data_label'
    n_1d_data_entry_key = 'n_1d_data_entry'
    logistic_para_label_key = 'logistic_para_label'
    logistic_para_entry_key = 'logistic_para_entry'
    poly1d_para_label_key = 'poly1d_para_label'
    poly1d_para_entry_key = 'poly1d_para_entry'
    data_1d_range_label_key = 'data_1d_range_label'
    data_1d_min_entry_key = 'data_1d_min_entry'
    data_1d_max_entry_key = 'data_1d_max_entry'

    def __init__(self, rootgui):
        self.window = rootgui

        # define the scanning display
        self.frame00 = tk.Frame(self.window)
        self.frame00.grid(row=0, column=0, sticky=tk.NSEW, rowspan=3)
        self.frame01 = tk.Frame(self.window)
        self.frame01.grid(row=0, column=1, sticky=tk.N, padx=10, pady=40)
        self.frame02 = tk.Frame(self.window)
        self.frame02.grid(row=1, column=1, sticky=tk.N, padx=10, pady=0)
        self.frame03 = tk.Frame(self.window)
        self.frame03.grid(row=2, column=1, sticky=tk.N, padx=10, pady=0)

        self.label_font_16 = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.label_font_12 = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.report_font_12 = tkfont.Font(family="Helvetica", size=12)

        self.x_data = None
        self.y_data = None
        self.y_predict = None

        self.n_1d_data_label = None
        self.n_1d_data_entry = None
        self.logistic_para_label = None
        self.logistic_para_entry = None
        self.wgts = {
            self.n_1d_data_label_key: None,
            self.n_1d_data_entry_key: None,
            self.logistic_para_label_key: None,
            self.logistic_para_entry_key: None,
            self.data_1d_range_label_key: None,
            self.data_1d_min_entry_key: None,
            self.data_1d_max_entry_key: None,
            self.poly1d_para_label_key: None,
            self.poly1d_para_entry_key: None,
        }

        self.gen_data_opt_list = ['Logistic1D', 'Logistic2D', 'Polynomial']
        self.gen_data_opt_var = tk.StringVar(self.window, value=self.gen_data_opt_list[0])
        self.n_1d_data_var = tk.IntVar(self.window, value=100)
        self.logistic_para_var = tk.DoubleVar(self.window, value=0.5)
        self.step_size_var = tk.DoubleVar(self.window, value=0.1)
        self.data_1d_min_var = tk.DoubleVar(self.window, value=-0.5)
        self.data_1d_max_var = tk.DoubleVar(self.window, value=0.5)
        self.poly1d_para_var = tk.StringVar(self.window, value='1,0')

        # ========= config matplot display =============
        self.plot = Plots()
        self.plot.config(self.frame00)

        gen_label = tk.Label(self.frame01, text='Data Simulator', font=self.label_font_12)
        gen_label.grid(row=0, column=0, columnspan=2)

        gen_data_menu = tk.OptionMenu(self.frame01,
                                      self.gen_data_opt_var,
                                      *self.gen_data_opt_list,
                                      command=lambda selection: self.set_gen_data_option()
                                      )
        gen_data_menu.grid(row=1, column=1, sticky=tk.EW)

        gen_data_btn = tk.Button(self.frame01, text='Gen', font=self.label_font_12,
                                 command=self.generate_sim_data)
        gen_data_btn.grid(row=2, column=1)

        logistic_fit_btn = tk.Button(self.frame01, text='Fit', font=self.label_font_12,
                                     command=self.fit_logistic_1d)
        logistic_fit_btn.grid(row=2, column=2)
        self.window.update_idletasks()

        GuiWindow.__instance__ = self
    # end __init__

    def __del__(self):
        GuiWindow.__instance__ = None
    # end __del__

    def set_gen_data_option(self):
        gen_data_opt = self.gen_data_opt_var.get()
        print(' Gen Data option is %s' % gen_data_opt)
        for key in self.wgts:
            if self.wgts[key] is not None:
                self.wgts[key].destroy()

        if gen_data_opt == self.gen_data_opt_list[0]:
            self.wgts[self.n_1d_data_label_key] = tk.Label(self.frame02, width=12,
                                                           text='N of Point')
            self.wgts[self.n_1d_data_label_key].grid(row=0, column=1)
            self.wgts[self.n_1d_data_entry_key] = tk.Entry(self.frame02, width=10,
                                                           textvariable=self.n_1d_data_var)
            self.wgts[self.n_1d_data_entry_key].grid(row=0, column=2)

            self.wgts[self.data_1d_range_label_key] = tk.Label(self.frame02, width=12,
                                                               text='Min ~ Max')
            self.wgts[self.data_1d_range_label_key].grid(row=1, column=1)
            self.wgts[self.data_1d_min_entry_key] = tk.Entry(self.frame02, width=10,
                                                             textvariable=self.data_1d_min_var)
            self.wgts[self.data_1d_min_entry_key].grid(row=1, column=2)
            self.wgts[self.data_1d_max_entry_key] = tk.Entry(self.frame02, width=10,
                                                             textvariable=self.data_1d_max_var)
            self.wgts[self.data_1d_max_entry_key].grid(row=1, column=3)

            self.wgts[self.logistic_para_label_key] = tk.Label(self.frame02, width=12,
                                                               text='Logistic Para')
            self.wgts[self.logistic_para_label_key].grid(row=2, column=1)
            self.wgts[self.logistic_para_entry_key] = tk.Entry(self.frame02, width=10,
                                                               textvariable=self.logistic_para_var)
            self.wgts[self.logistic_para_entry_key].grid(row=2, column=2)
        elif gen_data_opt == self.gen_data_opt_list[2]:
            self.wgts[self.n_1d_data_label_key] = tk.Label(self.frame02, width=12,
                                                           text='N of Point')
            self.wgts[self.n_1d_data_label_key].grid(row=0, column=1)
            self.wgts[self.n_1d_data_entry_key] = tk.Entry(self.frame02, width=10,
                                                           textvariable=self.n_1d_data_var)
            self.wgts[self.n_1d_data_entry_key].grid(row=0, column=2)

            self.wgts[self.data_1d_range_label_key] = tk.Label(self.frame02, width=12,
                                                               text='Min ~ Max')
            self.wgts[self.data_1d_range_label_key].grid(row=1, column=1)
            self.wgts[self.data_1d_min_entry_key] = tk.Entry(self.frame02, width=10,
                                                             textvariable=self.data_1d_min_var)
            self.wgts[self.data_1d_min_entry_key].grid(row=1, column=2)
            self.wgts[self.data_1d_max_entry_key] = tk.Entry(self.frame02, width=10,
                                                             textvariable=self.data_1d_max_var)
            self.wgts[self.data_1d_max_entry_key].grid(row=1, column=3)
            self.wgts[self.poly1d_para_label_key] = tk.Label(self.frame02, width=12,
                                                             text='Poly1D Para')
            self.wgts[self.poly1d_para_label_key].grid(row=2, column=1)
            self.wgts[self.poly1d_para_entry_key] = tk.Entry(self.frame02, width=10,
                                                             textvariable=self.poly1d_para_var)
            self.wgts[self.poly1d_para_entry_key].grid(row=2, column=2)
        else:
            for key in self.wgts:
                if self.wgts[key] is not None:
                    self.wgts[key].destroy()

    def get_poly_coefficient(self):

        p_str = self.poly1d_para_var.get()
        p_str_v = p_str.split(',')
        p_v = []
        for it in p_str_v:
            p_v.append(float(it))

        print(' Poly paramters are ')
        print(p_v)
        return p_v

    def generate_sim_data(self):
        gen_data_opt = self.gen_data_opt_var.get()
        ya = np.array([])
        xa = np.array([])
        if gen_data_opt == self.gen_data_opt_list[0]:
            n_pt = self.n_1d_data_var.get()
            x_min = self.data_1d_min_var.get()
            x_max = self.data_1d_max_var.get()
            logistic_p = self.logistic_para_var.get()
            ya, xa = DataGen.gen_logistic_1d(x_min=x_min, x_max=x_max, n_pt=n_pt, f1=logistic_p)
            self.x_data = xa
            self.y_data = ya

        if gen_data_opt == self.gen_data_opt_list[2]:
            n_pt = self.n_1d_data_var.get()
            plist = self.get_poly_coefficient()
            x_min = self.data_1d_min_var.get()
            x_max = self.data_1d_max_var.get()
            ya, xa = DataGen.gen_polynomid_1d(plist, x_range=(x_min, x_max), n_pt=n_pt, err_scale=1.)
            self.x_data = xa
            self.y_data = ya

        self.plot.ax1.cla()
        self.plot.ax1.grid()
        self.plot.ax1.plot(xa, ya, marker='^', color='r')
        # self.plot.ax2.plot(ta, sa, 'b')
        self.plot.ax1.set_ylabel('Y')
        self.plot.ax1.set_xlabel('X')

        self.plot.canvas.draw()
        self.plot.canvas.flush_events()

    def fit_logistic_1d(self):

        if self.x_data is None:
            tkmsg.showwarning('Warning', 'No Data for fitting')
            return
        else:
            self.y_predict = LogisticFit.logistic_1d_fit(self.x_data, self.y_data)
            self.plot.ax1.cla()
            self.plot.ax1.grid()
            self.plot.ax1.plot(self.x_data, self.y_data, marker='^', color='r')
            self.plot.ax1.plot(self.x_data, self.y_predict, color='b')
            self.plot.ax1.set_ylabel('Y')
            self.plot.ax1.set_xlabel('X')

            self.plot.canvas.draw()
            self.plot.canvas.flush_events()
