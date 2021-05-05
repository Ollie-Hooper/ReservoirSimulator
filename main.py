import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def main():
    reservoir_model = ReservoirModel()

    # initial condition
    V0 = reservoir_model.volume(80)

    # time points
    t = np.linspace(0, 60 * 60 * 24 * 365, num=1000000)  # 1 year

    # solve ode
    V = odeint(reservoir_model.model, V0, t)

    # plot results
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('volume', color=color)
    ax1.plot(t, V, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('height', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, reservoir_model.height(V), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


class ReservoirModel:

    def __init__(self):
        # model parameters
        self.residual_volume_rate = 0
        self.max_height = 100
        self.l = 100
        self.k_input = 0.01
        self.k_output = 0.02
        self.pipe_height = 5
        self.pipe_radius = 2
        self.g = 9.8
        self.discharge_coefficient = 0.98
        self.V_max = self.volume(self.max_height)

    def height(self, V):
        h = ((3 * V) / (4 * self.l)) ** (2 / 3)
        return h

    def volume(self, h):
        V = (4 * self.l * h ** (3 / 2)) / 3
        return V

    def surface_area(self, V):
        h = self.height(V)
        A = self.l * 2 * np.sqrt(h)
        return A

    # function that returns dV/dt
    def model(self, V, t):
        didt = self.input_rate(V)  # input rate
        dodt = self.output_rate(V)  # output rate
        drdt = self.rainfall(t)  # rainfall
        # evaporation
        # seepage
        c = self.residual_volume_rate
        dVdt = didt - dodt + c + drdt
        return dVdt

    def input_rate(self, V):
        k = self.k_input
        didt = k * (self.V_max - V)
        return didt

    def output_rate(self, V):
        pipe_area = np.pi * self.pipe_radius ** 2

        h = self.height(V)

        if self.pipe_height < h:
            exit_velocity = np.sqrt(2 * self.g * (h - self.pipe_height))
        else:
            exit_velocity = 0

        dodt = self.discharge_coefficient * pipe_area * exit_velocity
        return dodt

    # rainfall
    def rainfall(self, t):
        years = t / (365.25 * 24 * 60 * 60)
        drdt = self.volume(0.3 * np.cos(2 * np.pi * years) + 0.8)
        return drdt


if __name__ == '__main__':
    main()
