import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def main():
    for model in ['UK', 'Cameroon']:
        V, h, t = run_model(model)
        plot(V, h, t, model)


def run_model(model='UK', length=60 * 60 * 24 * 365.25 * 1):
    if model == 'UK':
        rainfall_func = rainfall_uk
        l = 500
        max_height = 20
        seepage_coefficient = 0.014
        elevation = 93
        latitude = 51.8
        humidity = 73
        temperature_data = 'temperatures_uk'
    elif model == 'Cameroon':
        rainfall_func = rainfall_uk
        l = 5000
        max_height = 40
        seepage_coefficient = 0.005625
        elevation = 223
        latitude = 9.05
        humidity = 75
        temperature_data = 'temperatures_cameroon'
    else:
        raise Exception('Model not recognised')

    reservoir_model = ReservoirModel(model_name=model, k_input=1e-5, rainfall_func=rainfall_func, l=l,
                                     max_height=max_height, seepage_coefficient=seepage_coefficient,
                                     elevation=elevation, latitude=latitude, humidity=humidity,
                                     temperature_data=temperature_data)

    if model == "UK":
        # initial condition
        V0 = reservoir_model.volume(18.834)  # reservoir_model.max_height / 2)
    elif model == "Cameroon":
        V0 = reservoir_model.volume(39.901)  # reservoir_model.max_height / 2)

    # time points
    t = np.linspace(0, length, num=1000000)

    V = solve(reservoir_model, V0, t)

    h = reservoir_model.height(V=V)

    return V, h, t


def plot(V, h, t, model_name):
    # plot results
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Volume (m^3)', color=color)
    ax1.plot(t, V, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Height (m)', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, h, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # otherwise the right y-label is slightly clipped
    #plt.title(f'Volume/height change of {model_name} reservoir in first week')
    plt.suptitle(f'Volume/height change of {model_name} reservoir over the year', size=12, y=0.98)
    plt.show()


def solve(model, V0, t):
    # solve ode
    V = odeint(model.model, V0, t)
    return V


def rainfall_uk(t):
    years = t / (365.25 * 24 * 60 * 60)
    weeks = t / (7 * 24 * 60 * 60)
    dhdt = (0.03 * np.cos(2 * np.pi * years) + 0.08 + 0.05 * np.sin(2 * np.pi * weeks)) / (
            60 * 60 * 24 * (365.25 / 12))
    return dhdt


def rainfall_cameroon(t):
    years = t / (365.25 * 24 * 60 * 60)
    weeks = t / (7 * 24 * 60 * 60)
    dhdt = (0.1 * np.cos(2 * np.pi * years + np.pi) + 0.1 + 0.01 * np.sin(np.pi * years) * np.sin(
        2 * np.pi * weeks)) / (60 * 60 * 24 * (365.25 / 12))
    return dhdt


class ReservoirModel:

    def __init__(self, model_name, residual_volume_rate=0, max_height=20, l=100, k_input=1e-5, pipe_height=5,
                 pipe_radius=0.1, g=9.8, discharge_coefficient=0.98, seepage_coefficient=0.014,
                 rainfall_func=rainfall_uk, elevation=93, latitude=51.8, humidity=73,
                 temperature_data='temperatures_uk'):
        # model parameters
        self.model_name = model_name
        self.residual_volume_rate = residual_volume_rate
        self.max_height = max_height
        self.l = l
        self.k_input = k_input
        self.pipe_height = pipe_height
        self.pipe_radius = pipe_radius
        self.g = g
        self.discharge_coefficient = discharge_coefficient
        self.seepage_coefficient = seepage_coefficient
        self.rainfall_func = rainfall_func
        self.elevation = elevation
        self.latitude = latitude
        self.humidity = humidity
        self.temperature_data = np.genfromtxt(f'{temperature_data}.csv', delimiter=',')
        self.temperature_coeff = ()
        self.set_temperature_curve()
        self.V_max = self.volume(self.max_height)

    def height(self, V=None, w=None):
        L = self.l
        m = self.max_height
        if V is not None:
            return ((3 * np.sqrt(5) * V) / (20 * L * np.sqrt(m))) ** (2 / 3)
        elif w is not None:
            return (w ** 2) / (20 * m)

    def width(self, V=None, h=None):
        if h is None:
            h = self.height(V=V)
        m = self.max_height
        return 2 * np.sqrt(5 * m * h)

    def cross_sectional_area(self, V=None, h=None):
        if h is None:
            h = self.height(V=V)
        m = self.max_height
        return (4 * np.sqrt(5) * (m * h) ** (3 / 2)) / (3 * m)

    def volume(self, h):
        L = self.l
        A = self.cross_sectional_area(h=h)
        return L * A

    def surface_area(self, V=None, h=None, w=None):
        if w is None:
            if V is not None:
                w = self.width(V=V)
            elif h is not None:
                w = self.width(h=h)
        L = self.l
        return w * L

    def bed_area(self, V=None, h=None):
        if h is None:
            h = self.height(V=V)
        L = self.l
        m = self.max_height
        a = np.sqrt((4 * h + 5 * m) / h)
        return (L / 4) * (5 * m * np.log((a + 2) / np.abs(a - 2)) + 4 * h * a)

    # function that returns dV/dt
    def model(self, V, t):
        didt = self.input_rate(V)  # input rate
        dodt = self.output_rate(V)  # output rate
        drdt = self.rainfall(V, t)  # rainfall
        dsdt = self.seepage(V)  # seepage
        dedt = self.evaporation(V, t)  # evaporation
        c = self.residual_volume_rate
        dVdt = didt - dodt + drdt - dsdt - dedt + c
        return dVdt

    def input_rate(self, V):
        k = self.k_input
        didt = k * (self.V_max - V)
        return didt

    def output_rate(self, V):
        pipe_area = np.pi * self.pipe_radius ** 2

        h = self.height(V=V)

        if self.pipe_height < h:
            exit_velocity = np.sqrt(2 * self.g * (h - self.pipe_height))
        else:
            exit_velocity = 0

        dodt = self.discharge_coefficient * pipe_area * exit_velocity
        return dodt

    def rainfall(self, V, t):
        dhdt = self.rainfall_func(t)
        w = self.width(V=V)
        L = self.l
        drdt = dhdt * w * L
        return drdt

    def seepage(self, V):
        h = self.height(V=V)
        dsdt = (self.seepage_coefficient * self.bed_area(h=h)) / (24 * 60 * 60)
        return dsdt

    def evaporation(self, V, t):
        month = (t / (60 * 60 * 24 * (365.25 / 12))) % 12
        T = self.temperature(month)
        Tm = T + 0.006 * self.elevation
        RH = self.humidity
        Td = T - (100 - RH) / 5
        A = self.latitude
        dhdt = (((700 * Tm) / (100 - A) + (15 * (T - Td))) / (80 - T)) / (1000 * 24 * 60 * 60)
        w = self.width(V=V)
        L = self.l
        dedt = dhdt * w * L
        return dedt

    def set_temperature_curve(self):
        data = self.temperature_data
        start_of_year_mean = data[[0, -1]].mean()
        x_data = np.array([0, *np.arange(0.5, 12.5), 12])
        y_data = np.array([start_of_year_mean, *data, start_of_year_mean])

        sigma = np.ones(len(x_data))
        sigma[[0, -1]] = 0.01

        o = 5

        self.temperature_coeff, pcov = curve_fit(self.temperature, x_data, y_data, p0=np.ones(o + 1), sigma=sigma)

        x = np.linspace(0, 12, 1000)
        y = self.temperature(x)
        plt.plot(x, y)
        plt.scatter(x_data, y_data, c="r", marker="x")
        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
        plt.title(f'{self.model_name} Average Temperature over the Year')
        plt.show()

    def temperature(self, month, *coeff):
        if not coeff:
            coeff = self.temperature_coeff
        T = 0
        for p, c in enumerate(coeff):
            T += c * month ** p
        return T


if __name__ == '__main__':
    main()
