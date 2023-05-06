import numpy as np
from scipy.interpolate import interp1d


def inter1pd_iv_curve(v_near, data):
    xdata = data.voltage.values
    ydata = data.current.values
    func = interp1d(xdata, ydata, kind='linear', fill_value="extrapolate")
    if v_near > xdata.max() or v_near < xdata.min():
        return 0
    x_new = np.linspace(start=v_near-100, stop=v_near+100, num=10)
    y_new = func(x_new)
    y = np.maximum(y_new, 0)
    y = np.minimum(y, ydata.max())
    return y.mean()/1e3
