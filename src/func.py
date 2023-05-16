import numpy as np
from scipy.interpolate import interp1d


def inter1pd_iv_curve(v_near, data):
    xdata = data.voltage.values
    ydata = data.current.values
    if v_near in data.voltage.values:
        print(1)
        return data[data.voltage == v_near].current.values[0]/1e3
    if v_near > xdata.max() or v_near < xdata.min():
        print('电压月线图')
        return 0
    func = interp1d(xdata, ydata, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(start=v_near-100, stop=v_near+100, num=10)
    y_new = func(x_new)
    y = np.maximum(y_new, 0)
    y = np.minimum(y, ydata.max())
    return y.mean()/1e3


def calc_reward(dp, mpp, p):
    abs_diff_p = abs(p - mpp)
    r = 0
    if abs_diff_p < mpp / 30:
        r = p * p / (mpp * mpp) + p / mpp
    else:
        if dp < -mpp / 100:
            r = dp / mpp
        else:
            r = 0.5
    if dp < -1:
        r = -1
    elif -1 <= dp < 1:
        r = 0.1
    else:
        r = 1
    return r
    # return round(dp, 0)
