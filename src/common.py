from dataclasses import dataclass, field
import collections

StepResult = collections.namedtuple(
    "StepResult", field_names=["obs", "reward", "done", "info"]
)


@dataclass
class History:
    g: list = field(default_factory=list)  # 光强
    t: list = field(default_factory=list)  # 温度
    p: list = field(default_factory=list)  # 功率
    v: list = field(default_factory=list)  # 电压
    v_pv: list = field(default_factory=list)  # 组件电压
    i: list = field(default_factory=list)  # 电流
    dp: list = field(default_factory=list)  # dp = p[t] - p[t-1]
    dv: list = field(default_factory=list)
    dv_set2pv: list = field(default_factory=list)  # dv_set2pv = v - v_pv
    di: list = field(default_factory=list)
    g_norm: list = field(default_factory=list)  # g_norm = g / G_MAX
    t_norm: list = field(default_factory=list)  # t_norm = t / T_MAX
    p_norm: list = field(default_factory=list)  # p_norm = p / pmax
    v_norm: list = field(default_factory=list)  # v_norm = v / voc
    i_norm: list = field(default_factory=list)  # i = i / isc
    dp_norm: list = field(default_factory=list)
    dv_norm: list = field(default_factory=list)
    deg: list = field(default_factory=list)