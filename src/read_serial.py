import serial  # 导入串口包
import time  # 导入时间包
import pandas as pd
# from src.pv_array import *
# READ_SENSOR_TIME = 0


def read_serial_data_sim(i):
    v_list = [25000, 23981, 24415, 19241]
    i_list = [750, 78, 154, 212.5]
    # print((i % 800) % 200)

    voltage_in = v_list[int((i % 800) / 200)]
    voltage_out = 14500
    current_in = i_list[int((i % 800) / 200)]

    return voltage_in, voltage_out, current_in


def read_csv_data():
    data = pd.read_csv('../data/data_for_train_A2C.csv')


def read_serial_data():
    ser = serial.Serial("COM8", 115200, timeout=5)  # 开启com3口，波特率115200，超时5
    ser.flushInput()  # 清空缓冲区
    # voltage_in = 0
    # voltage_out = 0
    # current_in = 0
    while True:
        count = ser.inWaiting()  # 获取串口缓冲区数据
        if count != 0:
            try:
                recv = ser.read(ser.in_waiting).decode("gbk")  # 读出串口数据，数据采用gbk编码
            except:
                pass
        try:
            voltage_in = float(recv[recv.find('input voltage = ')+16: recv.find('mV in')-1])
            voltage_out = float(recv[recv.find('output voltage = ')+17: recv.find('mV out')-1])
            current_in = float(recv[recv.find('input current = ')+16: recv.find('mA')-1])
            break
        except:
            pass
        time.sleep(0.1)  # 延时0.1秒，免得CPU出问题
    return voltage_in, voltage_out, current_in


if __name__ == '__main__':
    # v_in, v_out, I_in = read_serial_data()

    v_in, v_out, I_in = read_serial_data_sim(1)
    print('voltage_in', v_in)
    print('voltage_out', v_out)
    print('current_in', I_in)