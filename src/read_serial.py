import serial  # 导入串口包
import time  # 导入时间包
import pandas as pd
from tqdm import tqdm
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


def read_serial_data(com_port):
    ser = serial.Serial(com_port, 115200, timeout=5)  # 开启com3口，波特率115200，超时5
    ser.flushInput()  # 清空缓冲区
    # voltage_in = 0
    voltage_out = 0
    mode = 'MPPT_ON'
    # current_in = 0
    while True:
        count = ser.inWaiting()  # 获取串口缓冲区数据
        if count != 0:
            try:
                recv = ser.read(ser.in_waiting).decode("gbk")  # 读出串口数据，数据采用gbk编码
            except:
                pass
        try:
            voltage_in = float(recv[recv.find('input voltage = ')+16: recv.find('V in')-1])
            voltage_out = float(recv[recv.find('output voltage = ')+17: recv.find('V out')- 1])
            current_in = float(recv[recv.find('input current = ')+16: recv.find('A in')-1])
            current_out = float(recv[recv.find('output current = ') + 16: recv.find('A out') - 1])
            # mode = recv[recv.find('!!!!!!!! ') + 9: recv.rfind('!!!!!!!') - 1]
            duty = int(recv[recv.find(' /1024') - 5: recv.find(' /1024') - 1])
            # print(mode)
            break
        except:
            pass
        time.sleep(0.2)  # 延时0.1秒，免得CPU出问题
    return voltage_in, voltage_out, current_in, current_out, mode, duty


def save_serial_data():
    df_1 = pd.DataFrame(columns=['v_in', 'v_out', 'i_in', 'i_out', 'duty_num'])
    time_now = time.time()

    for i in tqdm(range(300)):
        # com19 #1组件
        # com20 #2组件
        # com21 #3组件
        # com22 #4组件
        # print('============20================')
        v_in, v_out, i_in, i_out, state, duty_num = read_serial_data("COM24")
        df_1.loc[df_1.shape[0], :] = [v_in, v_out, i_in, i_out, duty_num]
    df_1.to_csv('../data/serial_data_pv_{}.csv'.format('test_keil2'), encoding='utf-8-sig')



def get_port_str(port_1, port_2, port_3, port_4):
    ser_1 = serial.Serial(port_1, 115200, timeout=5)  # 开启com3口，波特率115200，超时5
    ser_1.flushInput()  # 清空缓冲区
    ser_2 = serial.Serial(port_2, 115200, timeout=5)  # 开启com3口，波特率115200，超时5
    ser_2.flushInput()  # 清空缓冲区
    ser_3 = serial.Serial(port_3, 115200, timeout=5)  # 开启com3口，波特率115200，超时5
    ser_3.flushInput()  # 清空缓冲区
    ser_4 = serial.Serial(port_4, 115200, timeout=5)  # 开启com3口，波特率115200，超时5
    ser_4.flushInput()  # 清空缓冲区
    recv_1 = 'read_failed'
    recv_2 = 'read_failed'
    recv_3 = 'read_failed'
    recv_4 = 'read_failed'
    time.sleep(0.1)
    count_1 = ser_1.inWaiting()  # 获取串口缓冲区数据
    if count_1 != 0:
        try:
            recv_1 = ser_1.read(ser_1.in_waiting).decode("gbk")  # 读出串口数据，数据采用gbk编码
            print(recv_1)
        except:
            pass
    time.sleep(0.1)  # 延时0.1秒，免得CPU出问题
    count_2 = ser_2.inWaiting()  # 获取串口缓冲区数据
    if count_2 != 0:
        try:
            recv_2 = ser_2.read(ser_2.in_waiting).decode("gbk")  # 读出串口数据，数据采用gbk编码
        except:
            pass
    time.sleep(0.1)  # 延时0.1秒，免得CPU出问题
    count_3 = ser_3.inWaiting()  # 获取串口缓冲区数据
    if count_3 != 0:
        try:
            recv_3 = ser_3.read(ser_3.in_waiting).decode("gbk")  # 读出串口数据，数据采用gbk编码
        except:
            pass
    time.sleep(0.1)  # 延时0.1秒，免得CPU出问题
    count_4 = ser_4.inWaiting()  # 获取串口缓冲区数据
    if count_4 != 0:
        try:
            recv_4 = ser_4.read(ser_4.in_waiting).decode("gbk")  # 读出串口数据，数据采用gbk编码
        except:
            pass
    # time.sleep(0.1)  # 延时0.1秒，免得CPU出问题
    return recv_1, recv_2, recv_3, recv_4


def record_serial():
    pv_1 = 'COM19'
    pv_2 = 'COM20'
    pv_3 = 'COM21'
    pv_4 = 'COM22'
    df_1 = pd.DataFrame(columns=['pv_1', 'pv_2', 'pv_3', 'pv_4'])
    time_now = time.time()
    while True:
        str_1, str_2, str_3, str_4 = get_port_str(pv_1, pv_2, pv_3, pv_4)
        df_1.loc[df_1.shape[0], :] = [str_1, str_2, str_3, str_4]
        # print(str_2)
        df_1.to_csv('../data/serial_data_4_pv{}.csv'.format(time_now), encoding='utf-8-sig')


def read_serial_data_ide(com_port):
    ser = serial.Serial(com_port, 115200, timeout=5)  # 开启com3口，波特率115200，超时5
    ser.flushInput()  # 清空缓冲区
    # voltage_in = 0
    voltage_out = 0
    mode = 'MPPT_ON'
    # current_in = 0
    while True:
        count = ser.inWaiting()  # 获取串口缓冲区数据
        if count != 0:
            try:
                recv = ser.read(ser.in_waiting).decode("gbk")  # 读出串口数据，数据采用gbk编码
            except:
                pass
        try:
            # print(recv)
            voltage_in = float(recv[recv.find('input voltage_new = ')+20: recv.find('mV in')-1])
            current_in = float(recv[recv.find('input current_new = ')+20: recv.find('mA in')-1])
            break
        except:
            pass
        time.sleep(0.2)  # 延时0.1秒，免得CPU出问题
    return voltage_in, current_in


def save_serial_data_ide():
    df_1 = pd.DataFrame(columns=['v_in', 'v_out'])
    time_now = time.time()

    for i in tqdm(range(2000)):
        # com19 #1组件
        # com20 #2组件
        # com21 #3组件
        # com22 #4组件
        # print('============20================')
        v_in, i_in = read_serial_data_ide("COM24")
        df_1.loc[df_1.shape[0], :] = [v_in, i_in]
    df_1.to_csv('../data/serial_data_pv_{}.csv'.format('test_ide_1_5-618W'), encoding='utf-8-sig')


if __name__ == '__main__':
    import time
    # save_serial_data()
    save_serial_data_ide()
    # record_serial()