import logging
import os
import datetime
import time


def set_log():
    MY_APP_NAME = os.environ.get('APP_NAME')
    MY_K8S_POD_ID = os.environ.get('K8S_POD_ID')
    MY_K8S_POD_NS = os.environ.get('K8S_POD_NS')
    MY_K8S_POD_UID = os.environ.get('K8S_POD_UID')
    route = '/data/{}/{}_{}_{}/log_files/'.format(MY_APP_NAME, MY_K8S_POD_NS, MY_K8S_POD_ID, MY_K8S_POD_UID)
    if os.path.lexists('/data'):
        if not os.path.lexists(route):  # 判断系统是否存在该指定路径，若不存在则创建该路径与子文件夹
            os.makedirs(route)
    else:
        print('No path /data')
        route = 'log_files/log_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    tz = time.strftime("%z", time.localtime())
    tz_s = ":".join([tz[0:3], tz[3:]])
    logging.basicConfig(
        filename=''.join([route, 'info.log_files']),
        format='P %(asctime)s.%(msecs)03d{} %(levelname)s [-] %(module)s - - %(name)s: %(message)s'.format(tz_s),
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO)
