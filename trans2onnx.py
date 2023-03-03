from mppt_ac import *
import onnx
from onnx_tf.backend import prepare
# from onnx2keras import onnx_to_keras
# import keras
# import tensorflow as tf
import numpy as np
from torch.autograd import Variable
# from pytorch2keras.converter import pytorch_to_keras
import torchvision.models as models
export_onnx_file = "RL_MPPT.onnx"

def tran2onnx():
    checkpoint = torch.load(CHECKPOINT_PATH)
    batch_size = 1  # 批处理大小
    input_shape = (1, 3)  # 输入数据,改成自己的输入shape
    device = torch.device("cpu")
    model = DiscreteActorCriticNetwork(input_size=3, n_actions=13).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    # #set the model to inference mode
    model.eval()

    x = torch.randn(batch_size, *input_shape)  # 生成张量
      # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=10,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                    "output": {0: "batch_size"}})

def onnx_to_h5(output_path):
    '''
    将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
    '''
    onnx_model = onnx.load(output_path)
    k_model = onnx_to_keras(onnx_model, ['input'])
    export_h5_file = "RL_MPPT.h5"
    keras.models.save_model(k_model, export_h5_file, overwrite=True, include_optimizer=True)
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = tf.keras.models.load_model(export_h5_file)
    model.summary()
    print(model)


# def pytorch2keras():
#     input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
#     input_var = Variable(torch.FloatTensor(input_np))
#
#     checkpoint = torch.load(CHECKPOINT_PATH)
#     device = torch.device("cpu")
#     model = DiscreteActorCriticNetwork(input_size=3, n_actions=13).to(device)
#
#     model.load_state_dict(checkpoint["model_state_dict"])
#     # #set the model to inference mode
#     model.eval()
#     k_model = \
#         pytorch_to_keras(model, input_var, [(3, 224, 224,)], verbose=True, change_ordering=True)
#
#     for i in range(3):
#         input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
#         input_var = Variable(torch.FloatTensor(input_np))
#         output = model(input_var)
#         pytorch_output = output.data.numpy()
#         keras_output = k_model.predict(np.transpose(input_np, [0, 2, 3, 1]))
#         error = np.max(pytorch_output - keras_output)
#         print('error -- ', error)  # Around zero :)


if __name__ == '__main__':
    tran2onnx()
    # pytorch2keras()
    # onnx_to_h5(export_onnx_file)