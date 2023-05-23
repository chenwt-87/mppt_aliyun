from mppt_ac import *
from mppt_ddpg import *
import onnx
# from onnx_tf.backend import prepare
# from onnx2keras import onnx_to_keras
# import keras
# import tensorflow as tf
import numpy as np
from torch.autograd import Variable
# from pytorch2keras.converter import pytorch_to_keras
# import torchvision.models as models
export_onnx_file = "SB3_RL_MPPT.onnx"


def tran2onnx():
    # checkpoint = torch.load(AGENT_CKP_PATH_3)
    checkpoint = sb3.A2C.load(AGENT_CKP_PATH_3)
    batch_size = 1  # 批处理大小
    input_shape = (1, 3)  # 输入数据,改成自己的输入shape
    device = torch.device("cpu")
    # model = DiscreteActorCriticNetwork(input_size=2, n_actions=15).to(device)
    my_actions = np.array([-25, -15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15, 25]) / 56
    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,  # 光伏组件参数
        HiS_DATA_PATH,  # 光伏组件历史数据
        pvarray_ckp_path=PVARRAY_CKP_PATH,  # 训练过程数据存储
        states=["v_norm", "i_norm"],
        # states=["v", "i"],
        mode='Train',
        # states=["v", "i", 'v_pv'],# 训练输入，可以有多种组合
        # reward_fn=RewardDeltaPowerVoltage(2, 0.1, 1),  # 奖励函数
        reward_fn=RewardDeltaPower(4, 2),
        # actions=np.array([-10, 0, 10])/56,  # 策略函数
        actions=my_actions,
    )
    net_arch = [128, dict(pi=[128, 3], vf=[128])]  # 定义一个新的神经网络架构
    model = sb3.A2C("MlpPolicy",
                    env,
                    verbose=1,
                    n_steps=2,
                    # learning_rate=0.0008,
                    learning_rate=linear_schedule(0.0004),
                    gamma=0.05,
                    # use_sde=True,
                    # batch_size=16,
                    # normalize_advantage=True,
                    create_eval_env=True,
                    tensorboard_log='./log_files/',
                    policy_kwargs={'net_arch': net_arch,
                                   'activation_fn': th.nn.SELU,
                                   'ortho_init': True,
                                   # 'use_sde': True
                                   }
                    # policy_kwargs=a2c_param_dict

                    )
    # model.load_state_dict(checkpoint["model_state_dict"])
    # #set the model to inference mode
    # model.eval()

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

# def onnx_to_h5(output_path):
#     '''
#     将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
#     '''
#     onnx_model = onnx.load(output_path)
#     k_model = onnx_to_keras(onnx_model, ['input'])
#     export_h5_file = "RL_MPPT.h5"
#     keras.models.save_model(k_model, export_h5_file, overwrite=True, include_optimizer=True)
#     # 下面内容是加载该模型，然后将该模型的结构打印出来
#     model = tf.keras.models.load_model(export_h5_file)
#     model.summary()
#     print(model)


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

def tran_sb3_to_onnx():
    import stable_baselines3 as sb3
    import os
    import torch as th
    export_onnx_file = "SB3_RL_MPPT.onnx"

    AGENT_CKP_PATH_3 = os.path.join("models", "sb3-onnx.tar")

    class OnnxablePolicy(th.nn.Module):
        def __init__(self, extractor, action_net, value_net):
            super().__init__()
            self.extractor = extractor
            self.action_net = action_net
            self.value_net = value_net

        def forward(self, observation):
            # NOTE: You may have to process (normalize) observation in the correct
            #       way before using this. See `common.preprocessing.preprocess_obs`
            action_hidden, value_hidden = self.extractor(observation)
            return self.action_net(action_hidden), self.value_net(value_hidden)

    # Example: model = PPO("MlpPolicy", "Pendulum-v1")
    model = sb3.A2C.load(AGENT_CKP_PATH_3, device="cpu")
    onnxable_model = OnnxablePolicy(
        model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
    )

    observation_size = model.observation_space.shape
    dummy_input = th.randn(1, *observation_size)
    th.onnx.export(
        onnxable_model,
        dummy_input,
        "my_a2c_model.onnx",
        opset_version=9,
        input_names=["input"],
    )


if __name__ == '__main__':
    tran2onnx()
    # pytorch2keras()
    # onnx_to_h5(export_onnx_file)