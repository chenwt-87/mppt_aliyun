from mppt_ac import *


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
    export_onnx_file = "RL_MPPT.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=10,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                    "output": {0: "batch_size"}})


if __name__ == '__main__':
    tran2onnx()