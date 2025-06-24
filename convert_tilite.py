import torch
import networks
import torch.nn as nn


class FullModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(FullModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)

        disp1 = output[('disp', 0)]
        disp2 = output[('disp', 1)]
        disp3 = output[('disp', 2)]
        
        return disp1, disp2, disp3
    

model_encoder = networks.LiteMono(model='lite-mono')

models_depth = networks.DepthDecoder(model_encoder.num_ch_enc,
                                    [0,1,2])


model_encoder.load_state_dict(torch.load("/home/dev/Lite_Mono/encoder.pth"), strict=False)
models_depth.load_state_dict(torch.load("/home/dev/Lite_Mono/depth.pth"))

model = FullModel(model_encoder, models_depth)
model.eval()

# 더미 입력 (입력 사이즈에 맞게)
dummy_input = torch.randn(1, 3, 192, 640)  # 예시

# ONNX로 내보내기
torch.onnx.export(
    model, dummy_input, "monodepth.onnx",
    input_names=["input"], output_names=["disp0", "disp1", "disp2"],
    opset_version=11
)