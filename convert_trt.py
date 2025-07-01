import os.path as osp
import torch
import tensorrt as trt
import time
import onnx

from PIL import Image
# from networks import depth_encoder_ori, depth_encoder_v2, depth_decoder, depth_decoder_v2, depth_encoder_only_ghostinCDC
from networks import depth_encoder_v2, depth_decoder_v2
from glob import glob
from torchvision import transforms
from onnx import numpy_helper
import onnxoptimizer


class FullModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        
        disp = out[('disp', 0)]
        disp_2x_down = out[('disp', 1)]
        disp_4x_down = out[('disp', 2)]
        
        return disp
    
    
def custom_load_state_dict(loaded_enc, loaded_dec, model_type=None):
    with torch.no_grad():
        if model_type == 'original':
            encoder = depth_encoder_ori.LiteMono(model="lite-mono",
                                                        drop_path_rate=0.2,
                                                        width=640, height=192)
            decoder = depth_decoder.DepthDecoder(encoder.num_ch_enc, scales=[0, 1, 2])
        
        elif model_type == 'proposal':
            encoder = depth_encoder_only_ghostinCDC.LiteMono(model="lite-mono",
                                                        drop_path_rate=0.2,
                                                        width=640, height=192)
            decoder = depth_decoder.DepthDecoder(encoder.num_ch_enc, scales=[0, 1, 2])
        
        # elif model_type == 'proposal2':
        elif model_type == 'AsymDC':
            encoder = depth_encoder_v2.LiteMono(model="lite-mono",
                                                        drop_path_rate=0.2,
                                                        width=640, height=192)
            decoder = depth_decoder_v2.DepthDecoder(encoder.num_ch_enc, scales=[0, 1, 2])
        
        enc_state_dict, dec_state_dict = encoder.state_dict(), decoder.state_dict()
        
        encoder.load_state_dict({
            k: v for k, v in loaded_enc.items() 
            if k in enc_state_dict and enc_state_dict[k].shape == v.shape})
        decoder.load_state_dict({
            k: v for k, v in loaded_dec.items() 
            if k in dec_state_dict and dec_state_dict[k].shape == v.shape})

        encoder.to(device)
        decoder.to(device)
        
        encoder.eval()
        decoder.eval()
    
    return encoder, decoder


def convert_onnx_and_trt(onnx_dir, models, device='cpu'):
    dummy_input = torch.randn(1, 3, 192, 640).to(device)  # 입력 사이즈에 맞게 조절
    
    encoder, decoder = models
    model = FullModel(encoder, decoder).eval()
    model = model.to(device)
    
    # from torchview import draw_graph
    # graph = draw_graph(model, input_data=dummy_input,expand_nested=True)
    # graph.visual_graph.render("model_architecture", format="png")  # 저장 옵션
    
    torch.onnx.export(
        model, dummy_input, osp.join(onnx_dir, model_type+'.onnx'),
        input_names=["input"], output_names=["output"],
        dynamic_axes=None,
        export_params=True,
        do_constant_folding=True,
        opset_version=17
    )
    
    # 그래프 최적화
    model = onnx.load(osp.join(onnx_dir, model_type+'.onnx'))
    passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_transpose',
        'fuse_consecutive_transposes',
        'fuse_bn_into_conv',
        'fuse_pad_into_conv',  # Depthwise-friendly
        'fuse_add_bias_into_conv'  # TensorRT에서 Conv + Bias로 병합
    ]

    # passes = onnxoptimizer.get_available_passes()  # 모든 최적화 패스 리스트
    optimized_model = onnxoptimizer.optimize(model, passes)
    for init in optimized_model.graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            arr = numpy_helper.to_array(init)
            arr32 = arr.astype('int32')
            init.CopyFrom(numpy_helper.from_array(arr32, init.name))
    
    onnx.save(optimized_model, osp.join(onnx_dir, f'optimized_{model_type}.onnx'))
    engine = build_engine(osp.join(onnx_dir, f'optimized_{model_type}.onnx'))
    
    with open(osp.join(onnx_dir, f"optimized_{model_type}.engine"), "wb") as f:
        f.write(engine.serialize())
    print('done')


# region [build trt]
def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB


    # ✅ weight streaming 활성화
    # config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

    # ✅ FP16 지원 여부 확인
    config.set_flag(trt.BuilderFlag.FP16)
    
    
    # ✅ 네트워크 생성 (explicit batch)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # ✅ ONNX 파싱
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as f:
        parser.parse(f.read())

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build engine")
        return None
    

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # # ✅ 엔진 빌드
    # engine = builder.build_engine(network, config)
    return engine



def infer_pth():
    test_image_dir = r"/home/dev/DATASET/kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data"
    test_image_paths = glob(osp.join(test_image_dir, "*.jpg"))
    
    enc_state_dict = torch.load(enc_model_path, map_location=device)
    dec_state_dict = torch.load(dec_model_path, map_location=device)
    
    encoder, decoder = custom_load_state_dict(enc_state_dict, dec_state_dict, model_type)
    
    transform = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
    ])
    
    avg_elapsed_time = 0
    for idx, test_image_path in enumerate(test_image_paths):
        image = Image.open(test_image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        for _ in range(4):
            torch.cuda.synchronize()
            init_start = time.perf_counter()
            
            
            features = encoder(image)
            outputs = decoder(features)
            
            inference_elapsed_time = time.perf_counter() - init_start
            torch.cuda.synchronize()
        
        avg_elapsed_time += inference_elapsed_time
        idx += 1
        print(f'Elapsed Time: {inference_elapsed_time * 1000:.5f} ms')
    
    avg_elapsed_time = (avg_elapsed_time / idx)
    print(f'{avg_elapsed_time * 1000:.5f} ms')



def main():
    onnx_dir = osp.join(exp_dir, model_type, 'lite-mono')
    
    enc_state_dict = torch.load(enc_model_path, map_location=device)
    dec_state_dict = torch.load(dec_model_path, map_location=device)
    
    encoder, decoder = custom_load_state_dict(enc_state_dict, dec_state_dict, model_type)
    convert_onnx_and_trt(onnx_dir, models=[encoder, decoder], device=device)


if __name__ == "__main__":
    device = 'cuda'
    # device = 'cpu'
    model_type = 'AsymDC'
    exp_dir = osp.join(osp.dirname(__file__), "experiments")

    enc_model = 'encoder.pth'
    dec_model = 'depth.pth'
    
    enc_model_path = osp.join(exp_dir, model_type, 'lite-mono', enc_model)
    dec_model_path = osp.join(exp_dir, model_type, 'lite-mono', dec_model)
    
    main()
    # infer_pth()