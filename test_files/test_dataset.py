from test_files.utils import timer
     
@timer
def test_dataset():
    import numpy as np
    import os
    from PIL import Image
    from utils.dataset import MultilayerConditionalRain
    
    rain_generator = MultilayerConditionalRain((256, 256))
    rain_generator.set_camera_para()
    rain_generator.set_condition()
    rain, depth = rain_generator.generate_rain()
    rain_img = Image.fromarray(np.clip((rain * 255), 0, 255).astype(np.uint8))
    depth_maps = [Image.fromarray(np.clip((d * 255), 0, 255).astype(np.uint8)) for d in depth]
    
    save_dir = "test_files/out"
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = f"{save_dir}/rain_generation"
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    rain_img.save(f"{save_dir}/img.png")
    for idx, d in enumerate(depth_maps):
        d.save(f"{save_dir}/depth_map_{idx}.png")
    print(f"image saved to {save_dir}")
    
@timer
def test_depth_prediction():
    from PIL import Image
    import os
    import numpy as np
    from utils.depth_prediction import DepthPrediction
    from utils.utils import to_device
    
    source_path = "test_imgs/img1.jpg"
    img = np.array(Image.open(source_path))
    dp = DepthPrediction()
    img_t = dp.midas_transforms(img)
    img_t = to_device(img_t)
    depth = dp(img_t, ret_shape=(256,256))
    depth_img = Image.fromarray(np.array(dp.normalize(depth).cpu().squeeze() * 255).astype(np.uint8))
    
    save_dir = "test_files/out"
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    depth_img.save(f"{save_dir}/depth_map.png")
    print(f"image saved to {save_dir}/depth_map.png")
    
@timer
def test_rain_synthesis():
    from PIL import Image
    import os
    import numpy as np
    from utils.depth_prediction import DepthPrediction
    from utils.rain_synthesis import RainSynthesisDepth
    from utils.utils import to_device
    import torch

    source_path = "test_files"
    rain_layers = [torch.Tensor(np.array(Image.open(f"{source_path}/out/rain_generation/depth_map_{idx}.png")))
                                for idx in range(6)]
    rain_layers = torch.stack(rain_layers).unsqueeze(0)
    img = np.array(Image.open(f"test_imgs/img1.jpg"))
    dp = DepthPrediction()
    depth = dp(to_device(dp.midas_transforms(img)), ret_shape=(256, 256))
    rain_synthesizer = RainSynthesisDepth()
    syn_img = rain_synthesizer.synthesize(to_device(torch.Tensor(img.transpose((2,0,1)))/255.), depth, rain_layers/255.)
    syn_img = Image.fromarray((np.clip(syn_img.squeeze().cpu().numpy(), 0, 1).transpose((1,2,0)) * 255).astype(np.uint8))
    
    save_dir = "test_files/out"
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    syn_img.save(f"{save_dir}/syn_img.png")
    print(f"image saved to {save_dir}/syn_img.png")
    
# @timer
# def test_PQGAN_inference():
#     from models import PQGAN
#     from utils.config import args
#     args.is_train = 'False'
#     checkpoints_dir = "models/pretrained/nz128"
    
#     pqgan = PQGAN()
    