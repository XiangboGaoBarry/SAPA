# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
# 	from PIL import Image
# 	from tqdm import tqdm
# 	for i in tqdm(range(50)):
from test_files.utils import timer
     
@timer
def test_dataset():
    import numpy as np
    import os
    from PIL import Image
    from utils.dataset import MultilayerConditionalRain
    rain = MultilayerConditionalRain((512, 512))
    rain.set_camera_para()
    rain.set_condition()
    r, d = rain.generate_rain()
    print(r.max(), r.min())
    rain_img = Image.fromarray(np.clip((r * 255), 0, 255).astype(np.uint8)).convert('RGB')
    
    save_dir = "test_files/dataset"
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    rain_img.save(f"{save_dir}/test.png")