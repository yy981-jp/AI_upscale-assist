from openvino import Core
import numpy as np
import cv2
import sys

ie = Core()
model = ie.read_model("models/realesrgan_x4plus.xml")
compiled_model = ie.compile_model(model, "NPU")

# 入力画像
img = cv2.imread("input/" + sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# [N,C,H,W]
input_tensor = img.transpose(2,0,1)[None, ...]

# 推論
infer = compiled_model.create_infer_request()
infer.infer({compiled_model.input(0): input_tensor})
output = infer.get_output_tensor(0).data

# 後処理
out = output[0].transpose(1,2,0)
out = (out * 255).clip(0,255).astype(np.uint8)
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

cv2.imwrite("output/" + sys.argv[1], out)
