from openvino import Core
import numpy as np
import cv2
import sys

# モデル読み込み
ie = Core()
model = ie.read_model(model="models/realesrgan_x4plus.xml")
compiled_model = ie.compile_model(model=model, device_name="GPU")  # NPU, GPU, CPU自動選択

# 入力画像
img = cv2.imread("input/"+sys.argv[1]).astype(np.float32) / 255.0
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# サイズ調整（[N,C,H,W]）
input_tensor = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

# 推論
output_tensor = compiled_model([input_tensor])[0]

# 出力を画像形式に変換
output_img = np.squeeze(output_tensor)
output_img = np.clip(output_img, 0, 1)
output_img = (output_img * 255).astype(np.uint8)
output_img = np.transpose(output_img, (1, 2, 0))  # CHW→HWC
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

# 保存
cv2.imwrite("output/"+sys.argv[1], output_img)
