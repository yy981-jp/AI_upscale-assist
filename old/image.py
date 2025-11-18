from openvino.runtime import Core
import cv2
import numpy as np

# OpenVINO初期化
core = Core()

# モデル読み込み（例: intelのサンプル super-resolution モデル）
model_path = "single-image-super-resolution-1032.xml"  # IR形式（.xml + .bin）
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")  # AI Boost自動利用

# 入力レイヤー取得
input_layer = compiled_model.input(0)

# 画像読み込み（実写画像）
img = cv2.imread("input.jpg")
img = cv2.resize(img, (480,270))  # モデルの入力サイズに合わせる

# 前処理（NCHW・BGR・float32・0~1スケーリング）
input_tensor = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32) * 255.0

# 推論
result = compiled_model([input_tensor])[compiled_model.output(0)]

# 出力後処理（CHW → HWC・uint8）
output_image = np.clip(result[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

# 保存
cv2.imwrite("output.jpg", output_image)

print("高画質化完了！output.jpgに保存したよ。")
print("入力画像（正規化前）の平均：", img.mean())
print("入力テンソルの平均：", input_tensor.mean())
print("出力テンソルの最小値:", result.min())
print("出力テンソルの最大値:", result.max())
print("出力テンソルの平均:", result.mean())
