from openvino import Core
import numpy as np
import cv2
from tqdm import tqdm


def core(compiled_model, frame):
	# 前処理（HWC uint8前提）
	input_tensor = frame.transpose(2,0,1)[None, ...]
	input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
	input_tensor *= (1.0 / 255.0)

	# 推論
	result = compiled_model({0: input_tensor})
	output = result[compiled_model.output(0)]

	# 後処理
	output = output[0]                    # squeeze
	output = np.clip(output, 0, 1)
	output *= 255.0
	output = output.astype(np.uint8)
	output = output.transpose(1,2,0)      # HWC

	return output


# モデル読み込み
ie = Core()
model = ie.read_model(model="models/realesrgan_x4plus_dynamic.xml")
compiled_model = ie.compile_model(
	model,
	"GPU",
	config={"INFERENCE_PRECISION_HINT": "f16"}
)

# 入力動画
input_path = "input.mp4"
output_path = "output.mp4"

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
	raise RuntimeError("動画を開けない")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# x4モデル前提
scale = 4
out_width = width * scale
out_height = height * scale

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames) as pbar:
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# ===== 推論 =====
		output_frame = core(compiled_model, frame)
		# =================

		out.write(output_frame)

		pbar.update(1)

cap.release()
out.release()
