from openvino import Core
import numpy as np
import cv2
from tqdm import tqdm


def core_batch(compiled_model, frames):
	# frames: list of HWC uint8

	# ===== 前処理 =====
	arr = np.stack(frames, axis=0)          # [B,H,W,C]
	arr = arr.transpose(0,3,1,2)            # [B,C,H,W]
	arr = np.ascontiguousarray(arr, dtype=np.float32)
	arr *= (1.0 / 255.0)

	# ===== 推論 =====
	result = compiled_model({0: arr})
	output = result[compiled_model.output(0)]  # [B,3,H*,W*]

	# ===== 後処理 =====
	output = np.clip(output, 0, 1)
	output *= 255.0
	output = output.astype(np.uint8)

	# CHW → HWC
	output = output.transpose(0,2,3,1)

	return output  # shape: [B,H,W,C]


# モデル読み込み
ie = Core()
model = ie.read_model(model="models/realesrgan_x4plus_dynamic_batch.xml")
compiled_model = ie.compile_model(
	model,
	"GPU",
	config={"INFERENCE_PRECISION_HINT": "f16"}
)

print(model.input(0).partial_shape)
exit

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

BATCH = 4
frame_buffer = []

with tqdm(total=total_frames) as pbar:
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_buffer.append(frame)

		if len(frame_buffer) == BATCH:
			output_batch = core_batch(compiled_model, frame_buffer)

			for out_frame in output_batch:
				out.write(out_frame)
				pbar.update(1)

			frame_buffer.clear()

	# 余り処理
	if len(frame_buffer) > 0:
		output_batch = core_batch(compiled_model, frame_buffer)

		for out_frame in output_batch:
			out.write(out_frame)
			pbar.update(1)

cap.release()
out.release()
