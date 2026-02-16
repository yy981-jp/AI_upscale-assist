from openvino import Core
import numpy as np
import cv2
from tqdm import tqdm


def infer_single(infer_request, frame):
    # ===== 前処理 =====
    frame = cv2.resize(frame, (384, 384))
    arr = frame.transpose(2, 0, 1)          # HWC → CHW
    arr = np.expand_dims(arr, 0)            # [1,C,H,W]
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr *= (1.0 / 255.0)

    # ===== 推論 =====
    infer_request.infer({0: arr})
    output = infer_request.get_output_tensor(0).data  # [1,3,H*,W*]

    # ===== 後処理 =====
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)

    output = output[0].transpose(1, 2, 0)   # CHW → HWC
    return output


# ==========================
# モデル読み込み
# ==========================
ie = Core()
model = ie.read_model(model="models/realesrgan_384.xml")

# shape固定
model.reshape({model.input(0): [1, 3, 384,384]})

compiled_model = ie.compile_model(
    model,
    "NPU",
    {
        "INFERENCE_PRECISION_HINT": "f16",
        "PERFORMANCE_HINT": "THROUGHPUT"
    }
)

infer_request = compiled_model.create_infer_request()


# ==========================
# 入力動画
# ==========================
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

        out_frame = infer_single(infer_request, frame)
        out.write(out_frame)
        pbar.update(1)


cap.release()
out.release()
