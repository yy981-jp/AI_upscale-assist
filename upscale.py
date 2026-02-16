from openvino import Core
import numpy as np
import cv2
from tqdm import tqdm


def preprocess(frame):
    arr = frame.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr *= (1.0 / 255.0)
    return arr


def postprocess(output, scale):
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = output[0].transpose(1, 2, 0)
    output = output[4*scale:-4*scale, :, :]
    return output


def pad_360_to_368(frame):
    h, w, c = frame.shape
    assert h == 360 and w == 640

    pad_top = 4
    pad_bottom = 4

    padded = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded


# ==========================
# モデル読み込み
# ==========================
ie = Core()
model = ie.read_model(model="models/realesrgan_368640.xml")

# 入力サイズを368x640に固定
model.reshape({model.input(0): [1, 3, 368, 640]})

compiled_model = ie.compile_model(
    model,
    "NPU",
    {
        "INFERENCE_PRECISION_HINT": "f16",
        "PERFORMANCE_HINT": "THROUGHPUT"
    }
)


# ==========================
# 入力動画
# ==========================
input_path = "input.mp4"
output_path = "output.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("動画を開けない")

fps = cap.get(cv2.CAP_PROP_FPS)

scale = 4
out_width = 640 * scale
out_height = 360 * scale

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



num_requests = 3
free_requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
busy_requests = []

with tqdm(total=total_frames) as pbar:
    while True:

        # 空いてるrequestがあれば投入
        if free_requests:
            ret, frame = cap.read()
            if ret:
                req = free_requests.pop()

                frame_360p = cv2.resize(frame, (640, 360))
                padded = pad_360_to_368(frame_360p)
                arr = preprocess(padded)

                req.start_async({0: arr})
                busy_requests.append(req)
            else:
                break

        # 完了チェック
        finished = []
        for req in busy_requests:
            if req.wait_for(0):  # 完了
                output = req.get_output_tensor(0).data
                sr = postprocess(output, scale)
                out.write(sr)
                pbar.update(1)

                finished.append(req)

        for req in finished:
            busy_requests.remove(req)
            free_requests.append(req)

    # 残りを全部回収
    for req in busy_requests:
        req.wait()
        output = req.get_output_tensor(0).data
        sr = postprocess(output, scale)
        out.write(sr)
        pbar.update(1)

cap.release()
out.release()
