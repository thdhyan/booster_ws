"""Convert a K1 stereo bag (rectified NV12 pair) into a cuVSLAM EDEX folder.
Usage: python3 bag2edex.py <bag_dir> <out_dir> [left_topic right_topic]
Writes out/images/cam{0,1}/NNNNNN.png (Y plane = mono8), stereo.edex, frame_metadata.jsonl,
odom.tum (from /odometer_state, for comparison)."""
import sys, os, json
import numpy as np, cv2
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, out = sys.argv[1], sys.argv[2]
LT = sys.argv[3] if len(sys.argv) > 3 else '/boostercamera/head/rgb'
RT = sys.argv[4] if len(sys.argv) > 4 else '/boostercamera/head/right/rgb'
LI, RI = LT + '/camera_info', RT + '/camera_info'
for c in (0, 1):
    os.makedirs(f'{out}/images/cam{c}', exist_ok=True)

r = SequentialReader()
r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = {LT, RT, LI, RI, '/odometer_state'}
cls = {t: get_message(types[t]) for t in want if t in types}

pend = {LT: {}, RT: {}}; info = {}; frames = []; odom = []

def y_plane(m):
    a = np.frombuffer(bytes(m.data), np.uint8)
    return a[:m.height * m.width].reshape(m.height, m.width)  # NV12: Y first

while r.has_next():
    t, data, _ = r.read_next()
    if t not in cls:
        continue
    m = deserialize_message(data, cls[t])
    if t in (LI, RI):
        info.setdefault(t, m)
    elif t == '/odometer_state':
        odom.append((_, m))
    else:
        st = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        other = RT if t == LT else LT
        if st in pend[other]:
            o = pend[other].pop(st)
            L, R = (m, o) if t == LT else (o, m)
            i = len(frames)
            for c, im in ((0, L), (1, R)):
                cv2.imwrite(f'{out}/images/cam{c}/{i:06d}.png', y_plane(im))
            frames.append(st)
        else:
            pend[t][st] = m
            if len(pend[t]) > 50:  # drop unmatched old frames
                pend[t].pop(min(pend[t]))

li, ri = info[LI], info[RI]
fx, fy, cx, cy = li.p[0], li.p[5], li.p[2], li.p[6]
base = -ri.p[3] / ri.p[0]
print(f'pairs={len(frames)} size={li.width}x{li.height} fx={fx:.3f} fy={fy:.3f} '
      f'cx={cx:.3f} cy={cy:.3f} baseline={base:.4f} m  R.P={list(ri.p)}')

def cam(tx):
    return {'intrinsics': {'distortion_model': 'pinhole', 'distortion_params': [],
                           'focal': [fx, fy], 'principal': [cx, cy], 'size': [li.width, li.height]},
            'transform': [[1, 0, 0, tx], [0, 1, 0, 0], [0, 0, 1, 0]]}

edex = [{'version': '0.9', 'frame_start': 0, 'frame_end': len(frames) - 1,
         'cameras': [cam(0.0), cam(base)]},
        {'sequence': [[f'images/cam{c}/{i:06d}.png' for i in range(len(frames))] for c in (0, 1)],
         'frame_metadata': 'frame_metadata.jsonl'}]
json.dump(edex, open(f'{out}/stereo.edex', 'w'), indent=1)
with open(f'{out}/frame_metadata.jsonl', 'w') as f:
    for i, st in enumerate(frames):
        f.write(json.dumps({'frame_id': i, 'cams': [
            {'id': c, 'filename': f'images/cam{c}/{i:06d}.png', 'timestamp': st} for c in (0, 1)]}) + '\n')

if odom:
    m0 = odom[0][1]
    fields = [f for f in m0.get_fields_and_field_types()]
    print('odometer fields:', m0.get_fields_and_field_types())
    with open(f'{out}/odom_raw.txt', 'w') as f:
        for ts, m in odom[::10]:
            f.write(f'{ts} ' + ' '.join(str(getattr(m, k)) for k in fields) + '\n')
