---
name: k1-camera-stream
description: Start the RGB + depth side-by-side MJPEG server on a K1 and view or snapshot it. Use for vision input to the chatbot or to check camera health.
---

# K1 camera stream

```bash
scp tools/depth_sbs_server.py boosterk1.<n>:~/
ssh -f boosterk1.<n> 'bash -c "source /opt/ros/humble/setup.bash; setsid nohup python3 ~/depth_sbs_server.py --port 8090 >~/depth_sbs.log 2>&1 </dev/null &"'
curl http://<ip>:8090/stats          # rgb_fps / depth_fps
curl -o frame.jpg http://<ip>:8090/snapshot.jpg   # frame for the vision model
```
- Browser: `http://<ip>:8090/`. Plain HTTP MJPEG, not websocket/ROS on the wire.
- Depth is float32 millimetres; needs fw 1.8.1.2 for ~25 Hz (1.6.2.2 gives ~4 Hz).
- Not persistent across robot reboots. Stop: `ssh boosterk1.<n> "pkill -f '[d]epth_sbs_server'"` (pattern with `[d]` so it does not kill the ssh shell).
