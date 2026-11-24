# sdetector: on-camera person detection for GK7205V300 + Majestic

What it is: a small program that finds people (or other COCO classes) on the camera's own SoC, with no cloud and no extra hardware. It takes frames from Majestic's VPSS channels in memory, finds moving regions with the IVE hardware, cuts a window around the target and checks it with YOLO26n (ncnn, fp16) - about 0.8 s per check. On an intrusion it prints one EVENT line and saves a JPEG with the boxes.

Tested on: GK7205V300, IMX335, 128 MB, kernel 4.9.37. RSS ~33 MB (peak ~41 MB), model 4.8 MB.

How to use: copy bin/sdetector, lib/ (the vendor ive_neo and mpi_neo libraries and libstubs.so) and model/ to /detector, stop other copies, then
  cd /detector; export LD_LIBRARY_PATH=./lib/ive_neo:./lib/mpi_neo LD_PRELOAD=./lib/libstubs.so
  ./bin/sdetector --model=./model/yolo26n-fp16_256_192 --scene=indoor
The secured area is the zone drawn in Majestic (read from /etc/majestic.yaml). Majestic must be running.

Memory on 128 MB boards is tight (see OpenIPC/firmware#2828): set isp.blkCnt to 5 (it gives back ~7 MB; the cost is a dropped frame per JPEG snapshot). The detector also skips a job while MemAvailable is below 15 MB (--min-avail).

Night: YOLO gets a grey picture while Majestic's night mode is on; the e-mail picture stays in colour. --scene indoor|outdoor and --range near|far are presets . If the sensor delivers far fewer frames than video1.fps (slow shutter at dusk), it writes a warning to the log.

Limits:  YOLO26 is Ultralytics AGPL-3.0.
