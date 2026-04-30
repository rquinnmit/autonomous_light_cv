# VisionBeam — hardware inventory

## Moving Stage Light


| Item                  | Details                                                                                                                                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model (sticker)**   | `ZQ02360`                                                                                                                                                                                                                                 |
| **OEM / supplier**    | Shenzhen Zhuoqiong Technology Co., Ltd. (typical); sold under **UKING** and other names                                                                                                                                                   |
| **Type**              | Mini LED moving head — RGBW spot, **LED ring** around lens, 7 colors / gobos, prism (marketing: 120W LED module, ~150W fixture class)                                                                                                     |
| **Power**             | AC 100–240V, 50/60Hz                                                                                                                                                                                                                      |
| **Control surface**   | DMX IN, DMX OUT (typically **3-pin XLR**), power, **4-digit display**, **MENU / UP / DOWN / ENTER**                                                                                                                                       |
| **DMX personalities** | **13 CH** or **15 CH** (set on fixture; must match software fixture profile)                                                                                                                                                              |
| **Pan / tilt (spec)** | **540°** pan, **270°** tilt (with 16-bit-style fine channels in DMX table)                                                                                                                                                                |
| **References**        | [Alibaba listing — model ZQ02360](https://www.alibaba.com/product-detail/U-King-120W-LED-RGBW-4_1601469261848.html); [ADJ forum — MyDMX profile / 15 CH vs 16 CH cousin](https://forums.adj.com/topic/uking-zq02360-120w-led-moving-head) |


**TBD (fill in when known):** exact **DMX start address** in use; whether you run **13** or **15** channel mode; **USB-to-DMX adapter** make/model and serial port path (e.g. macOS `/dev/tty.usbserial-`*).

---

## Webcam


| Item         | Details                                                                 |
| ------------ | ----------------------------------------------------------------------- |
| **Model**    | **Logitech HD 1080p** USB webcam                                        |
| **Rig plan** | **Mounted on the moving head** (camera pans and tilts with the fixture) |


---

## Rest of stack (project standard — confirm what you own)

From `[README.md](README.md)` for a full live system:


| Item                       | Role                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| **USB ↔ DMX512 interface** | Enttec Open DMX–class serial device (see `visionbeam/dmx.py`); USB-C vs USB-A is only the computer plug |
| **Computer**               | M5 Max Macbook Pro w/ 64 GB RAM                                                                         |
| **ArUco markers (×4)**     | Printed markers, known floor layout, for `calibration/homography.json`                                  |
| **Measuring / layout**     | Tape or surveyed points for homography and light triangulation                                          |


