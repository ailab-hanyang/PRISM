# How to download PRISM-Dataset

The PRISM dataset is hosted on Hugging Face and can be downloaded using the provided script.

```bash
pip install huggingface_hub
python download_dataset.py --local_dir /your/dataset/path
```

---

# Dataset layout
The dataset is distributed as per-session ZIP files, organised at the top level by train / validation split:

```bash
PRISM-Dataset/
├── README.md
├── labels.json                          # frame-level material/state labels (whole dataset)
├── calibration/                         # camera intrinsics + inter-sensor extrinsics
│   ├── 0106_calibration.yaml
│   ├── …
│   └── 0327_calibration.yaml
├── train/
│   ├── 0106_dataset.zip
│   ├── 0112_dataset.zip
│   ├── 0119_1_dataset.zip
│   ├── …
│   └── 0327_11_dataset.zip
└── val/
    ├── 0112_dataset.zip          # last 30 % of one snow session (intra-session split)
    ├── 0119_1_dataset.zip
    └── …
```
labels.json and calibration/ are kept outside the per-session ZIPs so they can be inspected without unpacking any archive.

Some session names appear in both train/ and val/: those are sessions that were split within a recording (intra-session split) to preserve training-set coverage of rare classes such as slush and snow-covered, which originate from a small number of capture sites. See paper Appendix E for the full split protocol.

Inside each ZIP, the layout is:

```bash
{session_anon}/                              # e.g. 0106_dataset/
├── sequence_001/
│   ├── rgb/                                 # *.png   RGB images
│   ├── polar/
│   │   ├── 0d/                              # *.png   polariser at  0°
│   │   ├── 45d/                             # *.png   polariser at 45°
│   │   ├── 90d/                             # *.png   polariser at 90°
│   │   └── 135d/                            # *.png   polariser at 135°
│   └── lidar_accum_scan/                    # *.pcd   accumulated scan
├── sequence_002/
│   └── …
└── vehicle_state/                           # session-level (NOT per-sequence)
```

File naming. All files share the same nanosecond Unix timestamp as the filename stem (e.g. 1736200123456789012.png). RGB, the four polarisation images, and the corresponding accumulated LiDAR scan for the same instant share a stem, which is also how the train / val split is internally applied to intra-session sessions.

Polarisation. PRISM ships raw four-orientation polariser-resolved intensities rather than pre-computed Stokes / AoLP / DoLP maps. This keeps the release closer to the sensor and lets users compute polarimetric quantities under their own conventions. The Stokes-to-AoLP/DoLP formulas are given in the paper (Section 3) and a reference implementation is included in the code release.

LiDAR. The lidar_accum_scan/ directory contains LiDAR-inertial SLAM-accumulated point clouds (one PCD per frame) produced by the pipeline described in paper Section 4.3. These accumulated clouds are already deskewed, aligned to a common ground frame, and ICP-refined on static ground segments, so they can be used directly. They serve as the source of dense road-surface elevation supervision for Track 2: elevation BEV is rasterised from the PCD onto the 10 cm × 10 cm grid at training and evaluation time. Single-sweep raw LiDAR is not released at this time.

Vehicle state. vehicle_state/ is a session-level directory (not per-sequence) containing RTK-INS pose and synchronised vehicle-bus signals.

Labels. Frame-level surface material and state labels are released as a single labels.json at the repository root, indexed by (session_id, sequence_id, timestamp).

Calibration. Camera intrinsics and inter-sensor extrinsics (RGB ↔ polarisation, camera ↔ LiDAR) are released as YAML files under the calibration/ directory at the repository root.

Privacy. Faces and licence plates throughout the released imagery are replaced in-place by Gaussian blur (OpenCV cv2.GaussianBlur, 31×31 kernel) on the raw polariser-resolved intensities. The road region is unaffected. See paper Appendix K.