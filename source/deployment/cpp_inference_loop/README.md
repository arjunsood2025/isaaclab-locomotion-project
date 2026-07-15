# C++ inference loop

Fixed-rate (default 50 Hz) control loop that loads the TorchScript policy exported by
`source/policies/export_policy.py` and reports loop latency + deadline misses.
Robot I/O is stubbed behind `read_robot_state()` / `write_joint_targets()`; on real
hardware those become vendor SDK calls (e.g. `unitree_sdk2` LowState/LowCmd).

## Build

```bash
# download libtorch (CPU build is enough for a 3-layer MLP):
# https://pytorch.org/get-started/locally/ -> LibTorch -> cxx11 ABI
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

## Run

```bash
./policy_loop ../../../exported/policy.pt 48 50
# <model> <obs_dim> <rate_hz>; runs 20 s and prints latency percentiles
```

Expected output on a laptop CPU: mean well under 1 ms for the blind MLP policy —
comfortably inside the 20 ms budget of a 50 Hz loop.
