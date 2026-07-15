# Assets

Nothing needs to be downloaded manually into this folder.

- **Robot (Unitree Go2):** the environment uses `UNITREE_GO2_CFG` from
  `isaaclab_assets`, which pulls the official USD from the Isaac Sim asset library
  (Nucleus / cloud) on first use. To use a custom robot, convert its URDF with
  Isaac Lab's `convert_urdf.py` tool, drop the USD in `assets/robot_usd/`, and point an
  `ArticulationCfg(spawn=sim_utils.UsdFileCfg(usd_path=...))` at it in
  `source/tasks/locomotion_env.py`.
- **Terrains:** generated procedurally at startup by `TerrainGeneratorCfg`
  (see `source/tasks/terrain_curriculum.py`) — no terrain meshes are stored on disk.
