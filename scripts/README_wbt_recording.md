# WBT 测试与录制脚本说明

这个 README 说明如何使用仓库里的两个脚本，分别在 `IsaacSim` 和 `MuJoCo` 中运行 `G1 29-DoF WBT` 模型，并录制本地视频。

当前配套脚本：

- `scripts/run_wbt_isaacsim_record.sh`
- `scripts/run_wbt_isaacsim_record.py`
- `scripts/run_wbt_mujoco_official.sh`
- `scripts/run_wbt_mujoco_official_policy.py`

默认模型：

- `/root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.pt`

对应 ONNX：

- `/root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.onnx`

## 1. IsaacSim WBT 运行与录制

### 用途

这条脚本走的是仓库内部的 `eval_agent` 评估逻辑，但绕过了当前旧 checkpoint 在 CLI 层的 `tyro` 兼容问题，直接调用内部 API 在 `IsaacSim` 中评估并录制视频。

### 命令

```bash
bash scripts/run_wbt_isaacsim_record.sh \
  /root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.pt \
  3000
```

参数说明：

- 第 1 个参数：`.pt` checkpoint 路径
- 第 2 个参数：`max_eval_steps`

如果省略参数，脚本会使用默认 checkpoint 和 `3000` 步。

### 输出位置

脚本会创建一个新的目录：

```text
logs/WholeBodyTrackingOfficial/<UTC时间戳>-g1_29dof_wbt_isaacsim_official/
```

视频会保存在：

```text
logs/WholeBodyTrackingOfficial/<UTC时间戳>-g1_29dof_wbt_isaacsim_official/renderings_training/
```

### 特点

- `headless=True`
- `num_envs=1`
- 本地离线录制，不上传 wandb
- 视频 overlay 关闭
- 默认保留 checkpoint 原始 motion 配置

如果你想直接改 Python 脚本参数，可以运行：

```bash
source scripts/source_isaacsim_setup.sh
python scripts/run_wbt_isaacsim_record.py --help
```

## 2. MuJoCo WBT 官方 sim2sim 运行与录制

### 用途

这条脚本走的是更接近正式部署的路径：

1. 用 `run_sim.py` 启动 `MuJoCo + bridge + video recorder`
2. 用 `holosoma_inference` 的 `WholeBodyTrackingPolicy` 启动官方 policy
3. 自动执行：
   - stiff startup hold
   - start policy
   - start motion clip
4. 等待模拟器按固定时长自然退出并收尾写视频

### 命令

```bash
bash scripts/run_wbt_mujoco_official.sh \
  /root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.pt \
  30
```

参数说明：

- 第 1 个参数：`.pt` checkpoint 路径
- 第 2 个参数：正式 motion runtime，单位秒

注意：

- 这个脚本会自动把 `.pt` 路径转换成同目录下的 `.onnx`
- simulator 的最大运行时间会自动设置为 `motion_runtime + 12` 秒，用于给 stiff hold、启动和视频收尾留空间

### 输出位置

脚本会创建一个新的目录：

```text
logs/WholeBodyTrackingOfficial/<UTC时间戳>-g1_29dof_wbt_mujoco_official/
```

日志：

- `sim.log`
- `policy.log`

视频：

```text
logs/WholeBodyTrackingOfficial/<UTC时间戳>-g1_29dof_wbt_mujoco_official/renderings_training/
```

### 已验证的短录制示例

短验证成功样例：

```text
logs/WholeBodyTrackingOfficial/20260409_020553-g1_29dof_wbt_mujoco_official/
```

正式录制成功样例：

```text
logs/WholeBodyTrackingOfficial/20260409_021014-g1_29dof_wbt_mujoco_official/
```

对应视频：

```text
logs/WholeBodyTrackingOfficial/20260409_021014-g1_29dof_wbt_mujoco_official/renderings_training/episode_1_1775700670.mp4
```

## 3. 环境要求

### IsaacSim

要求 `hssim` 环境可用。

常用检查：

```bash
source scripts/source_isaacsim_setup.sh
python -c "import isaacsim; from isaaclab.app import AppLauncher; print('ISAACSIM_OK')"
```

### MuJoCo

要求 `hsmujoco` 环境可用，并且已经能 import：

- `unitree_interface`
- `netifaces`
- `sshkeyboard`
- `pinocchio`

常用检查：

```bash
source scripts/source_mujoco_setup.sh
python -c "import mujoco, unitree_interface, netifaces, sshkeyboard, pinocchio; print('MUJOCO_WBT_OK')"
```

如果缺依赖，可以在 `hsmujoco` 环境里安装：

```bash
source scripts/source_mujoco_setup.sh
pip install netifaces sshkeyboard pin
```

## 4. 常见问题

### 1. MuJoCo 脚本运行了，但没有视频

先看：

- `sim.log`
- `policy.log`

重点检查：

- `policy.log` 里是否出现 `Policy enabled after stiff hold`
- `policy.log` 里是否出现 `Motion clip started`
- `sim.log` 里是否出现 `Successfully saved video file`

### 2. MuJoCo 提示找不到 ONNX

`run_wbt_mujoco_official.sh` 需要 checkpoint 同目录下存在同名 `.onnx` 文件。

例如：

- `model_36000.pt`
- `model_36000.onnx`

### 3. IsaacSim 的旧 checkpoint CLI 起不来

不要直接用：

```bash
python src/holosoma/holosoma/eval_agent.py --checkpoint=...
```

当前旧 checkpoint 会在 CLI 的 `tyro` 解析层报错。请使用：

```bash
bash scripts/run_wbt_isaacsim_record.sh ...
```

## 5. 推荐用法

如果你只是想快速生成一条可检查的视频：

```bash
bash scripts/run_wbt_isaacsim_record.sh \
  /root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.pt \
  3000

bash scripts/run_wbt_mujoco_official.sh \
  /root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.pt \
  30
```

前者用于看训练原生 `IsaacSim` 表现，后者用于看正式 `MuJoCo sim2sim` 表现。
