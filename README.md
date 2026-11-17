# MJX Ascend

在 Ascend NPU 上执行固定 LIBERO 场景的 MuJoCo MJX 物理回放，并逐步对照原生 MuJoCo CPU 结果。

本仓库只保留一条运行路径：**MJX 3.2.7 + JAX 0.8.1 + ACLNN PJRT 后端 + 75 步固定控制序列**。包含运行代码、无损压缩场景、数值回归测试和后端构建补丁，不包含历史实验入口、失败候选、模型权重或编译缓存。

## 支持范围

- 单环境、Euler 积分、FP32 MJX，CPU 对照为 MuJoCo 3.2.7 FP64。
- 固定场景 271 个几何体、43 个速度自由度、62,709 个接触槽；接触槽容量不等于实际接触数。
- 每步验证位置、姿态、速度、时间、有效接触数量和 NPU 设备驻留。
- CPU 对照在 NPU 回放之前独立生成；回放期间拒绝受监控的原生 MuJoCo 调用，禁止静默 CPU 回退。
- 圆柱–box / 圆柱–mesh 只接受严格分离状态；触碰、穿透和无效查询拒绝执行。

这不是完整 LIBERO 控制器/策略闭环，也不代表任意场景兼容。固定轨迹没有有效 BOX–MESH 接触，不能据此证明通用 mesh 接触行为。当前目标是可复现正确性，尚未达到实用仿真速度。

## 环境

已使用的平台：Linux x86_64、Ascend 910B2C、CANN 9.0.0、Python 3.11。驱动与 CANN 必须由运行平台提供；pip 不安装设备驱动。

在有持久存储的目录 clone 并创建环境：

```bash
git clone https://github.com/cjt0313/mjx-ascend.git
cd mjx-ascend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

`set_env.sh` 路径按 CANN 安装位置调整。需要提供 `libpython3.11.so` 的 Python 安装；若环境缺失共享库，先安装匹配的 Python，而不是链接其他 Python ABI。

## 构建后端

首次构建需要 JDK 21、Bazel 7.7.0、C/C++ 编译工具和联网下载 OpenXLA 依赖。构建可能耗时较长、占用较多磁盘。详情见 [后端构建说明](backend/README.md)。所有源码和缓存位于显式指定目录：

```bash
python backend/build.py --work-dir "$PWD/.cache/backend" --jobs 4
export MJX_NPU_PLUGIN_PATH="$PWD/.cache/backend/output/pjrt_c_api_npu_plugin.so"
```

后端版本、补丁与源码校验值全部固定。仓库不提交 `.so`。已有匹配构建可直接设置 `MJX_NPU_PLUGIN_PATH`，回放报告记录实际插件 hash；不能用任意插件冒充已验证版本。

## 运行

从仓库根目录执行，输出目录必须是新目录：

```bash
bash scripts/run.sh --output "$PWD/outputs/replay75" \
  --steps 75 --reference "$PWD/scene/reference.npz" --stage-timing
```

- `--reference`：同时比较随包提供的已验证 NPU 轨迹，要求 qpos/qvel/time 逐字节一致；CPU 数值门槛仍独立执行。
- `--stage-timing`：保存同步阶段计时，会增加测量开销；测总步延迟时应另开输出目录并去掉该参数。
- `--device 0`：选择可见 NPU 列表中的第 0 张，不是 `npu-smi` 显示的物理编号；进程之间的设备资源由运行平台管理。
- `--steps 1`：仅作单步检查，不等于完整 75 步验收。
- 失败时返回非零退出码，并尽可能在 `report.json` 保留错误和已完成状态。

程序自动校验并解压 `scene/model.mjb.gz` 到输出目录，不需要安装 LIBERO、访问旧工作区或下载 mesh 资产。仅解压文件约 251 MiB，请为运行目录预留空间。

## 结果判断

`report.json` 顶层 `verdict == "passed"` 才表示所请求步数通过；完整验收还要求 `steps == 75`。

| 指标 | 门槛 |
|---|---:|
| 非四元数 qpos 最大绝对误差 | ≤ 1e-5 |
| 姿态夹角误差 | ≤ 1e-4 rad |
| qvel 最大绝对误差 | ≤ 1e-4 |
| 仿真时间误差 | ≤ 1e-6 |
| 每步有效接触数量 | 与 CPU 参考一致 |
| 受监控原生调用次数 | 0 |
| 指定 NPU 参考轨迹时 | qpos/qvel/time 字节一致 |

第一个物理步包含 JIT 编译开销。热运行统计应排除首步，区分阶段同步计时与无阶段计时，并报告硬件、插件 hash、median/p95。物理步吞吐不能当作策略 FPS，多环境摊销延迟不能当作单环境延迟。

本次仓库整理后的实际验证结果见 [验证说明](docs/验证说明.md)。

## 开发检查

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
python -m compileall -q mjx_ascend tests backend/build.py
# 可选开发工具，不属于运行依赖：python -m pip install ruff
python -m ruff check mjx_ascend tests backend/build.py
```

测试覆盖保守剔除/溢出回退、候选打包、接触模板、box 接触修正、圆柱数值处理、椭圆锥 Hessian、数据完整性与失败拒绝。通过 CPU 单元测试不能替代真实 NPU 75 步回放。

## 目录

```text
mjx_ascend/       固定回放入口、场景加载、数值验证与必要适配
scene/           压缩模型、初始状态、75 步控制及参考轨迹
backend/         固定上游版本、源码补丁、构建与运行环境脚本
tests/           与当前运行路径相关的回归测试
scripts/run.sh   可重复调用的运行入口
docs/            中文架构、数据与验证说明
licenses/        第三方许可证
```

详细内容：[实现说明](docs/实现说明.md) · [场景说明](docs/场景说明.md) · [第三方来源](THIRD_PARTY_NOTICES.md)。
