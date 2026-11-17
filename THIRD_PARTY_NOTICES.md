# 第三方来源

- 运行适配代码整理自 MINT 项目的已验证 Ascend/MJX 路径，保留原 MIT 声明：`licenses/MINT.txt`。
- 场景来自 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 的 `libero_spatial` 任务 `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`。保留其 MIT 声明：`licenses/LIBERO.txt`。模型内包含 LIBERO/robosuite 的机器人、物体及纹理资产，不将这些第三方资产重新声明为本项目原创。
- MuJoCo/MJX 3.2.7 为 Apache-2.0，作为固定 Python 依赖安装；未复制完整引擎。
- xla-npu 和 OpenXLA 的版本及来源见 `backend/versions.json`；后端补丁保留上游版权头与 Apache-2.0 声明。上游许可证仍适用于修改后的文件。

本仓库不包含模型权重、数据集、用户凭据、编译产物或旧实验日志。
