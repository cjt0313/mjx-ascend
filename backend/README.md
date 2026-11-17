# Ascend 后端源码构建

本目录只保存当前可复现实现的上游版本、源码补丁和构建入口，不包含旧实验、预编译插件、下载包或构建缓存。MuJoCo/MJX 的运行入口和场景见仓库主文档。

## 固定依赖

版本见 `versions.json`：xla-npu 固定到 `b3a3e9c498ceb88278d9ba3d9e4d202bb1b8c24a`，OpenXLA 固定到 `a7e58876b73dfa9bb6bc785f6981a332c133bb55`。该组合使用 Python 3.11、JAX/JAXLIB 0.8.1、Bazel 7.7.0、JDK 21、CANN 9.0.0，选择 ACLNN 后端（`NPU_AUTO_FUSE_BACKEND=0`）。已验证机器为 Linux x86_64；其他架构尚未验证。

新机器需预先安装适配硬件的 Ascend 驱动与 CANN、Git、Python 3.11（提供共享库 `libpython3.11.so`）、JDK 21 和 Bazel 7.7.0，以及常规 C/C++ 构建工具。请从对应供应商官方渠道安装这些工具；脚本不会修改系统安装、隐式安装依赖或替换驱动。OpenXLA 的 Bazel 规则会下载固定依赖，因此首次构建需要访问其上游源和依赖下载站点。

## 构建

先激活仓库的 Python 环境，并加载本机 CANN 的 `set_env.sh`。在本项目所在持久文件系统中选择构建目录；不要使用系统临时目录。

```bash
# 根据本机实际安装位置调整。
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export JAVA_HOME=/path/to/jdk-21
export PATH="$JAVA_HOME/bin:$PATH"

python backend/build.py --work-dir /persistent/path/mjx-ascend-build --prepare-only
python backend/build.py --work-dir /persistent/path/mjx-ascend-build --jobs 2
```

`--prepare-only` 完成固定版本 clone、应用补丁、校验源码与创建 XLA 源码链接，不进行编译。已有目录的 commit 不符时脚本会报错，不会强制覆盖本地修改；请选用新目录。可通过 `--bazel /absolute/path/to/bazel` 指定 Bazel。

构建产物为 `<work-dir>/output/pjrt_c_api_npu_plugin.so`。构建源码、Bazel 缓存、临时文件、Ascend 缓存与日志均放在 `--work-dir` 下。首次构建可能较慢，磁盘需要为 OpenXLA 完整源码和编译缓存预留空间。

## 运行环境

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source backend/runtime.sh /persistent/path/mjx-ascend-build/output/pjrt_c_api_npu_plugin.so
# 接着使用仓库主文档的运行命令。
```

`runtime.sh` 设置 `MJX_NPU_PLUGIN_PATH` 和 Python 共享库搜索路径；运行入口仍需注册 PJRT 插件。`ldd` 检查显示插件直接依赖 `libpython3.11.so.1.0`、系统 C/C++ 运行库等。Ascend 库由插件动态加载，单独检查 `ldd` 无法证明 CANN 已就绪；必须加载 CANN 环境并实际运行 NPU 验证。插件不是能够脱离 CANN/驱动独立运行的文件。

## 源码一致性与验证边界

`xla-npu.patch` 包含 40 个必要修改／新增源文件。`source-sha256.json` 校验 1,384 个上游及补丁后文件，保证构建输入固定。其中编译器文件 SHA256 为 `13316d09ca85153c5feb6b212fde95b21cc8fa45a3fe8b27c40f4ca207993116`。

旧上游安装脚本 `build/build.sh` 的本地调整不影响本目录构建方式，因此不进入补丁及校验清单。整理前已核对原记录中除安装脚本以外的全部 1,386 个源码哈希。整理后移除了未使用的 pose 实验 thunk、两份源码及四处构建／发射器接入；这四份文件的哈希已更新，清单现在包含 1,384 个文件。编译器数值优化代码保持原已验证版本；没有携带后续尚未通过连续验证的 gather-index 优化。

因此本补丁是清理后的后端，需要重新构建插件并运行回归；不能将旧插件的测试结果当成新插件已验证的证据。

已完成：在干净上游源码快照应用补丁并核对全部 1,384 个文件哈希。源码重建不承诺插件二进制逐字节一致，编译路径和工具链会影响文件哈希。已在隔离的固定版本源码副本上通过本目录 `build.py` 完成编译，构建前后全部 1,384 个源码哈希一致；使用了现有依赖下载缓存。空缓存新机器首次下载流程尚未验证。功能与性能仍须以仓库运行入口的新验证结果为准。

上游 xla-npu 使用 Apache-2.0，原许可证保存于 `LICENSE.xla-npu`，补丁保留源文件版权声明。OpenXLA 的许可证随独立下载源码保留。未将预编译插件或 CANN SDK 纳入仓库。
