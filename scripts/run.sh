#!/usr/bin/env bash
# 请先激活 Python 3.11 环境，并 source CANN 的 set_env.sh。
set -eo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MJX_NPU_PLUGIN_PATH:?请设置 MJX_NPU_PLUGIN_PATH 为已构建的 PJRT 插件路径}"
python_bin="${MJX_PYTHON:-python}"
python_lib="$($python_bin -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export LD_LIBRARY_PATH="$python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export JAX_USE_SHARDY_PARTITIONER=false
unset USE_OLD_CANN_STYLE
exec "$python_bin" -m mjx_ascend --plugin "$MJX_NPU_PLUGIN_PATH" --scene "$repo_dir/scene" "$@"
