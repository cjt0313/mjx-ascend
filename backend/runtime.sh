#!/usr/bin/env bash
# Source after the installed CANN set_env.sh and Python environment activation.
# Usage: source backend/runtime.sh /persistent/backend-build/output/pjrt_c_api_npu_plugin.so
if [[ $# != 1 || ! -f "$1" ]]; then
    printf 'Usage: source backend/runtime.sh /absolute/path/to/plugin.so\n' >&2
    return 2 2>/dev/null || exit 2
fi
export MJX_NPU_PLUGIN_PATH="$(realpath -- "$1")"
# The plugin links libpython and loads Ascend libraries dynamically at runtime.
_mjx_python_lib="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export LD_LIBRARY_PATH="$_mjx_python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _mjx_python_lib
