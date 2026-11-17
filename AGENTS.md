# 项目约定

- 保持固定场景的物理语义；不为通过测试削减几何、容量、迭代次数或数值门槛。
- 运行代码在 `mjx_ascend/`，场景在 `scene/`，构建与输出放在被忽略的持久目录。
- 禁止 CPU 静默回退。CPU 对照只能在回放前独立生成。
- 修改前运行相关回归；修改后执行 CPU 单测、静态检查和适当 NPU 验证。
- 文档以中文为主，明确已测能力与限制；不提交实验日志、二进制插件、缓存和凭据。
- 后端源码固定上游 commit + patch；改 patch 后更新源文件 hash 清单，验证可应用并重建。
- 提交使用 intent 标题，按需附 Tested/Not-tested/Constraint/Confidence/Scope-risk 等 Lore trailers。
