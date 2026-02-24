# astrbot_plugin_memorix

基于 A_memorix 机制的 AstrBot 记忆插件，包含：
- 自动消息写入与检索注入
- 记忆维护（protect / reinforce / restore）
- 人物画像（query / override）
- 总结导入（summary_now）
- 内嵌 WebUI（迁移自 A_memorix）

## 主要命令

- `/mem status`
- `/mem query <query> [top_k]`
- `/mem time <time_from> [time_to] [query]`
- `/mem protect <hash_or_query> [hours]`
- `/mem reinforce <hash_or_query>`
- `/mem restore <hash> [relation|entity]`
- `/mem profile <person_keyword_or_id> [top_k]`
- `/mem profile_override <person_id> <text>`
- `/mem profile_clear <person_id>`
- `/mem summary_now [context_length]`
- `/mem ui`

## WebUI

- 默认监听：`0.0.0.0:8092`
- 端口冲突：自动尝试后续端口（最多 `webui.port_fallback_max_tries` 次）
- 默认无鉴权（局域网开放），请按需在配置中开启 `webui.auth.enabled` 并设置 token。

## 存储

默认存储路径：
`data/plugin_data/astrbot_plugin_memorix/scopes/<platform>/`

## 依赖

见 `requirements.txt`。默认即可运行（numpy 后备向量索引）。

如需高性能向量检索，可额外安装：

```bash
pip install faiss-cpu
```
