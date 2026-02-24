# astrbot_plugin_memorix

基于 A_memorix 的 AstrBot 记忆插件，包含：
- 自动消息写入与检索注入
- 记忆维护（protect / reinforce / restore）
- 人物画像（query / override）
- 总结导入（summary_now）
- 内嵌 WebUI

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

## 作用域模式

`scope.mode` 决定“哪些消息共享同一份记忆”：

- `platform_global`：同一平台共用记忆（默认，推荐大多数场景）。
- `user_global`：同一平台下按用户隔离，每个用户独立记忆。
- `group_global`：同一平台下按群隔离；群聊按群共享，私聊自动按用户隔离。
- `umo`：按 `unified_msg_origin` 隔离，粒度最细，隔离最严格。

选择建议：

- 机器人希望“同平台越聊越懂你们整体语境”选 `platform_global`。
- 强隐私/强隔离需求优先选 `user_global` 或 `umo`。
- 主要在群聊中使用且希望“每个群形成自己的记忆”选 `group_global`。

## 依赖

默认依赖包含 `faiss-cpu` 优先使用 Faiss 高性能向量检索。

若环境中 `faiss-cpu` 初始化失败，插件会自动降级到 numpy 向量后备实现，保证功能可用。

## 特别感谢

- [ARC](https://github.com/A-Dawn)
