# astrbot_plugin_memorix

`astrbot_plugin_memorix` 是一套完整的记忆系统插件，围绕 Memorix 工作流设计：
消息写入 -> 混合检索 -> 记忆维护 -> 人物画像 -> 总结导入 -> WebUI 运维。

## 特色亮点

- 不是黑盒缓存：同一条记忆同时落在段落、关系、时间线上，检索结果可追溯、可解释。
- 不止“记住”，还能“修正”：支持 `protect / reinforce / restore`，可对记忆生命周期做主动干预。
- 不只聊天增强，还能运营：内置 WebUI，可直接查看图谱、回收站、来源链路和画像覆盖。
- 不绑死外部模型：即使未开启远程 embedding 也能启动并工作，便于本地和内网部署。
- 作用域可切换：可在平台共享、用户隔离、群隔离、最细粒度隔离之间按业务切换。

## 适合场景

- 长会话助手：让机器人在多轮对话里保持上下文连续性，而不是每轮“失忆”。
- 社群机器人：按群沉淀长期知识，避免不同群之间互相污染记忆。
- 人设驱动助手：通过人物画像和手工覆盖，让回复风格与事实记忆长期稳定。
- 运营型机器人：需要可视化检查记忆来源、纠错、恢复与审计的团队协作场景。

## 核心能力

- 自动消息写入：默认记录消息事件，沉淀为可检索记忆。
- 混合检索：向量召回 + 稀疏召回 + 图谱信号（可启用 PageRank 重排）。
- 记忆维护：支持 `protect / reinforce / restore`，并有后台衰减、冻结、剪枝。
- 人物画像：支持 registry、自动画像、手工覆盖（override）与清除。
- 总结导入：支持 transcript summary，并回写为结构化知识。
- WebUI 运维：图谱浏览、关系编辑、来源管理、画像管理、回收站恢复等。

## 从一条消息到一条记忆（工作流）

1. 收到消息并按 `scope.mode` 路由到作用域。
2. 写入 transcript 与段落元数据。
3. 提取实体/关系并写入图谱、向量索引、稀疏索引。
4. 用户后续提问时执行混合检索，将结果注入到 LLM System Prompt。
5. 后台任务定期执行维护（衰减、冻结、剪枝、画像刷新、总结任务）。

## 主要命令

- `/mem status` 查看作用域、WebUI、调度状态。
- `/mem query <query> [top_k]` 常规检索。
- `/mem time <time_from> [time_to] [query]` 时序检索。
- `/mem protect <hash_or_query> [hours]` 保护记忆（管理员）。
- `/mem reinforce <hash_or_query>` 强化记忆（管理员）。
- `/mem restore <hash> [relation|entity]` 从回收站恢复（管理员）。
- `/mem profile <person_keyword_or_id> [top_k]` 查询人物画像。
- `/mem profile_override <person_id> <text>` 覆盖画像（管理员）。
- `/mem profile_clear <person_id>` 清除画像覆盖（管理员）。
- `/mem summary_now [context_length]` 立即总结当前会话。
- `/mem ui` 返回 WebUI 地址。

## 作用域模式（scope.mode）

`scope.mode` 决定“哪些会话共享同一份记忆”：

- `platform_global`：同平台共享（默认，推荐）。
- `user_global`：同平台按用户隔离。
- `group_global`：同平台按群隔离，私聊自动退化到用户隔离。
- `umo`：按 `unified_msg_origin` 细粒度隔离（最严格）。

选择建议：

- 希望机器人跨群/跨会话持续记住平台语境，用 `platform_global`。
- 强隔离需求优先 `user_global` 或 `umo`。
- 以群为单位沉淀记忆，选 `group_global`。

## WebUI 说明

- 默认监听：`0.0.0.0:8092`
- 端口冲突：自动向后探测（`webui.port_fallback_max_tries`）
- 默认无鉴权（局域网），可开启 `webui.auth.enabled` + token
- 使用稳定的 `/api/*` 接口族，支持图谱与记忆运维操作

## 配置重点

- `embedding.enabled=false` 时，依然可运行（会使用本地 embedding 回退，不阻塞插件加载）。
- `embedding.openapi.base_url` 可不带 `/v1`，系统会自动补全。
- `embedding.openapi.model` 填 embedding 模型；`embedding.openapi.chat_model` 填 summary/chat 模型。
- `retrieval.enable_ppr` 是图算法重排，不需要额外 reranker 模型。

## 存储目录

默认存储路径：
`data/plugin_data/astrbot_plugin_memorix/scopes/<scope_key>/`

其中 `<scope_key>` 由 `scope.mode` 计算得到（例如 `aiocqhttp`、`aiocqhttp:user:123456` 等）。

## 依赖与性能

- 默认依赖包含 `faiss-cpu`，优先使用高性能向量检索。
- 若 `faiss-cpu` 初始化失败，会自动降级到 numpy 后备实现，保证功能可用。

## 特别感谢

- [ARC](https://github.com/A-Dawn)
