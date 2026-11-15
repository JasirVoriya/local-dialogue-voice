# 本机台词配音工具

在 Apple Silicon Mac 本地识别视频台词、编辑台词并克隆原人物声音。支持**自由台词配音**和**时间轴台词替换**，仅生成音频，不修改视频画面。

时间轴模式下，每句新台词使用**对应原句的短音频及准确原话**创建独立 Qwen3-TTS Voice Clone Prompt，不再拼接整段视频原声。该策略提供局部语音上下文，**不保证**音色、方言、情绪、节奏或停顿精确复现；本次工程测试没有验证真实声音效果提升。

视频、识别结果、参考和生成音频始终保存在本机。仅首次安装依赖和下载模型需要联网；没有远程推理或媒体上传到第三方服务的功能。

## 安装与启动

需要 Apple Silicon Mac、Python **3.12**、uv、FFmpeg；启动器通过 uv 创建独立 TTS/ASR 环境。

```bash
brew install uv ffmpeg
git clone https://github.com/JasirVoriya/local-dialogue-voice.git
cd local-dialogue-voice
chmod +x 启动.command
./启动.command
```

也可在 Finder 双击 `启动.command`。首次准备建议预留至少 12 GB 磁盘空间，长视频的 PCM 音轨需要额外空间。关闭终端会停止服务。服务继续只监听 `127.0.0.1`；首次模型下载使用 ModelScope，已有 `.download-complete` 标记的模型不会重复下载。

模型、项目资料和媒体保存在：

```text
~/Library/Application Support/TaiCiPeiyin/
```

自由模式的完整 MP3 保存在项目 `outputs/`；时间轴导出保存在各任务的 `exports/`，从网页下载。不要把应用数据、模型或生成媒体提交到 Git。

## 模式 B：时间轴台词替换

1. 导入 MP4、MOV、M4V、MKV 或 WebM（不超过 2 GB），点击“识别视频台词”。
2. Qwen3-ASR 与 ForcedAligner 生成字词时间戳，按标点和至少 0.5 秒的停顿组成自然句。ASR 标点与词级结果能匹配时，会恢复标点。时间对齐失败时不猜测时间轴，可按视频选区手动新增。
3. 在每句卡片中校对原台词、编辑新台词、调整开始/结束和参考范围。点击“跳转到本句”定位视频；“试听原声”和“试听参考”分别播放句子及参考选区。
4. 默认参考恰好是该句原声。改变参考范围后，先保存，系统仅提取完整落入范围的字词，取消“已确认”。试听后校对参考原话，再勾选确认并保存。切到字词中间的范围需手动修正；系统不会把整个字词文字声称为匹配的半个字词。
5. 勾选需要生成的台词。“选择与视频选区相交的完整台词”会按上方选区选择整句；若只替换句中一部分，先在合适字词边界**拆分**，分别填写两侧原话和新台词。支持相邻片段合并、恢复原始台词、移除片段和单句重生成。
6. 选择时长策略，点击“生成选中片段”。每句显示原始、生成、调整后时长、实际语速倍率、超出状态及中文错误。每句生成结果独立保存；某句失败不会抹掉其他句子的成果。
7. 逐句试听，选择导出方式及 WAV/MP3，点击“导出所选音频”。

### 全局音色与局部语气

- **局部参考**：每句对应的原声音频及准确原话；优先使用，不额外拼入全局音频。
- **全局参考**：可用视频上方选区替换，限 0.5–30 秒，要求清晰、连续、单一说话人，并校对原话。
- 原声少于 0.5 秒或 RMS 低于 0.003 时拒绝局部参考。仅当用户确认全局参考、为该句和全局参考填写**相同且非空的说话人标记**时，才回退到全局参考，并提示局部语气可能丢失。
- 没有自动说话人识别、可靠噪声检测或语音活动检测。不会自动扩展到未知说话人的相邻句，也不会跨说话人拼接。背景音乐本身可能有足够 RMS，请人工选取干净人声。
- Qwen3-TTS Base 在本项目中接收一组音频/文字条件，没有独立的音色参考与情绪参考输入。上述“全局/局部”是**应用层选择与回退策略**，不代表模型将两者完全解耦。局部参考有效时，全局参考不叠加控制音色。

### 时长策略

| 策略 | 行为 |
|---|---|
| 自然生成 | 保留模型自然时长，仍固定放在该句 start 位置 |
| 适配原句 | 目标为 end − start，尝试 FFmpeg atempo 保持音高拉伸 |
| 自定义时长 | 使用用户指定目标秒数，采用相同的拉伸边界 |

Qwen3-TTS Base 当前不发送虚构的语速或目标时长参数。atempo 仅允许 **0.8–1.25 倍**；超出范围保留自然时长并提示调整文案。适配范围内的少量尾部误差通过补静音/采样长度处理。自然生成或适配失败后，若超过原句槽位，仍可单句/顺序导出，但时间轴导出会明确拒绝，防止截断词句、覆盖后句或改变后续位置。

### 导出

| 方式 | 定位与未修改区域 |
|---|---|
| 单句音频 | 只选一句，导出该句调整后音频 |
| 完整新台词 | 选中片段按 start 顺序直接连接，不加入原时间轴空隙 |
| 时间轴：仅新配音 | 按各句 start 放置；其余位置静音，总长度为原视频长度 |
| 时间轴：原声替换 | 未修改区域保留未滤波的原混合音轨；所选句子的完整 start/end 区域用新配音和必要静音替换 |

时间轴导出拒绝所选句子重叠或超出槽位；保留 3–5 秒、8–10 秒这类原有间隙。内部生成/混音使用 24 kHz PCM，原声替换保留混合音轨的立体声，新单声道人声写入两个声道。MP3 编码可能引入解码器延迟，精确视频剪辑优先用 **WAV**。

**原声替换会一起移除修改区内的背景音乐和环境声。** 本版本未安装/实现 Demucs 人声分离，不能宣称保留该区背景音。未来可在导出前接入人声/背景分轨再混音，不改变现有纯本机默认设置。

## 模式 A：自由台词配音

切换为“自由台词配音”，在视频上方选择参考原声并校对参考原话。有时间戳时，自由模式会将新选区对齐到完整字词边界，避免半个字词与整词文字不匹配。输入任意多行新台词，每行一句；空一行增加停顿。点击“生成完整音频”，可逐句试听、编辑后“单独重做”、试听整段并下载 MP3。

保留原有 0.38/1.05 秒句间停顿，仅此模式使用固定停顿。参考现在只用所选短片段及原话；整段转写无需拼入参考。旧项目的自由台词和输出保持可读，时间轴片段从已保存的对齐结果懒迁移，**不会**把旧自由文案硬映射到原视频时间。

## 恢复、缓存与清理

- 保存编辑后关闭/重新打开网页，可以恢复最近项目；未保存的时间轴草稿保存在浏览器本机存储，仅在服务器版本相同时恢复。多个标签页编辑时使用版本号防止覆盖。
- 在任务进行中，服务拒绝编辑、合并、拆分、清理和释放模型等冲突操作。页面关闭不取消后台任务；再次打开可恢复已保存的各句状态。服务重启后中断句会提示重新生成。
- Prompt 缓存仅驻留内存，LRU 最多 **8 项/128 MiB 可计量张量**。键包含任务、句子 ID、模型/SDK版本和本机模型文件签名、源音频路径/大小/纳秒修改与变更时间、参考范围和文字、全局配置及相关选项。修改新台词可复用原参考；更改参考条件使用新键。重启后按需重建，不序列化不安全的模型对象。
- 模型整任务复用，不逐句重载。转写前释放 TTS；也可点“释放模型内存”清除模型和参考缓存。
- “清理当前项目”删除该项目媒体、参考、逐句音频及输出；不删除其他任务和模型。**清理前下载要保留的结果。** 移除片段后保存会清理其生成/参考文件；重新编辑或生成会使旧时间轴导出过期，最多保留最近 12 份导出。

## 架构与扩展

```text
app.py                  FastAPI、本机校验、上传、任务队列、旧自由配音接口
asr/worker.py           独立 ASR/ForcedAligner 环境、时间戳归一化与标点对应
 timeline/models.py     片段与参考数据模型、句子分段、旧数据迁移
 timeline/service.py    编辑/拆分/合并、逐句参考选择、生成和状态保存
 timeline/audio.py      时长适配、磁盘分块混音、WAV/MP3
 timeline/api.py        时间轴 API、版本与任务锁、导出下载
 tts/base.py            TTSBackend 与 VoiceReference 契约
 tts/qwen3_tts.py       真实 Qwen3-TTS Base SDK 调用、MPS/CPU 懒加载
 tts/reference.py       短音频截取、参考签名与有界 Prompt LRU
 tts/registry.py        仅注册已实现的 Qwen 后端
web/index.html          保留原视频预览、选区与自由配音
web/timeline.js         无额外前端框架的时间轴编辑器
```

处理流程：提取 ASR/TTS/未滤波混合音轨 → ASR/ForcedAligner → 句子模型 → 校对/编辑 → 每句参考选择及缓存 → 独立克隆 → 有界时长适配 → 固定时间轴混音/导出。

新后端实现 `load / prepare_reference / synthesize / unload / capabilities / identity` 并注册。`identity` 必须随模型和参考编码行为变化；能力字段要如实声明。如果加入支持独立情绪条件的后端，还需扩展 `VoiceReference` 和参考选择策略来传递两个条件，不能假装 Qwen 已经接受这类输入。未来远程 CUDA 服务需要单独的显式启用和媒体传输授权流程；当前没有远程 URL 配置或自动上传路径。口型同步可作为音频导出之后的独立渲染步骤接入 MuseTalk/LatentSync，当前没有这些功能。

主要接口：

| API | 用途 |
|---|---|
| `GET/PUT /api/jobs/{job}/timeline` | 读取/保存片段、全局参考和 revision |
| `POST …/timeline/split`、`…/merge` | 手动拆分/相邻合并 |
| `POST …/timeline/generate` | 传 `segment_ids`，一项即单句重做 |
| `GET …/timeline/{id}/audio` | 试听本句新声 |
| `POST …/timeline/export` | single/sequence/silence/replace，wav/mp3 |
| `GET /api/tts/capabilities` | 当前后端真实能力 |
| `POST /api/tts/unload` | 空闲时释放模型与缓存 |

## 模型兼容性与限制

| 模型 | 本版本状态 |
|---|---|
| Qwen3-TTS Base（锁定 SDK 0.1.1） | 唯一已接入后端，保留原有 MPS/CPU float32/eager 路径；已测试调度/参考策略，**本次未实跑模型** |
| Qwen3-ASR 0.0.6 / ForcedAligner | 保留原独立 uv 环境与 MPS→CPU、无对齐降级逻辑；测试序列化和子进程交接，**本次未实跑模型** |
| IndexTTS 2.5 | [官方代码](https://github.com/index-tts/index-tts/blob/main/indextts/infer_v2_5.py)确有 MPS 选择路径，`infer` 接受 `spk_audio_prompt`、`emo_audio_prompt`、`duration_factor`；**没有证明能在本项目的 Mac 环境稳定运行，所以未安装、未注册后端** |
| OmniVoice | 预留统一后端接口，未接入、未验证 |

IndexTTS 的独立说话人/情绪参考见[官方使用说明](https://github.com/index-tts/index-tts#readme)。源码存在 MPS 分支不等于所有依赖、算子、内存占用和语音质量已在 Apple Silicon 验证。后续优先在真实 Mac 验证，或对经过授权的局域网 CUDA 服务增加独立可选后端。

更多验收记录和未验证事项见 [docs/verification.md](docs/verification.md)。

## 自动化测试

仅安装轻量测试组，不下载 torch、Qwen 或模型权重：

```bash
uv sync --locked --only-group test
uv run --no-sync python -m pytest -q
node --check web/timeline.js
```

运行真实模型时，重新运行启动器前确保主环境完整；若曾在同一环境执行仅测试组安装，删除 `.venv/.setup-complete` 后启动，或执行 `uv sync --locked --no-install-project` 恢复运行依赖。

测试用合成音频/视频验证参考、缓存、单句生成范围、文件不变性、FFmpeg 拉伸/MP3、时间轴静音、立体声替换、上传与校验、持久化、拆分合并及自由模式。生成测试使用可观察的假 TTS，不等同于真实推理。可选浏览器测试：

```bash
uv pip install playwright
uv run --no-sync python -m playwright install chromium
RUN_BROWSER_TESTS=1 CHROMIUM_EXECUTABLE_PATH=/path/to/chromium uv run --no-sync python -m pytest tests/test_browser.py -q
```

默认跳过浏览器测试；必须在允许 Chromium 进程的环境开启。GitHub Actions 安装 FFmpeg，执行无模型测试和可选浏览器烟测。已知当前环境的 Starlette TestClient 会产生 httpx 兼容性弃用警告，不影响测试结果。

本次本地无模型测试 29 项通过；真实 Chromium 烟测已在 [GitHub Actions](https://github.com/JasirVoriya/local-dialogue-voice/actions/runs/43686802780) 单独执行，1 项通过。真实 Mac 模型推理和真人听感仍待验证。
