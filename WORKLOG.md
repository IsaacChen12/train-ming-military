# 工作日志 / Work Log

本文件记录每次实际动手做的工作（跑了什么、改了什么代码、遇到什么问题、怎么解决的），按时间倒序排列（最新的在最上面）。这和 [RUNBOOK.md](RUNBOOK.md) 里的"调整记录"不同——那里只记模型/参数选型这类决策，这里记录**所有**实际操作过程。

This file records the actual work done in each session (what was run, what code changed, what problems came up and how they were resolved), newest first. This is different from the "Changelog" section in [RUNBOOK.md](RUNBOOK.md), which only records model/parameter selection decisions — this file logs **all** hands-on work.

> 📌 **维护规则 / Maintenance rule**：每次对本项目做了实质性操作（跑通某个阶段、改了脚本、新增素材、修了bug、发现新问题等），都应该在这里补一条记录；新增语料导致 `data/chunks/` 变化时，同时按 [README.md](README.md) 与 [RUNBOOK.md](RUNBOOK.md) §1.5 的规则更新语料统计。
>
> Whenever substantive work is done on this project (getting a stage running, editing a script, adding new material, fixing a bug, finding a new issue, etc.), add an entry here. When new material changes `data/chunks/`, also update the corpus statistics per the rule in [README.md](README.md) and RUNBOOK.md §1.5.

---

## 2026-09-27

- **在代码里标注分块策略待研究 / Marked the chunking strategy as future research in code**：在 `04_chunk_text.py` 顶部模块docstring和 `CHAPTER_RE` 定义处各加了一条TODO注释，指向 RUNBOOK.md §1.4/1.5，提醒 `CHAPTER_RE` 目前未被实际调用、分块策略仍需进一步研究。项目本身新增了 `CLAUDE.md`，把环境说明/文档双语约定/文档同步维护规则从个人记忆迁移到了这里（仓库内、任何session都能读到），避免规则只存在于我的个人记忆里。
  Added TODO comments in `04_chunk_text.py` (module docstring and next to the `CHAPTER_RE` definition) pointing to RUNBOOK.md §1.4/1.5, flagging that `CHAPTER_RE` isn't actually called yet and the chunking strategy still needs further research. Also added `CLAUDE.md` to the project, moving the environment notes / bilingual-docs convention / docs-sync maintenance rule out of personal assistant memory and into the repo itself (readable by any session, not dependent on memory retrieval).
- **文档双语化 / Bilingualized the docs**：[README.md](README.md) 和 [RUNBOOK.md](RUNBOOK.md) 改为中英双语（同一文件内中英段落交替），代码块保持单一版本、注释改为中英对照。
  Converted [README.md](README.md) and [RUNBOOK.md](RUNBOOK.md) to bilingual (Chinese/English paragraphs interleaved within the same file); code blocks stay as a single version with bilingual inline comments.
- **补充分块策略说明 / Documented the chunking strategy**：在 RUNBOOK.md §1.4 里写清楚了 `04_chunk_text.py` 目前的实际切分逻辑，并指出脚本里定义的章节正则 `CHAPTER_RE` 其实没有被调用——目前只是"段落+字数"切分，不具备章节感知能力；标注为后续研究待办。
  Documented the actual splitting logic of `04_chunk_text.py` in RUNBOOK.md §1.4, and flagged that the `CHAPTER_RE` chapter-heading regex defined in the script is never actually called — splitting today is purely "paragraph + character count" with no chapter awareness; marked as future research.
- **语料统计 + 维护规则 / Corpus stats + maintenance rule**：在 README.md 和 RUNBOOK.md §1.5 里加了当前语料的块数/字符数统计表，并定下规则：以后每次往 `data/raw/` 加新素材、重新跑分块后，必须同步更新这两处统计，并在本文件记一条日志。
  Added a table of current chunk/character counts to README.md and RUNBOOK.md §1.5, and set the rule that any future addition of material to `data/raw/` followed by re-chunking must update both statistics sections and get a log entry here.
- **Q&A自动生成（阶段1 Step 5）冒烟测试，已暂停 / Q&A auto-generation (Stage 1 Step 5) smoke test, paused**：装了 `openai` 包，用本机 `qwen3.5:9b`（`http://localhost:11434/v1`）对2个chunk做冒烟测试；用户反馈本机跑 `qwen3.5:9b` 速度太慢不理想，任务已停止（`TaskStop`），`data/qa_pairs/` 尚未生成正式产出，`data/qa_pairs_test/` 为测试残留可清理。计划等局域网大模型主机联通后改用它来跑。
  Installed the `openai` package and ran a 2-chunk smoke test of Q&A generation against the local `qwen3.5:9b` (`http://localhost:11434/v1`). The user flagged that running `qwen3.5:9b` locally is too slow, so the run was stopped (`TaskStop`) before completion; `data/qa_pairs/` has no real output yet, and `data/qa_pairs_test/` is leftover test output that can be cleaned up. Plan is to point this step at the LAN model host once it's reachable.
- **局域网大模型主机连通性排查（第二次），仍不通 / LAN model host connectivity check (second attempt), still unreachable**：`192.168.0.110` 的 `ping`、8080、11434 端口均不通（"Destination host unreachable"）。本机（`192.168.0.103`）与目标同网段，理论应能互通，怀疑对方主机没连这个局域网，或DHCP重新分配了IP。等待确认对方主机当前实际IP。
  Re-tested connectivity to `192.168.0.110` — ping, port 8080, and port 11434 all failed ("Destination host unreachable"). This machine (`192.168.0.103`) is on the same subnet, so it should be reachable in principle; suspect the target host isn't on this LAN right now, or DHCP reassigned its IP. Waiting to confirm the host's current actual IP.

## 2026-09-26

- **项目环境摸底 / Environment survey**：确认本机没装 WSL2、没装 Docker；Ollama 已原生装在 Windows（v0.34.4），本地已有 `qwen3.5:2b/4b/9b`；系统 Python 是 Anaconda 3.14.6（和 `requirements.txt` 里锁定的老版本包不兼容），但另有 `py -3.11` 可用。结论：跑纯RAG（阶段1+2）不需要 WSL/Docker，`chromadb.PersistentClient` 是本地嵌入式库，Ollama本机原生可用。
  Surveyed the local environment: no WSL2, no Docker; Ollama is natively installed on Windows (v0.34.4) with `qwen3.5:2b/4b/9b` already pulled; the system Python is Anaconda 3.14.6 (incompatible with the old pinned versions in `requirements.txt`), but `py -3.11` is also available. Conclusion: running plain RAG (Stage 1+2) needs neither WSL nor Docker — `chromadb.PersistentClient` is a local embedded library, and Ollama already runs natively.
- **局域网大模型主机架构讨论 / Discussed the LAN model-host architecture**：确认史料/素材留在本机，大模型部署在局域网另一台主机（本机性能不足以承担生成）。在 `RAGMing_可能用到的操作与密钥.txt` 中找到该主机地址 `http://192.168.0.110:8080/`（注意：该文件含GitHub/Open-WebUI账号密码与token，未在对话中复述，也未发送到任何外部服务）。首次连通性测试：ping 及 8080/11434 端口均不通，本机（`192.168.0.103`）与目标同网段，判断对方主机当时未联网/未开机，或DHCP改了IP。
  Discussed the architecture where source material stays local but the LLM runs on another LAN host (this machine isn't powerful enough for generation). Found that host's address, `http://192.168.0.110:8080/`, in `RAGMing_可能用到的操作与密钥.txt` (note: that file also contains GitHub/Open-WebUI account passwords and a token, which were not repeated in chat or sent to any external service). First connectivity test: ping and ports 8080/11434 all failed; this machine (`192.168.0.103`) is on the same subnet, suggesting the target host was off/not on the network at the time, or DHCP had changed its IP.
- **确定本轮语料范围 / Scoped this round's source material**：在 `Downloads/` 里发现大量明代军事相关素材（约20篇文字版论文、几个几十~200MB疑似扫描版古籍、13GB的《中国明朝档案总汇》101册扫描档案）。与用户确认后，本轮范围限定为4篇指定的Word文档：《梦到什么写什么——从车营到步火营》《大陆两端的纯步兵详解——从编制到战术》《中欧火绳枪的比较——以现有实验为基础的威力推算》《〈营要事宜〉中的明军阵列复原与同期瑞典军队的对战模拟》。
  Found a large amount of Ming-military-related material in `Downloads/` (~20 text-based papers, several suspected-scanned tomes of tens to ~200MB, and a 13GB 101-volume scanned archive). After confirming scope with the user, this round was limited to 4 specified Word documents: *梦到什么写什么——从车营到步火营*, *大陆两端的纯步兵详解——从编制到战术*, *中欧火绳枪的比较——以现有实验为基础的威力推算*, and *《营要事宜》中的明军阵列复原与同期瑞典军队的对战模拟*.
- **搭建数据管线运行环境 / Set up the data pipeline environment**：新建 `train-ming-military/.venv`（`py -3.11`），装了 `python-docx`、`tqdm`、`loguru`、`pymupdf`；未改动系统 Anaconda 环境。
  Created `train-ming-military/.venv` (via `py -3.11`) and installed `python-docx`, `tqdm`, `loguru`, `pymupdf`; the system Anaconda environment was left untouched.
- **给 `01_extract_pdf.py` 加 `.docx` 支持 / Added `.docx` support to `01_extract_pdf.py`**：原脚本只处理 `.pdf`/`.txt`，新增 `extract_docx()`（用 `python-docx` 按段落提取文字）。
  The script previously only handled `.pdf`/`.txt`; added an `extract_docx()` function (extracts text paragraph-by-paragraph via `python-docx`).
- **修了一个Windows平台的重复处理bug / Fixed a Windows duplicate-processing bug**：`glob("*.docx") + glob("*.DOCX")` 在大小写不敏感的Windows文件系统上会匹配到同一批文件，导致每个文件被处理两次；改成用 `iterdir()` + 后缀判断并去重。
  `glob("*.docx") + glob("*.DOCX")` matched the same files twice on Windows' case-insensitive filesystem, causing every file to be processed twice; fixed by switching to `iterdir()` plus a suffix check with deduplication.
- **跑通数据管线阶段1的提取→清洗→分块 / Ran extract → clean → chunk (Stage 1)**：4个docx → `01_extract_pdf.py` → `03_clean_text.py`（0个文件被跳过）→ `04_chunk_text.py`（`chunk_size=512`, `overlap=64`）→ 共316个chunk，130,302字符，产出到 `data/chunks/all_chunks.jsonl`。抽查内容确认中文无乱码（此前控制台里显示的乱码只是Git Bash终端编码问题，文件本身是UTF-8）。
  Ran the 4 docx files through `01_extract_pdf.py` → `03_clean_text.py` (0 files skipped) → `04_chunk_text.py` (`chunk_size=512`, `overlap=64`), producing 316 chunks totaling 130,302 characters in `data/chunks/all_chunks.jsonl`. Spot-checked the content and confirmed no mojibake (the garbled console output earlier was just a Git Bash terminal-encoding display issue; the files themselves are UTF-8).
