# WMT14 LoRA Fine-Tuning

本项目演示如何使用 Hugging Face `transformers` + `peft` 在 WMT14 (`wmt/wmt14`) 数据集上，以 SFT 形式对大型序列到序列模型执行 LoRA 微调，并使用 TensorBoard 观测 loss 与 BLEU 指标。

## 功能亮点
- ✅ 直接从 Hugging Face Hub 载入 `wmt/wmt14`（默认 `de-en`）数据集
- ✅ 采用 `facebook/mbart-large-50-many-to-many-mmt` 作为基座模型，可自定义语言对
- ✅ 完整的 LoRA 配置（`target_modules`, `r`, `alpha`, `dropout`）
- ✅ `Seq2SeqTrainer` 集成 `sacrebleu` 评估，并在 TensorBoard 中实时记录
- ✅ 提供 `launch_tensorboard.sh` 脚本，一键监控训练/评估曲线

## 环境准备
```bash
# 1. 创建虚拟环境（可选但推荐）
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 3. (可选) 登录 Hugging Face 以便访问受限模型/数据
huggingface-cli login
```

> **硬件建议**：建议使用具备 ≥24GB 显存的 GPU。如果显存不足，可在 `configs/lora_mt.yaml` 中开启 `model.use_4bit=true`（需要 `bitsandbytes` 与支持的显卡）。

## 运行训练
```bash
python src/train_lora_mt.py --config configs/lora_mt.yaml
```

运行后将在：
- `outputs/mbart-wmt14-en-de`：保存 LoRA 适配器与最佳模型
- `runs/mbart-wmt14-en-de`：TensorBoard 日志（loss、学习率、BLEU 等）

若需修改输出目录，可加入 `--output_dir NEW_PATH` 覆盖配置。

## 配置说明 (`configs/lora_mt.yaml`)
- `project`: 随机种子、输出 & 日志目录、缓存目录
- `data`: 数据集名称、语言对、最大长度、少样本调试开关（`max_*_samples`）
- `model`: 基座模型、Tokenizer 语言 code、是否启用 4bit、梯度检查点
- `lora`: LoRA 超参与注入模块
- `training`: `Seq2SeqTrainingArguments` 对齐的训练/评估/保存策略

可根据需求自定义：
```yaml
model:
  name: google/mt5-base
  tokenizer_src_lang_code: en_XX
  tokenizer_tgt_lang_code: de_DE
lora:
  target_modules: ["q", "v"]
```

## 评估与推理
训练过程中会周期性地在验证集上评估 BLEU；结束后脚本还会再次调用 `trainer.evaluate()` 输出最终指标。

若需单独执行推理，可使用保存的 LoRA 适配器：
```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

base = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = PeftModel.from_pretrained(base, "outputs/mbart-wmt14-en-de")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

## 监控 TensorBoard
```bash
chmod +x scripts/launch_tensorboard.sh
./scripts/launch_tensorboard.sh runs/mbart-wmt14-en-de
# 浏览器访问 http://localhost:6006
```

## 常见扩展
1. **调整语言对**：修改 `data.dataset_config`、`data.source_language`、`data.target_language` 以及 tokenizer 语言 code。
2. **更换模型或 LoRA 目标层**：更新 `model.name` 与 `lora.target_modules`；确保模块名称与新模型匹配。
3. **接入 LLaMA-Factory**：本仓库侧重 `transformers` 流程。如希望使用 LLaMA-Factory，可直接复用 `configs/lora_mt.yaml` 中的数据/训练超参思路，在其 `sft/translation` 任务中复写。

祝训练顺利，随时反馈需求以便进一步完善！
