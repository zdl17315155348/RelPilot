# RelPilot

RelPilot 是一个中文实体关系联合抽取 demo，实现“关系预判 + BiLSTM-GlobalPointer 联合训练”流程。系统输入中文文本，输出结构化三元组：

```text
主体 / 关系 / 客体
```

## 功能

- 中文文本三元组抽取
- 关系预判结果展示
- 主体、客体高亮
- JSON 结果查看
- GlobalPointer 解码范围对比
- 支持自建数据和 DuIE 数据训练

## 运行 Demo

默认使用自建 demo 模型：

```bash
python3 -m rpgp_demo.app --port 8005
```

访问：

```text
http://127.0.0.1:8005
```

使用 DuIE 训练模型：

```bash
python3 -m rpgp_demo.app \
  --port 8006 \
  --model models/relpilot_joint_duie_dev.best.pt
```

访问：

```text
http://127.0.0.1:8006
```

## 训练自建模型

```bash
python3 scripts/augment_data.py
python3 scripts/train_joint_model.py \
  --dataset data/train_augmented.json \
  --epochs 220 \
  --relations preset
```

输出：

```text
models/relpilot_joint.pt
models/relpilot_joint.best.pt
```

## 训练 DuIE 模型

先将 DuIE 数据转换为项目格式：

```bash
python3 scripts/convert_duie.py \
  --input /path/to/duie_dev.json \
  --output data/duie_dev_samples_5000.json \
  --limit 5000
```

再训练联合抽取模型：

```bash
python3 scripts/train_joint_model.py \
  --dataset data/duie_dev_samples_5000.json \
  --output models/relpilot_joint_duie_dev.pt \
  --best-output models/relpilot_joint_duie_dev.best.pt \
  --limit 5000 \
  --epochs 10 \
  --batch-size 4 \
  --max-length 128 \
  --span-pos-weight 40 \
  --relation-threshold 0.30 \
  --span-threshold 0.30
```

## 评估

```bash
python3 scripts/evaluate.py
python3 scripts/demo_check.py
```

## 主要目录

```text
rpgp_demo/              核心代码
rpgp_demo/static/       前端页面
scripts/                训练、评估、数据转换脚本
data/eval_samples.json  自建评估集
```

`models/`、`logs/`、公开数据下载文件和测试目录默认不提交到 Git。
