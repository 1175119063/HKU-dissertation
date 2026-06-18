# HKU Dissertation: Mobile Manipulation with LLM Agent vs VLA

RoboCasa 仿真环境中的移动操作任务对比：LLM Agent vs Behavior Cloning vs Diffusion Policy。

## 项目结构

```
thesis_eval/
├── eval_harness.py        # 统一评测框架 (--policy random|scripted|agent|bc|dp)
├── agent_policy.py        # Agent 策略适配层
├── bc_policy.py           # BC 策略适配层
├── dp_policy.py           # DP 策略适配层
├── path_fix.py            # sys.path 修复
├── agent/                 # === LLM Agent 模块 ===
│   ├── agent_loop.py      # Agent 主循环 (场景→LLM→技能→检查)
│   ├── llm_planner.py     # DeepSeek API 接口
│   ├── skills.py          # 技能库 (navigate_to, pick, place_at, open/close)
│   ├── scene_descriptor.py# 场景→文字描述
│   └── traj_skills.py     # 轨迹重放技能 (实验性)
└── bc/                    # === BC / DP 训练模块 ===
    ├── model.py           # BC: ResNet-18 + MLP 模型
    ├── dp_model.py        # DP: 轻量 DDPM (MLP 去噪, ~1.5GB 显存)
    ├── dataset.py         # LeRobot 数据加载器
    ├── train.py           # BC 训练脚本
    └── train_dp.py        # DP 训练脚本
train_all_dp.sh             # 批量训练 8 个 atomic 任务
实验记录.md                 # 完整实验日志 (2026-06-18 ~ 2026-06-19)
问题总结.md                 # Agent 执行层问题总结 (给师兄的)
```

## 环境配置

```bash
# conda 环境
conda create -n robocasa python=3.12
conda activate robocasa

# 依赖
pip install torch gymnasium mujoco lerobot==0.3.3 tianshou opencv-python h5py pyyaml

# RoboCasa + Robosuite
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite && pip install -e . && cd ..
git clone https://github.com/robocasa/robocasa.git
cd robocasa && pip install -e . && cd ..
```

## 运行评测

```bash
# 随机策略 (下界)
./run_eval.sh --task PickPlaceCounterToCabinet --policy random --episodes 10

# LLM Agent (需要 DeepSeek API Key)
export DEEPSEEK_API_KEY="sk-your-key-here"
./run_eval.sh --task PickPlaceCounterToCabinet --policy agent --episodes 3

# Behavior Cloning (需要先训练)
./run_eval.sh --task PickPlaceCounterToCabinet --policy bc --episodes 10

# Diffusion Policy (需要先训练)
./run_eval.sh --task PickPlaceCounterToCabinet --policy dp --episodes 10
```

## 训练模型

```bash
# Behavior Cloning (ResNet-18, ~20 min)
PYTHONPATH="/home/razor/miniconda3/lib/python3.13/site-packages:$PYTHONPATH" \
python thesis_eval/bc/train.py \
  --task PickPlaceCounterToCabinet \
  --epochs 50 --batch-size 16

# Diffusion Policy (轻量 DDPM, ~15 min)
PYTHONPATH="/home/razor/miniconda3/lib/python3.13/site-packages:$PYTHONPATH" \
python thesis_eval/bc/train_dp.py \
  --task PickPlaceCounterToCabinet \
  --epochs 30 --batch-size 8

# 批量训练所有 atomic 任务
bash train_all_dp.sh
```

## 实验结果 (2026-06-19)

| 方法 | 成功率 | 训练 | 说明 |
|------|------|------|------|
| Random | 0% | - | 下界 |
| LLM Agent (DeepSeek) | 0% | 零样本 | 任务分解 100% 正确，OSC 控制器精度不足 |
| BC (ResNet-18, 50 epoch) | 0% | 本地 20min | 累积误差，3000 步超时 |
| DP (轻量 MLP, 30 epoch) | 0% | 本地 15min | MLP 去噪容量不足，192维动作难建模 |
| DP (官方, 服务器) | 待测 | 待定 | 标准基线 |
| VLA (OpenVLA LoRA) | 待定 | 待定 | 最强方案 |
