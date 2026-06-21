# Bug: dist.broadcast 返回 0 导致 num_generated_blocks=0

## 问题描述

4 机 32 卡训练时，node 0 在初始化阶段卡死（加载 VAE 后无响应），其余 24 个进程在 `dist.broadcast` 处等待约 23 分钟后，收到的 `num_generated_blocks` 为 `tensor([0])` 而非 rank 0 应广播的值，导致后续生成 `noise_shape=[1, 0, 16, 60, 104]`（零帧张量）并崩溃。

## 出问题的代码

`model/base.py:170-176`

```python
num_generated_blocks = torch.randint(
    min_num_blocks, max_num_blocks + 1, (1,), device=self.device
)
# 每个 rank 都生成了自己的随机值，但只有 rank 0 的值会被广播
dist.broadcast(num_generated_blocks, src=0)
num_generated_blocks = num_generated_blocks.item()
```

**问题**：rank 0 未到达此处（卡在初始化），broadcast 超时后返回零初始化张量，所有 rank 得到 `num_generated_blocks=0`。

## 多 node 情况

- 4 机 × 8 卡 = 32 进程，使用 `hybrid_full` FSDP sharding
- node 0（rank 0-7）：日志在 `15:15:53` 加载 VAE 后停止，未进入训练循环
- node 1-3（rank 8-31）：在 `15:16:00` 左右到达 broadcast，等待约 23 分钟
- `15:39:15~17`：broadcast 返回，所有非 node 0 的 rank 收到 `tensor([0])`

## 相关日志

**rank 0（node 0）—— 最后一条日志，之后无输出：**
```
[rank:0] [INFO] [2026-04-08 15:15:53,334] : loading wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```
日志文件：`train-node0-rank0-2026-04-08_14-54-37.log`（仅 75 行，2.6KB）

**rank 27（node 1）—— broadcast 前后对比：**
```
[rank:27] [DEBUG] [2026-04-08 15:16:00,543] : DMD._run_generator: num_generated_blocks=tensor([7], device='cuda:3')
[rank:27] [DEBUG] [2026-04-08 15:39:17,599] : DMD._run_generator: Broadcasted num_generated_blocks=tensor([0], device='cuda:3')
[rank:27] [DEBUG] [2026-04-08 15:39:17,600] : DMD._run_generator: num_generated_blocks=0
[rank:27] [DEBUG] [2026-04-08 15:39:17,600] : DMD._run_generator: num_generated_frames=0
[rank:27] [DEBUG] [2026-04-08 15:39:17,600] : DMD._run_generator: noise_shape=[1, 0, 16, 60, 104]
```
日志文件：`train-node1-rank27-2026-04-08_14-54-37.log`

**rank 10（node 2）—— 同样现象：**
```
[rank:10] [DEBUG] [2026-04-08 15:09:14,847] : DMD._run_generator: num_generated_blocks=tensor([7], device='cuda:2')
[rank:10] [DEBUG] [2026-04-08 15:39:15,994] : DMD._run_generator: Broadcasted num_generated_blocks=tensor([0], device='cuda:2')
[rank:10] [DEBUG] [2026-04-08 15:39:15,995] : DMD._run_generator: num_generated_blocks=0
```
日志文件：`train-node2-rank10-2026-04-08_14-54-37.log`

## 根本原因

node 0 在 FSDP 初始化或 VAE 加载阶段卡死（具体原因待查：OOM、硬件故障、网络问题），导致 rank 0 从未参与 broadcast，其他 rank 超时后收到零值。

## 日志目录

`logs/2026-04-08_14-03-14-teacher-WAN2.1-T2V-1.3B-self_forcing_dmd/`
- 训练日志：`train-node{N}-rank{R}-2026-04-08_14-54-37.log`
- 集群日志：`fuyao_logs/rank{0-3}.log`

---

# trainer/diffusion.py 与 trainer/ode.py 的区别

- **ODE Trainer** (`trainer/ode.py`)：用 teacher 的 ODE 轨迹做监督回归，初始化 causal student。
- **Diffusion Trainer** (`trainer/diffusion.py`)：用 flow-matching loss 进一步训练，使模型能在少步数（3-4步）生成高质量视频。

## ODE Trainer

- 数据：完整 ODE 轨迹 LMDB，shape `[B, T_steps, F, C, H, W]`
- Loss：MSE(pred_x0, teacher_x0)，`t=0` 的帧 mask 掉
- Timestep：离散列表（如 `[1000, 757, 522, 0]`），每 block 均匀采样
- 只训练 generator

## Diffusion Trainer

- 数据：只取 clean latent（`ode_latent[:, -1]`），shape `[B, F, C, H, W]`
- Loss：标准 flow-matching MSE，加 timestep 权重：
  ```
  loss = MSE(flow_pred, training_target) * training_weight(t)
  ```
- Timestep：连续范围 `[0.02T, 0.98T]`
- **没有对抗蒸馏**（无 real_score / fake_score / KL divergence）
- `model/diffusion.py` 的 `generator_loss` 是纯 flow-matching，不含 DMD 逻辑
- 额外特性：`teacher_forcing`（传入 clean context latent）、`noise_augmentation_max_timestep`（对 context 加小噪声）、EMA

## 关键结论

`trainer/diffusion.py` 本身是训练循环的壳，核心 loss 在 `model/diffusion.py` 的 `CausalDiffusion.generator_loss`，该方法**不含蒸馏**，是 flow-matching 训练。

---

# Self Forcing DMD 训练流程

## 总览

DMD = Distribution Matching Distillation。把多步 teacher 蒸馏成少步 causal student。Self Forcing 的关键点是：student 在训练时**自回归 rollout**（含 KV cache，跟推理时一致），然后只对其中**一个 block / 一个去噪步**回传梯度，其余用 `torch.no_grad()` 跑。这样训练和推理的输入分布一致，避免 train-test gap。

数据流：

```
trainer/distillation.py::Trainer.train()
    └─ fwdbwd_one_step(batch, train_generator)
        ├─ train_generator=True  -> model.generator_loss(...)
        └─ train_generator=False -> model.critic_loss(...)

model/dmd.py::DMD.generator_loss(...)
    ├─ self._run_generator(...)               # SelfForcingModel
    │     └─ self._consistency_backward_simulation(...)
    │           └─ SelfForcingTrainingPipeline.inference_with_trajectory(...)
    │                 # 自回归生成 num_blocks 个 block 的 x0 latent
    │                 # 每个 block 内只在 exit_flag 处带梯度
    └─ self.compute_distribution_matching_loss(pred_image, ...)
          # real/fake score 在 pred_image 上算 DMD 梯度
```

## 1. trainer/distillation.py

`Trainer` 是训练循环外壳：

- **每步选择性训练 generator**：`TRAIN_GENERATOR = step % dfake_gen_update_ratio == 0`（配置 `dfake_gen_update_ratio: 5`，即每 5 步更新一次 generator，其余步只更新 critic / fake_score）。
- **fwdbwd_one_step**：
  - 取一个 batch（仅 text prompts，T2V 模式 `clean_latent=None`、`image_latent=None`）。
  - 文本编码 -> `conditional_dict`；缓存固定的 negative prompt -> `unconditional_dict`。
  - `train_generator=True` 调 `self.model.generator_loss(...)`，`backward()`，clip grad。
  - `train_generator=False` 调 `self.model.critic_loss(...)`。
- 维护 `generator_ema`（DMD 权重 0.99，从第 200 步开始）。

**注意**：训练是 data-free 的，不需要任何视频数据，所有 fake sample 都是 student 自己 rollout 出来的。

## 2. model/dmd.py::DMD

继承自 `SelfForcingModel`。核心方法：

### generator_loss

```
1. (pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to)
       = self._run_generator(...)               # 自回归生成 student 的 x0 输出
2. dmd_loss = self.compute_distribution_matching_loss(
       image_or_video=pred_image, ...)
```

### compute_distribution_matching_loss（标准 DMD2）

1. 在 `pred_image`（student 的 x0 估计）上**重新加噪声**到随机时间步 `t`。
2. `_compute_kl_grad`：
   - `fake_score(noisy, t)` 预测 fake 分布的 x0
   - `real_score(noisy, t)` 预测 real 分布的 x0（带 CFG，guidance_scale=3）
   - `grad = pred_fake_image - pred_real_image`（DMD 论文 eq.7 KL 梯度方向）
   - 用 `|x0 - pred_real|` 的均值做归一化（eq.8）
3. `dmd_loss = 0.5 * MSE(pred_image, (pred_image - grad).detach())`，触发反传时等价于把 `grad` 注入到 `pred_image` 上。

`gradient_mask`：当 rollout 的总帧数 > `min_num_frames` 时，**第一个 block 不参与 DMD loss**（因为它带 image latent / 重新编码过的帧，监督会有 bias）。

### critic_loss

只训 fake_score：让它学会去噪 student 自己生成的样本。

1. `with torch.no_grad(): generated = self._run_generator(...)`
2. 给 `generated` 加噪到随机 `t`。
3. `pred_fake_image = fake_score(noisy, t)`。
4. flow-matching loss（`denoising_loss_type: flow`）。

## 3. model/base.py::SelfForcingModel._run_generator

负责**调用 pipeline 做自回归 rollout**，并决定生成多少 block：

1. 训练时随机采样 `num_generated_blocks ∈ [min_num_blocks, max_num_blocks]`，**用 `dist.broadcast(src=0)` 全 rank 同步**（保证所有 DDP rank 生成相同帧数，否则 FSDP all-gather 会形状不一致）。
2. `num_generated_frames = num_generated_blocks * num_frame_per_block`（配置里 `num_frame_per_block=3`，`num_training_frames=21`，所以 blocks ∈ [7, 7]，恒等于 7 个 block × 3 帧 = 21 帧；日志里也始终 `num_blocks=7`）。
3. 调 `_consistency_backward_simulation(noise, **conditional_dict)`，进入 `SelfForcingTrainingPipeline.inference_with_trajectory`。
4. **取最后 21 帧**作为 DMD 监督对象 `pred_image_or_video_last_21`（如果总帧数 > 21，需要把第 1 帧重新通过 VAE encode/decode 一次得到 image latent，再拼上后 20 帧；T2V 配置下总帧数 == 21，这条分支不触发）。
5. `gradient_mask`：
   - 若 `num_generated_frames == min_num_frames`（即恰好 21）-> `gradient_mask = None`，整段 21 帧都参与 DMD loss。
   - 否则第一个 chunk 不参与（因为它含 image latent）。

## 4. pipeline/self_forcing_training.py::inference_with_trajectory

这是 self forcing 的核心。流程：

```
初始化 KV cache（按最大 num_max_frames=21）
for block_index in range(num_blocks):                # 时间维度
    noisy_input = noise[block range]
    for index, t in enumerate(denoising_step_list):  # 空间/去噪维度
        if index == exit_flag_for_this_block:
            带梯度跑 generator(noisy_input, t)
            break
        else:
            with torch.no_grad():
                denoised_pred = generator(noisy_input, t)
                noisy_input = scheduler.add_noise(denoised_pred, t_next)  # consistency 采样
    output[block range] = denoised_pred
    # context refill：把这个 block 的 K/V 写入 cache，让后续 block 能看到它
    # 先对 denoised_pred 加 context_timestep 级别的噪声（配置 context_noise=0，相当于不加噪）
    # 再用 timestep=context_noise(=0) 跑一遍 generator，K/V 落到 kv_cache1
    with torch.no_grad():
        denoised_pred = scheduler.add_noise(denoised_pred, noise, t=context_timestep)  # 默认 0，no-op
        generator(denoised_pred, timestep=context_timestep, kv_cache=...)
```

### exit flag 是怎么起作用的

`generate_and_sync_list(num_blocks, num_denoising_steps)`：
- 在 rank 0 上随机生成长度 `num_blocks`、范围 `[0, num_denoising_steps)` 的整数列表。
- `dist.broadcast(src=0)` 同步给所有 rank（保证 DDP 一致）。
- 配置 `denoising_step_list: [1000, 750, 500, 250]` -> `num_denoising_steps=4`，所以每个 exit_flag ∈ {0,1,2,3}。

**作用**：在每个 block 内的去噪循环中，只有当 `index == exit_flag` 那一步才**带梯度**跑 generator 并 `break` 退出循环；其他步全部 `with torch.no_grad()` 跑（包括 KV cache 更新那次 `context_timestep` 的回填）。

`same_step_across_blocks`（默认 True）：所有 block 共用 `exit_flags[0]`。否则每个 block 用自己的 exit_flag。

日志这一次 `exit_flags=[0,1,1,3,0,2,2]`，因为 `same_step_across_blocks=True` 所以只看 `exit_flags[0]=0`，**这一步**所有 block 都在 `index=0`（t=1000）退出，`denoised_timestep_from=1000, denoised_timestep_to=750`。

注意这只是一次随机采样的结果。`exit_flags[0]` 每次训练 step 都重新随机采样，可能取 `{0,1,2,3}` 中的任意一个：

| `exit_flags[0]` | 退出时的 `current_timestep` | `denoised_timestep_from` | `denoised_timestep_to` | 含义 |
|---|---|---|---|---|
| 0 | 1000 | 1000 | 750 | 第一步就 break，相当于 student 用最纯噪声做 1 步 x0 估计 |
| 1 | 750  | 750  | 500 | 先 no_grad 跑 t=1000 -> 加噪到 t=750 -> 带梯度跑 t=750 break |
| 2 | 500  | 500  | 250 | 先 no_grad 跑 t=1000、750 两步 -> 带梯度跑 t=500 break |
| 3 | 250  | 250  | 0   | 跑完 1000、750、500 三步 no_grad -> 带梯度跑最后一步 t=250 break（此时是最后一个 index，进入 `exit_flags[0] == len-1` 分支，`to=0`） |

也就是说带梯度的那一步 forward 看到的 noisy_input 不一定是纯噪声，而是已经 no_grad 去噪了 `exit_flags[0]` 步、再加噪到下一时间步的中间状态。这样 student 训练时见到的输入分布就和真实推理时第 `exit_flags[0]` 步的输入分布一致——这是 self forcing 解决 train-test gap 的核心。

### context refill 前的 add_noise

代码里 context refill 这一步实际上有两个动作：

```python
denoised_pred = scheduler.add_noise(denoised_pred, randn, context_timestep)
generator(denoised_pred, timestep=context_timestep, kv_cache=...)
```

配置里 `context_noise: 0`（默认值），所以 `context_timestep` 全是 0，`scheduler.add_noise` 在 `t=0` 时输出就是原 `denoised_pred`，等于**没加噪**。

这个 `add_noise` 看起来冗余，但留作开关：把 `context_noise` 设成非 0 的小值，就能让 KV cache 看到的 context 是带轻微噪声的版本，作为正则化（在某些 self forcing diffusion 变体里会用，但本次 DMD 配置不启用）。

`start_gradient_frame_index = num_output_frames - 21`：只有 `current_start_frame >= start_gradient_frame_index` 的 block 才真的开梯度，前面的 block 即使到达 exit_flag 也用 `torch.no_grad()`。这个机制是为支持「rollout > 21 帧、只对最后 21 帧计算 DMD」的长序列场景。T2V 配置下 `num_output_frames=21`，所以 `start_gradient_frame_index=0`，**所有 block 都开梯度**（与日志一致：每个 block 都打印 "running with gradients enabled"）。

### 直观图（`same_step_across_blocks=True`，假设 exit_flag=1）

```
block 0:
  index=0 (t=1000): no_grad     pred = G(noise, t=1000)
                                noisy = add_noise(pred, t=750)
  index=1 (t=750):  GRAD         pred = G(noisy, t=750)   <- backprop 这一步
                                  break
  context refill (t=0): no_grad  G(pred, t=0) -> KV cache

block 1:  (KV cache 里已有 block 0 的 K/V)
  ... 同上
```

注意：尽管 student 名义上是 4 步去噪（1000→750→500→250→x0），但训练时实际只跑到 exit_flag 步就停。退出时返回的 `denoised_pred` 是**那一步的 x0 估计**（不是真正最终去噪到 t=0 的 x0），这正是 DMD2 randomized truncation 思想。

### 用于 DMD supervision 的 latent 是哪一段？

`output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred`

也就是**每个 block 在 exit_flag 处的 x0 估计**被写入 `output`。然后：

- `_run_generator` 取 `pred_image_or_video_last_21 = output` 的最后 21 帧。
- T2V 21 帧场景下 `output.shape = [B, 21, C, H, W]`，整段都是「每个 block 在 exit_flag 处 x0 估计」拼起来的 latent。
- 这个 tensor 直接交给 `compute_distribution_matching_loss` 作为 student 的输出 latent，用于和 real_score 计算 DMD 梯度。

**梯度路径**：DMD loss → `pred_image` → 带梯度的那一步 generator forward（每个 block 一次）。`current_start_frame >= start_gradient_frame_index` 控制哪些 block 真正参与反传；KV cache 写入那一遍永远 `no_grad`，所以梯度不会通过 cache 跨 block 传播。

### exit_flag 同步的作用

1. **DDP 一致性**：所有 rank 的 generator 在同一个 (block_index, denoising_index) 处带梯度，FSDP all-gather 才能成功。
2. **DMD timestep 范围一致**：`denoised_timestep_from / denoised_timestep_to` 由 `exit_flags[0]` 决定，控制 `compute_distribution_matching_loss` 里 DMD 时间步采样的上下界（当 `ts_schedule=True`）。配置里 `ts_schedule: false` -> 此功能关闭，DMD 时间步在 `[min_score_timestep, num_train_timestep]` 内自由采样。

## 配置回顾（self_forcing_dmd.yaml）

| 参数 | 值 | 含义 |
|---|---|---|
| `denoising_step_list` | `[1000, 750, 500, 250]` | student 4 步去噪 |
| `num_frame_per_block` | 3 | 每个 block 3 个 latent 帧 |
| `image_or_video_shape` | `[1, 21, 16, 60, 104]` | 21 帧 latent |
| `num_train_timestep` | 1000 | DMD 加噪时间步范围 |
| `timestep_shift` | 5.0 | flow-matching 时间步重映射 |
| `guidance_scale` | 3.0 | real_score 的 CFG |
| `dfake_gen_update_ratio` | 5 | 每 5 步更新一次 generator |
| `ts_schedule` | false | DMD 不限制时间步范围（不依赖 exit_flag 输出的 from/to） |
| `same_step_across_blocks` | true（默认） | 所有 block 共用 exit_flag[0] |
| `ema_weight` / `ema_start_step` | 0.99 / 200 | generator EMA |
| `lr` / `lr_critic` | 2e-6 / 4e-7 | generator 比 critic 学得稍快 |

## EMA

DMD 训练**只对 generator 维护 EMA**，critic / fake_score 没有 EMA。EMA 是旁路维护的「指数滑动平均副本」，不参与训练 forward/backward，只在保存 checkpoint 和推理时使用。

代码在 `trainer/distillation.py`：

**1. 初始化**（构造时）：

```python
ema_weight = config.ema_weight                          # 0.99
self.generator_ema = None
if (ema_weight is not None) and (ema_weight > 0.0):
    self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
```

**2. 延迟启动**：

```python
if self.step < config.ema_start_step:                   # 200
    self.generator_ema = None
```

前 200 步把 EMA 副本丢掉，省显存和算力。原因：开头 generator 还在剧烈变化，平均没意义。

**3. 每次 generator 更新后 update EMA**（在训练循环里）：

```python
self.generator_optimizer.step()
if self.generator_ema is not None:
    self.generator_ema.update(self.model.generator)
```

注意只在 `TRAIN_GENERATOR=True` 的步数（每 5 步一次，因为 `dfake_gen_update_ratio=5`）才会更新 EMA——critic-only 的步数 EMA 保持不变。

**4. 到达 `ema_start_step` 时按需创建**：

```python
if (self.step >= self.config.ema_start_step) and (self.generator_ema is None) and (self.config.ema_weight > 0):
    self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
```

**5. 保存 checkpoint**：达到 `ema_start_step` 之后 state_dict 多一项 `generator_ema`：

```python
if self.config.ema_start_step < self.step:
    state_dict = {"generator": ..., "critic": ..., "generator_ema": self.generator_ema.state_dict()}
else:
    state_dict = {"generator": ..., "critic": ...}
```

## EMA 关键点

- **训练 forward 用的是实时 generator**（`self.model.generator`），不是 EMA 副本。`generator_loss` / `critic_loss` 里 `_run_generator` 调用的都是当前权重。EMA 完全不参与训练计算图。
- **推理用 EMA**：README 的 inference 命令带 `--use_ema`，加载 ckpt 中的 `generator_ema` 出视频。这是发布的最终权重。
- **`EMA_FSDP`** 在 `utils/distributed.py`，专门处理 FSDP 分片参数（每个 rank 只维护自己 shard 的 EMA，all-gather 时再合并）。
- 配置：`ema_weight: 0.99`、`ema_start_step: 200`，总训练 600 步——后 400 步维护 EMA，按 `dfake_gen_update_ratio=5` 算其实只 update 了约 80 次。decay=0.99 在 80 次更新下半衰期约 69 步，足够把后期权重平滑下来。

## 一句话总结

Self Forcing DMD = **student 自己用 KV cache 自回归跑出 21 帧 latent**（其中只有一个全 rank 同步的 (block, denoising_step) 步带梯度），把这 21 帧整体作为「fake 样本」喂给 DMD 的 real/fake score 算 KL 梯度回传。`exit_flag` 决定回传发生在每个 block 的哪个去噪步上，是把多步 rollout 的指数级显存炸裂降到 1 步反传的关键开关。

---

# SiD（Score identity Distillation）— 与 DMD 的差异

`model/sid.py` 的 `SiD` 类与 `DMD` 共用：`SelfForcingModel._run_generator`、self forcing rollout pipeline、`exit_flag` 机制、`critic_loss`（几乎逐字相同）。**只有 `compute_distribution_matching_loss` 的损失公式不同**。下面只列差异点。

## 差异 1：loss 公式（核心）

DMD 的做法是**算 KL 梯度然后用 MSE 注入**：

```python
with torch.no_grad():
    grad = pred_fake - pred_real
    grad = grad / |x0 - pred_real|.mean()      # 归一化
loss = 0.5 * MSE(x0, (x0 - grad).detach())     # 反传等价于把 grad 注入到 x0
```

SiD 的做法是**直接构造一个标量损失，让 pred_fake 自己进入计算图**：

```python
sid_loss = (pred_real - pred_fake) * (
    (pred_real - x0) - sid_alpha * (pred_real - pred_fake)
)
sid_loss = sid_loss / |x0 - pred_real|.mean()  # 同款归一化
sid_loss = sid_loss.mean()
```

把这个表达式按对 `x0`（student 输出）的梯度展开就能看出和 DMD 的等价/差异。SiD 论文（Score identity Distillation, https://arxiv.org/abs/2404.04057）的 idea 是用 score identity 把 reverse-KL 重写成可以直接做反传的形式，**不需要先 detach 梯度再用 MSE 注入**。`sid_alpha` 是 SiD 论文中的混合系数（默认 1.0）。

## 差异 2：score 前向**没有 `torch.no_grad()` 包裹**

DMD 里 `_compute_kl_grad` 全程在 `with torch.no_grad():` 下跑（见 `compute_distribution_matching_loss` Step 2）；梯度只通过最后那个 `MSE` 经由 `x0` 回到 generator。

SiD 的 fake_score / real_score forward **没有 `no_grad`**——`pred_fake_image` 直接进 loss 表达式，autograd 会构建完整计算图并反传：

- 对 `pred_fake` 的梯度 → 经 fake_score → 反传到 `noisy_image_or_video` → 反传到 `x0`（generator 输出）。
- 对 `pred_real` 的梯度同理（real_score 因 `requires_grad_(False)` 不会更新参数，但仍有激活值需要保存）。

这意味着 **SiD 的 generator 反传会经过 fake_score 和 real_score 两个网络的反传**——比 DMD 显存和算力开销大不少。这也是为什么 SiD 把 `gradient_checkpointing` 也开到 `real_score`（DMD 只对 generator 和 fake_score 开）：

```python
# sid.py
if args.gradient_checkpointing:
    self.generator.enable_gradient_checkpointing()
    self.fake_score.enable_gradient_checkpointing()
    self.real_score.enable_gradient_checkpointing()   # 比 DMD 多这一行
```

## 差异 3：没有 fake CFG

DMD 支持 `fake_guidance_scale`（对 fake_score 也做 CFG）；SiD 只对 real_score 做 CFG（`real_guidance_scale`），fake 部分单次前向。

## 差异 4：`gradient_mask` 没用上

`compute_distribution_matching_loss` 函数签名里有 `gradient_mask` 参数但**实际未使用**——SiD 直接 `sid_loss.mean()`，没有按 mask 过滤。如果在长 rollout（>21 帧）场景下使用，第一个 chunk 的 image latent 也会进 loss，可能引入 bias。代码里两个 `# TODO: Add alpha`、`# TODO: Double?` 注释说明这部分没写完。

## 差异 5：缺 `min_score_timestep`

SiD `__init__` 里没有 `self.min_score_timestep`。当 `ts_schedule=False` 且 `denoised_timestep_to is None` 时会走到 `else self.min_score_timestep` 分支，**直接 AttributeError**。所以 SiD 必须配 `ts_schedule: true`（或者 self forcing rollout 里 `same_step_across_blocks=True` 让 `denoised_timestep_to` 不为 None）才能跑。这是个潜在 bug / 未完成项。

## 一句话总结

SiD vs DMD：rollout、exit_flag、critic 训练**完全一样**，只换了 generator 损失公式——SiD 把 score identity 写成可微表达式直接反传（不 detach），代价是 generator 反传必须经过 real/fake score 两个网络，显存更大；DMD 在 `no_grad` 下算 grad 再用 MSE 注入，更省显存。两者都是 self forcing 框架下的 distribution matching distillation 变体。

---

# Gradient Checkpointing 在 backward 时重跑 forward

`gradient_checkpointing: true` 时，每个 transformer block 用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 包起来。forward 阶段不存激活；backward 阶段 autograd 对每个 checkpoint **再 forward 一次**重算激活，再回传梯度。

日志现象：`_run_generator` 返回（forward 走完 0→3→6→…→18）之后，紧接着出现一段 `current_start_frame = 18 → 15 → … → 0` 的反向序列，每个 block 内 ~30 行 self-attention forward（= 模型层数）。这就是 backward 阶段 7 个带梯度 block × 30 层的重算，按 block 倒序是因为 autograd 从 loss 端反推。

**注意**：`CausalWanModel` 里给非首 block 套的 `logging.disable(logging.CRITICAL)` 包裹只在原 forward 的 for-loop 里生效；recompute 不走那层 for-loop，所以 backward 时所有 block 都会打日志。要静音可以在 `backward()` 外面再套一层 `logging.disable`（见 `trainer/diffusion.py`），或给 checkpoint 调用传 `context_fn=` 提供 recompute 专用的 context manager（PyTorch 2.1+）。



# CUDA Memory 占用

diffusion trainer 训练: 8 卡训练时能跑， 接近满显存， 但中途 inference 时 CUDA Out of Memory

16 卡训练能跑， 接近满显存， 中途某一次 CUDA Out of Memory

## 2 机 16 卡 OOM 定位（2026-05-03）

### 显存现场

H800 80 GB，已用 78.52 GB，PyTorch 实际占 62.67 GB，**reserved-but-unallocated 10.75 GB**，只剩 679 MiB，连 678 MiB 的新 sharded grad buffer 都装不下。

### 可能原因（按可能性排序）

1. **碎片化是直接元凶**：空闲 679 MiB，但 PyTorch reserve 着不用的就有 10.75 GiB——反复的 KV cache、checkpoint recompute 反向时的临时激活反复 alloc/free 把 caching allocator 撑开。报错也建议 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
2. **`hybrid_full` + 2 机 16 卡并不省每卡显存**：`hybrid_full` 只在节点内（8 卡）分片，跨节点是复制。8 卡 → 16 卡每卡 shard 大小不变，反而多了一次跨节点 all-reduce 的临时 buffer。
3. **`generator_fsdp_wrap_strategy: size`** 切出来的 FSDP unit 比较粗。`empty_like(chunks[0])` 要 678 MiB ≈ 1.3B × 4 byte / 8 ≈ 650 MiB，对应一个粗 unit 的 shard。改成 transformer-block 粒度可降峰值。
4. **gradient checkpointing backward 重算**：每个带梯度的 block 在 backward 里把 30 层 attention 重新前向一遍，重算激活 + 该 block grad + KV cache 同时驻留。21 帧 × 7 个 block 全部带梯度（`num_output_frames=21`，`start_gradient_frame_index=0`）。
5. **inference_interval=50** 跑完 inference 没释放干净，下一次 train step 显存基线变高。`run_inference` 内部 pipeline 的 KV cache 可能没释放。

### 建议尝试顺序（收益/成本排序）

- 加环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `inference_interval` 调大或设 `-1` 禁用先跑通
- `generator_fsdp_wrap_strategy: size` → transformer-block 粒度
- `batch_size: 2` → `1`，配合 `total_batch_size` 累积
- 关 `mixed_precision` 排查 fp32 master grad

## 加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

2 机 16 卡训练还是 CUDA Out of Memory

```
[env check] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

环境变量设置成功， 看来还是不能 batch size > 1 .

## 改动：KV cache 懒加载 + 原地 reset

`pipeline/self_forcing_training.py`：

1. `inference_with_trajectory` 中 KV / crossattn cache 改为懒加载：第一次分配，后续步只 reset，不重新 alloc。原来的实现每步 `torch.zeros` 申请约 12 GB 的 K/V buffer 然后释放，是 allocator 碎片的主要来源。
2. 新增 `_reset_kv_cache` / `_reset_crossattn_cache`：
   - 原地 `tensor.zero_()` 把 K/V buffer 清零（行为和首次 `torch.zeros` 一致）
   - `detach_()` + 清 `.grad` + `requires_grad_(False)` 防御性地保证 buffer 不会被卷进 autograd 图
   - `kv_cache1` 重置 `global_end_index` / `local_end_index` 为 0
   - `crossattn_cache` 重置 `is_init=False`，让模型下一步根据新 prompt 重新填充

预期效果：减少每步 12 GB 的 alloc/free churn，缓解碎片。

结果： 还是 CUDA Out of Memory， 并且在 inference 的时候 CUDA Out of Memory。 说明 inference 的时候加载的 `CausalInferencePipeline` 非常吃显存。

## KV cache 复用时的 leaf / grad_fn 陷阱（也是原作者注释掉懒加载的根因）

### 现象

跑起来第 2 步开头，`_reset_kv_cache` 调 `tensor.requires_grad_(False)` 直接报：

```
RuntimeError: you can only change requires_grad flags of leaf variables.
If you want to use a computed variable in a subgraph that doesn't require
differentiation use var_no_grad = var.detach().
```

第 1 步训练成功，问题只在复用 buffer 时出现。每步都重新 `torch.zeros` 分配（即原始实现）就不会触发。

### 根因

模型 attention 写入 KV cache 是 in-place index 赋值（`wan/modules/causal_model.py:241-242`）：

```python
kv_cache["k"][:, local_start_index:local_end_index] = roped_key
kv_cache["v"][:, local_start_index:local_end_index] = v
```

当 self forcing 进入"带梯度"那一步，`roped_key` / `v` 来自 generator 的带梯度前向，`requires_grad=True`。这种 in-place 写入会被 autograd 当作 `IndexPut` (`CopySlices`) op 记录：

- 把作为目标的 leaf 张量 **染成 non-leaf**：`is_leaf=False`
- 同时把 `requires_grad` **隐式翻成 True**
- 给 leaf 挂上 `grad_fn=<CopySlices>`

backward 跑完之后，autograd 引擎释放了图里 saved-for-backward 的中间张量（释放显存），但 **不会清 tensor 上的 `grad_fn` 引用**——那个 `CopySlices` Python 对象仍然挂在 cache 张量上，地址在两次 backward 之间保持不变。这是 PyTorch 的设计（为了支持 `retain_graph=True`）。

下一步训练进 `_reset_kv_cache` 时，cache 张量还是 `is_leaf=False, requires_grad=True, grad_fn=<CopySlices>`，所以 `requires_grad_(False)` 直接报 leaf-only 的错。

原作者每步重新 `torch.zeros` 看似"浪费"，其实就是用每步丢掉旧张量来回避这个问题：旧张量带着残留的 `grad_fn` 一起被 GC 掉，新分配的张量是干净 leaf。代价是每步 12 GB 的 alloc/free，反过来撑碎片。

### 修法

`_reset_kv_cache` / `_reset_crossattn_cache` 第一行改成 `tensor.detach_()`（in-place 版本）：

- 清掉 `grad_fn`
- `requires_grad` 复位 False
- `is_leaf` 复位 True

之后再调 `requires_grad_(False)`、清 `.grad`、`zero_()` 都合法，buffer 也得以复用。

## inference 时显存压顶 + gc.collect 配对（实测有效）

### 改动

1. **训练 pipeline 在 inference 前释放**（`trainer/diffusion.py` train 循环）：
   ```python
   self.model.inference_pipeline = None
   gc.collect()
   torch.cuda.empty_cache()
   self.run_inference(...)
   gc.collect()
   torch.cuda.empty_cache()
   ```
   理由：`run_inference` 在 rank 0 上构造 `CausalInferencePipeline`（自带 ~12 GB KV cache）。如果训练 pipeline 的 12 GB KV cache 还在，rank 0 上两份 KV cache 同时占用 → OOM（note 早期记录的"中途 inference 时 OOM"就是这个）。下一步训练靠懒加载重建训练 pipeline。

2. **`CausalInferencePipeline` 加 `vae_offload_cpu` 选项**（仿 `stream_inference.py`）：
   - 平时 VAE 在 CPU
   - decode 前 `gc.collect + empty_cache`，再 `vae.to(device)`
   - decode 后立刻 `vae.to("cpu") + gc.collect + empty_cache`
   - `default_config.yaml` 默认 `vae_offload_cpu: true`

3. **`gc.collect()` 与 `torch.cuda.empty_cache()` 全代码库配对**：
   - `gc.collect()` 只释放 Python 对象，CUDA caching allocator 仍持有那些块；不配 `empty_cache` 等于白干
   - `trainer/base.py::maybe_run_gc`、`trainer/base.py::run_inference` 末尾、所有 trainer（diffusion / distillation / gan / ode）的 `step % 20 == 0` 处、save 前后均成对
   - 唯一保留单独 `empty_cache` 的位置：`trainer/gan.py:460` 每步末尾的兜底（每步 `gc.collect` 太贵）

### 实测效果

加完这一组改动后，2 机 16 卡 diffusion trainer：
- inference 时显存明显下降（VAE 不在 GPU 占地），训练 KV cache 也已释放，inference pipeline 有完整 headroom
- 不再出现 OOM


# Random seed 相关改动

目的：让两次 run 的随机数消费路径尽量一致，方便排查 loss 分叉。

做法：

- [trainer/base.py](trainer/base.py)：拆成 `model_init_seed = config.seed` 和 `training_seed = config.seed + global_rank`。
- [trainer/base.py](trainer/base.py)：初始化模型前用 `model_init_seed`，保证各 rank 初始权重一致。
- [trainer/base.py](trainer/base.py)：训练阶段调用 `set_training_seed()`，让每个 rank 使用自己的 `training_seed`。
- [trainer/base.py](trainer/base.py)：统一封装 `build_distributed_dataloader()`，给 `DistributedSampler` 固定 `seed=model_init_seed`。
- [trainer/base.py](trainer/base.py)：给 DataLoader 设置 `torch.Generator().manual_seed(training_seed)`。
- [trainer/base.py](trainer/base.py)：给 worker 设置 `random / numpy / torch` seed，值为 `training_seed + worker_id`。
- [utils/dataset.py](utils/dataset.py)：新增 `cycle_with_sampler_epoch()`，每轮调用 `sampler.set_epoch(epoch)`。
- [trainer/diffusion.py](trainer/diffusion.py)、[trainer/distillation.py](trainer/distillation.py)、[trainer/gan.py](trainer/gan.py)、[trainer/ode.py](trainer/ode.py)：统一使用上面的 dataloader 构造和 epoch cycle。
- [train_main.py](train_main.py)：设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

注意：这些改动只固定随机数和 sampler 路径，不保证 FSDP / NCCL / 浮点归约 bitwise deterministic。
