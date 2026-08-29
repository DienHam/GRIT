# Kế hoạch triển khai thực nghiệm GRIT

Tài liệu này chỉ là kế hoạch triển khai. Không có phần nào của GRIT được triển khai trong lần khảo sát này.

Thứ tự thực hiện bắt buộc:

```text
inspect repo
    ↓
design architecture
    ↓
implement minimal GRIT-FO
    ↓
test mathematical primitives
    ↓
smoke run
    ↓
baselines / ablations
```

Nguồn đã khảo sát và commit dùng để tái lập phân tích:

- GRIT proposal, nguồn chân lý của phương pháp: `D:\Desktop\GRIT_Final_Proposal-2.pdf`.
- [NSPO paper](https://arxiv.org/abs/2512.11391), [NSPO official repository](https://github.com/ivanniu/NSPO), commit `b891fdc7692e93833037187700a3bc5c48b47902`, embedded `verl` version `0.5.0.dev`.
- [TROLL paper](https://arxiv.org/abs/2510.03817), [TROLL official repository](https://github.com/NiklasFreymuth/troll), commit `cf06e8482e75e1e46d03f6bfa7cbbcdf2c8875a0`, embedded `verl` version `0.7.0.dev`.
- [discrete_trpl official repository](https://github.com/pbecker93/discrete_trpl), commit `e61cf3c97fb0b9ad70f27331d44f7025c6a39575`.
- Repository GRIT hiện tại: commit `c44d0116297e4053ced2275e8bf6ee12ac605977` trên `main`.

## 1. Hiểu biết về GRIT

GRIT giải quyết forgetting trong RL bằng hai cơ chế bổ sung nhau:

1. **Null-space projection** bảo vệ các ánh xạ tuyến tính quan sát trên `D_pres` ở mức gradient task.
2. **Fixed-base trust-region correction** kiểm tra policy tại predictor và kéo policy trở lại miền KL quanh `π_base` khi bảo vệ bằng linearization không đủ.

Với mỗi protected linear weight `W ∈ R^{m×d}`, chạy frozen `π_base` trên `D_pres`, thu input activations `K ∈ R^{d×N}`, rồi phân rã:

```text
K K^T = U Λ U^T
P = Û Û^T
```

`Û` chứa các eigenvectors có eigenvalue nhỏ hơn relative threshold; proposal và NSPO paper dùng `5e-4`. Task gradient được chiếu ở bên phải:

```text
ĝ_W = (∇_W J_task) P
```

Predictor là một virtual step, không phải một checkpoint được publish cho rollout:

```text
θ̃(θ) = θ + α P ∇_θ J_task(θ)
```

Trên một monitoring batch từ `D_pres`, với mỗi token context `o_t`:

```text
δ_t = KL(π_θ̃(o_t) || π_base(o_t))
```

Nếu `δ_t ≤ ε_pres`, giữ nguyên `π_θ̃`. Nếu vi phạm, giải I-projection:

```text
π_proj = argmin_q KL(q || π_θ̃)
         s.t. KL(q || π_base) ≤ ε_pres
```

Nghiệm có dạng geometric interpolation trong log-probability space:

```text
log π_proj ∝ (η* log π_base + log π_θ̃) / (η* + 1)
```

`η*=0` khi không vi phạm; ngược lại `η*` được tìm bằng scalar bracketing. Preservation loss dùng stop-gradient target:

```text
L_pres(θ̃) = E[KL(π_θ̃ || stopgrad(π_proj))]
L_total(θ) = -J_task(θ) + λ_pres L_pres(θ̃(θ))
```

Gọi `v = ∇_θ̃ L_pres`, exact correction là:

```text
∇_θ L_pres(θ̃(θ)) = v + α H(θ) P v
```

- `GRIT-FO` bỏ `α H(θ) P v`; đây là ưu tiên đầu tiên.
- `GRIT-HVP` tính `H(θ) P v` bằng Pearlmutter HVP hoặc dùng central finite difference chỉ để kiểm chứng.
- Preservation correction `v` **không** được null-space project; nếu project nó thì cơ chế trust-region mất khả năng sửa sai ngoài null space.
- Khi mọi token đều được chấp nhận, `v=0` và update phải suy biến về projected task update.
- Fixed anchor `π_base` không được thay bằng moving `π_old`; fixed anchor mới cho giới hạn không tích lũy `TV(π_proj, π_base) ≤ sqrt(ε_pres/2)`.

Giới hạn phải được phản ánh trong thiết kế thực nghiệm: độ phủ của `D_pres`, projector cố định có thể stale, đảm bảo từng linear layer không suy ra đảm bảo end-to-end qua nonlinearities, sparse KL là xấp xỉ, và `λ_pres`/optimizer quyết định khoảng cách giữa model thực và projected target.

## 2. Bài học liên quan từ NSPO

### 2.1 Phần nên tái sử dụng

- Quy trình khái niệm: forward frozen base model trên preservation data, hook input của `torch.nn.Linear`, cộng dồn non-central covariance `K K^T`, rồi xây null-space basis.
- Giới hạn phạm vi module bằng module-name pattern; official code dùng các `Linear` có `"mlp"` trong tên.
- Tách projection artifact khỏi task data; projector là fixed artifact được tạo trước training.
- Các phép kiểm tra từ paper: symmetry, idempotence, non-expansiveness và residual trên protected activations.
- Cấu hình tham khảo cho reproduction: `Qwen2.5-7B-Instruct`, `Llama3-8B-Instruct`, `PKU-SafeRLHF`, 1,000 preservation samples từ Common sense/Math/Code, threshold `5e-4`, `GRPO`, learning rate `1e-6`.

### 2.2 Phần không được copy nguyên trạng

Official implementation chèn `_get_proj_weight()` trực tiếp vào `verl/verl/workers/fsdp_workers.py` và gọi `amend_perturbation()` từ `RayPPOTrainer.fit()`. Đây là parameter-space retraction định kỳ:

```text
ΔW = W - W_base
W ← W_base + ΔW P
```

nó không phải gradient projection tại mỗi backward. Các vấn đề cụ thể tại commit đã khảo sát:

- `model_path` và `dataset_path` bị hard-code bằng placeholder.
- `store_embedding()` không nhận đối số nhưng được gọi bằng `store_embedding(prompt_length)`.
- Hàm có default threshold `0.0005` nhưng return dùng `0.00005`, lệch paper một bậc độ lớn.
- Full base model được load thêm trong mỗi rank trước FSDP; sample count giữa rank 0 và rank khác không giống nhau.
- CPU projector được `torch.distributed.broadcast()` trong một GPU distributed job; cần kiểm tra backend vì NCCL không broadcast CPU tensor.
- Full `d×d` projector được lưu cho mọi MLP layer, đặc biệt tốn bộ nhớ với `down_proj` có intermediate dimension lớn.
- `amend_perturbation()` được hard-code mỗi `lam=5` outer steps và dùng `FSDP.summon_full_params`; semantics khác `ĝ_W=(∇_W J_task)P`, nhất là với AdamW và nhiều optimizer steps.
- Không có config schema, unit test hoặc failure check riêng cho NSPO path.

Kết luận: tái sử dụng equations, activation-collection idea và benchmark setup; viết lại projector thành một module độc lập, có artifact validation và test. Không port nguyên block trong `fsdp_workers.py`.

## 3. Bài học liên quan từ TROLL

### 3.1 Thành phần có thể tái sử dụng trực tiếp

- `discrete_trpl.sdtrpl_layer.SdtrplLayer`: sparse token-wise trust-region projection, lọc chỉ các token vi phạm, align union support, giải dual theo batch và trả `initial_kl`, `final_kl`, `opt_eta`, projection count/fraction.
- `discrete_trpl.dtrpl_layer.DtrplLayer`: dense oracle thích hợp cho mathematical tests trên vocabulary nhỏ.
- `discrete_trpl.sparsify_logits.sparsify_and_pad_response_logits()` và `sparsify_normalized_logits()`: giữ high-mass tokens, luôn giữ selected token, gán `default_log_prob`, renormalize và chunk để tránh memory spike.
- `verl/verl/workers/actor/dp_actor.py::_forward_micro_batch()`: đã có đường trả sparse normalized logits.
- `verl/verl/workers/fsdp_workers.py::compute_ref_log_prob()`: standalone ref policy đã có thể trả `ref_logits` khi `data.meta_info["compute_logits"]=True`; đây là đường ngắn nhất để lấy fixed-base distributions cho `D_pres`.
- `verl/verl/trainer/ppo/core_algos.py::_compute_trpl_policy_loss()`: preservation regression đang dùng đúng hướng `KL(policy || stopgrad(projected))` qua `policy_logits.kl(projected_logits.detach())`.
- `main.py`, Hydra composition trong `config/`, Ray/FSDP worker lifecycle, vLLM rollout, logging và experiment scripts.

### 3.2 Thay đổi cần thiết khi dùng cho GRIT

- TROLL anchor là moving `π_old` trên task rollout; GRIT phải truyền frozen `π_base` logits trên một batch riêng từ `D_pres`.
- TROLL đưa projected distribution vào task importance ratio; GRIT chỉ dùng projection để tạo preservation target, còn task policy loss vẫn là vanilla PPO/GRPO loss trước null-space projection.
- TROLL tính current logits tại `θ`; GRIT phải tính preservation logits tại virtual predictor `θ̃`.
- Trainer hiện xóa `compute_logits` trước ref-policy call trên task batch. GRIT cần một ref-policy call riêng cho preservation batch, giữ `compute_logits=True` và đổi `ref_logits` thành `pres_base_logits`.
- LoRA branch của `compute_ref_log_prob()` làm rơi logits; minimal GRIT-FO phải reject `lora_rank>0` thay vì âm thầm chạy sai.

### 3.3 Hạn chế của code TROLL cần khóa bằng test/config

- Comment nhắc `trpl_dense`, nhưng end-to-end actor path chỉ wire `trpl` và `trpl_seq` với `SdtrplLayer`.
- `project_full_sequence` có trong config nhưng không được đọc ở path chính.
- `pyproject.toml` khai báo `discrete_trpl={path="./dtrpl"}` trong khi README hướng dẫn clone `./discrete_trpl`.
- Không tìm thấy TROLL-specific unit tests trong fork; `discrete_trpl` cũng không có test suite đi kèm. Thành phần đã được paper-scale experiment kiểm chứng nhưng vẫn cần regression tests trong GRIT.
- TROLL HEAD chứa embedded `verl 0.7.0.dev`; fork đã diverge đáng kể so với current upstream `verl`, vì vậy không rebase đồng thời với việc triển khai GRIT v1.

## 4. So sánh codebase

| Lựa chọn | Ưu điểm | Nhược điểm | Kết luận |
|---|---|---|---|
| **NSPO làm base + port TROLL** | Null-space idea đã nằm trong `fsdp_workers.py`; gần safety-alignment setup | Embedded `verl 0.5.0.dev`; projector code hard-code và có lỗi chạy; projection là retraction mỗi 5 steps; phải port toàn bộ sparse distribution, solver, actor loss và config từ TROLL | Không chọn; phần phải port là phần phức tạp hơn |
| **TROLL làm base + port NSPO** | Embedded `verl 0.7.0.dev`; có entry point, configs, task loaders, sparse logits, solver, ref logits, actor/trainer/FSDP wiring | Cần thêm `D_pres`, projector và virtual predictor; fork cũ hơn current upstream; thiếu tests và license files trong checkout | **Chọn cho v1**; thay đổi mới cô lập và nhỏ hơn |
| **Current upstream `verl` + port cả hai** | Dễ bảo trì lâu dài, nhận bug fixes/FSDP mới | Phải forward-port hai research forks cùng lúc; khó phân biệt lỗi GRIT với API drift; không tối thiểu | Chỉ làm sau khi GRIT-FO chạy và có regression suite |

## 5. Codebase khởi đầu được khuyến nghị

Khởi đầu từ TROLL commit `cf06e8482e75e1e46d03f6bfa7cbbcdf2c8875a0`, pin `discrete_trpl` commit `e61cf3c97fb0b9ad70f27331d44f7025c6a39575`, rồi port **khái niệm** projector của NSPO dưới dạng code mới.

Lý do quyết định:

1. Sparse/full output-distribution handling và differentiable trust-region solver là phần có surface area lớn nhất; TROLL đã nối chúng qua `DataProto`, actor forward, trainer và FSDP worker.
2. Null-space artifact generation và right-side gradient projection có thể cô lập thành một package nhỏ, kiểm thử độc lập.
3. TROLL đã có standalone ref-policy path trả `ref_logits`, phù hợp fixed-base anchoring mà không tạo một RL framework mới.
4. Giữ exact upstream snapshot trong v1 giúp smoke failure có một baseline đối chứng; rebase `verl` là một project riêng sau đó.

Điều kiện trước khi import code:

- [ ] Xác nhận quyền sử dụng/phân phối code từ TROLL, NSPO và `discrete_trpl`; các checkout đã khảo sát không chứa root `LICENSE`. Nếu chưa có quyền, chỉ pin chúng làm research references và port sạch lên một official Apache-2.0 `verl` snapshot tương thích.
- [ ] Ghi các commit trên vào `UPSTREAMS.md`, giữ nguyên provenance và không gộp upgrade `verl` vào PR GRIT-FO.
- [ ] Đặt `discrete_trpl` tại đúng path `dtrpl/` hoặc sửa duy nhất một source mapping trong `pyproject.toml`; không để README và package source lệch nhau.

## 6. Repository hiện tại / Training call graph

### 6.1 Trạng thái repository hiện tại

Tại commit `c44d0116297e4053ced2275e8bf6ee12ac605977`, repository chỉ có:

```text
README.md
LICENSE
.gitignore
```

Không có training entry point, Python package, dependency manifest, config, dataset loader, policy loss, backward, optimizer, FSDP, rollout, reference policy, test hoặc code dẫn xuất từ NSPO/TROLL. Vì vậy **không tồn tại actual local training step để trace**; mọi tên call graph dưới đây là call graph của upstream TROLL được đề nghị import, không được trình bày như code hiện có của GRIT.

### 6.2 Call graph của TROLL base cho một training step

```text
main.py:main
└── main.py:run_ppo
    └── verl.trainer.main_ppo.TaskRunner.run
        ├── RayPPOTrainer.init_workers
        └── RayPPOTrainer.fit
            ├── actor_rollout_wg.generate_sequences
            ├── actor_rollout_wg.compute_log_prob
            │   └── DataParallelPPOActor.compute_log_prob
            │       └── DataParallelPPOActor._forward_micro_batch
            ├── ref_policy_wg.compute_ref_log_prob          # khi cần reference
            ├── compute_advantage
            └── actor_rollout_wg.update_actor
                └── ActorRolloutRefWorker.update_actor
                    └── DataParallelPPOActor.update_policy
                        ├── DataParallelPPOActor._forward_micro_batch
                        ├── get_policy_loss_fn / compute_policy_loss_trpl
                        ├── loss.backward
                        └── DataParallelPPOActor._optimizer_step
                            ├── FSDP.clip_grad_norm_
                            └── actor_optimizer.step
```

`RayPPOTrainer.fit()` tạo rollout, reward và advantages trên driver; `ActorRolloutRefWorker` xử lý device/offload/sharding; `DataParallelPPOActor.update_policy()` chia mini/micro-batches, thực hiện forward/backward và một optimizer step mỗi mini-batch. TROLL lưu sparse `old_logits` từ policy trước update và đưa chúng vào `compute_policy_loss_trpl()`.

### 6.3 Call graph mục tiêu của GRIT-FO

```text
RayPPOTrainer.fit
├── task rollout/reward/advantage                         # giữ nguyên
├── next(preservation_dataloader)                         # mới
├── ref_policy_wg.compute_ref_log_prob(compute_logits=True)
│   └── pres_base_logits                                  # fixed π_base
└── actor_rollout_wg.update_actor(task + preservation)
    └── DataParallelPPOActor.update_policy
        ├── task forward/backward với projector enabled
        ├── GritFOUpdater.capture_projected_task_grad
        ├── GritFOUpdater.virtual_predictor_step          # θ → θ̃, no optimizer state
        ├── preservation forward tại θ̃
        ├── SdtrplLayer(pres_logits, pres_base_logits)
        ├── KL(pres_logits || stopgrad(projected_logits)).backward
        ├── GritFOUpdater.restore_parameters              # θ̃ → θ trong finally
        ├── combine task_grad + λ_pres * v
        └── một final actor_optimizer.step
```

## 7. Kiến trúc GRIT đề xuất

### 7.1 Ranh giới module

| Thành phần | Files | Trách nhiệm |
|---|---|---|
| Offline artifact preparation | `tools/prepare_grit_artifacts.py`, `verl/verl/experimental/grit/projector.py` | Load frozen base, tạo deterministic `D_pres` contexts, hook activations, tích lũy covariance, tạo compact projector basis và manifest |
| Trainer data flow | `verl/verl/trainer/main_ppo.py`, `verl/verl/trainer/ppo/ray_trainer.py` | Tạo ref worker khi `grit.enabled`, tạo/checkpoint preservation dataloader, lấy fixed-base sparse logits, ghép task/preservation data |
| FSDP integration | `verl/verl/workers/fsdp_workers.py`, `verl/verl/experimental/grit/projector.py` | Load/validate projector trước wrapping, đăng ký task-only gradient projection, đảm bảo behavior đồng nhất trên ranks |
| Actor update | `verl/verl/workers/actor/dp_actor.py`, `verl/verl/experimental/grit/update.py` | Hai backward passes, virtual predictor, restore an toàn, gradient combination, final optimizer step |
| Trust-region loss | `verl/verl/experimental/grit/trust_region.py`, existing `dtrpl/discrete_trpl/*` | Adapter mỏng quanh `SdtrplLayer`, mask, aggregation, stop-gradient và metrics; dense oracle cho tests |
| Config/experiments | `config/method/grit_fo.yaml`, `config/_runs/grit_smoke.yaml`, `config/preservation/default.yaml` | Một nguồn cấu hình cho method, preservation data và smoke run |

### 7.2 Public configuration contract

Thêm config tối thiểu sau; không thêm knobs chưa có use case:

```yaml
grit:
  enabled: true
  variant: fo
  lambda_pres: 1.0
  epsilon_pres: 0.05
  predictor_step_size: null       # null = current optimizer LR
  projector:
    path: ???
    module_regex: ".*mlp\\.(gate_proj|up_proj)$"
    eigenvalue_threshold: 5.0e-4
  preservation:
    train_files: ???
    batch_size: 4
    seed: 66
  sparsify_logits: ${actor_rollout_ref.actor.sparsify_logits}
```

Minimal v1 constraints được validate lúc startup:

- `grit.variant == fo`.
- `strategy == fsdp`, `fsdp_config.use_orig_params == true`; `fsdp2`, Megatron và LoRA trả lỗi rõ ràng thay vì fallback.
- `ppo_epochs == 1`; mỗi mini-batch nhận đúng một preservation mini-batch. Micro-batch accumulation vẫn được phép.
- `π_base` path/revision và tokenizer fingerprint phải khớp projector manifest.
- `predictor_step_size` phải dương; default là LR hiện tại của actor optimizer.
- Sparse path dùng cùng `default`, `threshold`, `total_default_mass` và `total_default_keep_maxnum` cho predictor/base distributions.

### 7.3 Data contract

Preservation data dùng teacher-forced contexts được frozen base generate một lần với fixed seed. Trước `update_actor`, trainer ghép các key cùng leading batch dimension:

```text
pres_input_ids
pres_attention_mask
pres_position_ids
pres_responses
pres_response_mask
pres_base_logits
```

`DataParallelPPOActor.update_policy()` tách prefix `pres_` trước khi gọi `_forward_micro_batch()`. Không dùng task responses làm `D_pres`, không dùng current actor để sinh preservation contexts, và không cập nhật `π_base` trong training.

### 7.4 Projector representation

Không lưu full `P ∈ R^{d×d}` nếu có thể tránh. Artifact lưu orthonormal basis nhỏ hơn:

- Nếu significant subspace nhỏ hơn, lưu `U_sig` và áp dụng `gP = g - (gU_sig)U_sig^T`.
- Nếu null subspace nhỏ hơn, lưu `U_null` và áp dụng `gP = (gU_null)U_null^T`.
- Manifest ghi `mode`, shape, dtype, rank, threshold, module name, model/dataset fingerprint và activation count.
- `gate_proj` và `up_proj` trong cùng MLP layer dùng chung input activations; deduplicate basis theo layer/input signature.
- Minimal v1 chỉ bảo vệ `gate_proj|up_proj`. `down_proj` được đưa vào layer-coverage ablation sau khi có memory benchmark, vì input dimension lớn hơn nhiều.

### 7.5 Virtual predictor và optimizer semantics

Minimal correctness mode dùng `torch.optim.SGD`, `momentum=0`, `weight_decay=0`, không gradient clipping để net update khớp equations. `GritFOUpdater`:

1. Lưu projected task grads trên local FSDP shards.
2. Cộng `-α g_task` vào parameter shards trong `torch.no_grad()` để tạo `θ̃`; không gọi `optimizer.step()` và không thay optimizer state/scheduler.
3. Tính `v` tại `θ̃` với projector disabled.
4. Khôi phục `θ` trong `try/finally`, kiểm tra restore residual.
5. Gán `g_task + λ_pres v` vào `.grad`, gọi đúng một final `optimizer.step()`, rồi advance scheduler đúng một lần.

AdamW là một ablation/engineering extension sau GRIT-FO smoke. Không gọi hai AdamW steps vì sẽ advance moments hai lần và không còn tương ứng với thuật toán.

## 8. Phase 1 — Minimal GRIT-FO

### 8.1 Bootstrap codebase

- [ ] **Files:** toàn bộ TROLL snapshot, `pyproject.toml`, `UPSTREAMS.md`.
  - **Reuse:** TROLL commit đã pin, embedded `verl`, existing Hydra/run configs.
  - **New:** provenance file và dependency pin; không sửa logic ở bước import.
  - **Dependencies:** license clearance, Python/CUDA/vLLM versions tương thích TROLL.
  - **Validation:** chạy unmodified `+_runs=debug` hoặc một config rút gọn; lưu exact command và environment lock.
  - **Done:** TROLL baseline khởi động, rollout một batch và hoàn thành một actor update trước mọi GRIT diff.

### 8.2 Tạo preservation/projector artifacts

- [ ] **Files:** `tools/prepare_grit_artifacts.py`, `verl/verl/experimental/grit/projector.py`.
  - **Reuse:** NSPO activation-hook/covariance workflow ở mức thuật toán; Hugging Face model/tokenizer loading hiện có.
  - **New:** deterministic sampling/generation, hook lifecycle, covariance accumulation, `torch.linalg.eigh`, compact basis, manifest/fingerprint và atomic artifact write.
  - **Affected APIs:** `ProjectorSpec`, `ProjectorManifest`, `load_projectors()`, `project_gradient()`.
  - **Dependencies:** frozen base model, mixed Common sense/Math/Code `D_pres`, enough CPU RAM/disk.
  - **Validation:** rerun cùng seed tạo cùng sample IDs/basis ranks; mọi module trong manifest tồn tại và shape khớp model.
  - **Done:** một artifact nhỏ cho smoke model và một preservation parquet có tokenized base-generated contexts được load độc lập.

### 8.3 Nối fixed-base preservation flow

- [ ] **Files:** `verl/verl/trainer/main_ppo.py`, `verl/verl/trainer/ppo/ray_trainer.py`, `config/preservation/default.yaml`.
  - **Reuse:** `TaskRunner.add_ref_policy_worker()`, `RayPPOTrainer` dataloader lifecycle, `compute_ref_log_prob(compute_logits=True)`, `DataProto`.
  - **New:** `grit.enabled` buộc tạo frozen ref worker; preservation dataloader/iterator có checkpoint state; một ref call riêng trả `pres_base_logits`; prefix/align/repeat preservation batch với actor mini-batches.
  - **Dependencies:** standalone ref policy; v1 reject `ref_in_actor`/LoRA nếu path không trả logits.
  - **Validation:** hash ref parameters không đổi; cùng context ở step khác cho cùng sparse base logits; preservation iterator resume đúng sau checkpoint.
  - **Done:** actor worker nhận đủ sáu `pres_*` keys, đúng shape/mask và fixed anchor.

### 8.4 Project task gradients dưới FSDP

- [ ] **Files:** `verl/verl/workers/fsdp_workers.py`, `verl/verl/experimental/grit/projector.py`.
  - **Reuse:** TROLL FSDP model construction, worker dispatch/offload lifecycle.
  - **New:** load projector trước FSDP wrapping; task-only hook/context flag; rank-consistency checks; lazy basis device transfer/cache.
  - **Affected APIs:** `ActorRolloutRefWorker._build_model_optimizer()`, `GritGradientProjector.enable_task_projection()`.
  - **Dependencies:** `use_orig_params=True`, identical artifact trên mọi rank.
  - **Validation:** protected `weight.grad` bằng dense `gP` oracle trên one-rank và two-rank toy FSDP; unprotected params giữ nguyên; preservation backward không bị project.
  - **Done:** mỗi task backward được chiếu đúng một lần trước optimizer step, không còn retraction định kỳ kiểu NSPO.

### 8.5 Virtual predictor và preservation correction

- [ ] **Files:** `verl/verl/workers/actor/dp_actor.py`, `verl/verl/experimental/grit/update.py`, `verl/verl/experimental/grit/trust_region.py`.
  - **Reuse:** `_forward_micro_batch()`, `agg_loss()`, `SdtrplLayer`, `SparseLogProb.kl()`, existing optimizer/scheduler ownership.
  - **New:** tách task/preservation micro-batches; capture grads; reversible virtual step; preservation forward; fixed-base projection; stop-gradient KL; combine gradients; hard failure trên non-finite hoặc restore mismatch.
  - **Affected APIs:** `DataParallelPPOActor.update_policy()`, `GritFOUpdater`, `compute_grit_preservation_loss()`.
  - **Dependencies:** Phase 8.3/8.4, SGD correctness config.
  - **Validation:** `λ_pres=0` bằng projected-task-only update; `P=I` tắt null projection; mọi token accepted cho `v=0`; violating tokens có `final_kl≤epsilon_pres+tolerance`.
  - **Done:** một mini-batch tạo đúng net update `θ_new = θ - α(g_task + λ_pres v)` với `v` đo tại `θ̃` và chỉ một final optimizer step.

### 8.6 Config và logging

- [ ] **Files:** `config/method/grit_fo.yaml`, `config/_runs/grit_smoke.yaml`, existing logging in `ray_trainer.py`/`dp_actor.py`.
  - **Reuse:** Hydra composition, TROLL metric reduction, W&B/logger adapters.
  - **New metrics:** `grit/task_grad_norm`, `grit/projected_grad_norm`, `grit/null_residual`, `grit/pres_loss`, `grit/pres_grad_norm`, `grit/initial_kl_{mean,max}`, `grit/final_kl_{mean,max}`, `grit/projected_fraction`, `grit/eta_mean`, `grit/predictor_restore_residual`, `grit/sparse_nnz`, time và peak memory.
  - **Validation:** metrics finite và giống nhau sau distributed reduction; config dump chứa model/data/artifact fingerprints.
  - **Done:** mỗi step đủ dữ liệu để phân biệt lỗi projector, solver, preservation loss và distributed update.

## 9. Phase 2 — Tests cho mathematical primitives

Tạo tests trong `verl/tests/experimental/grit/`; không bắt đầu smoke trước khi toàn bộ CPU/GPU-available tests xanh.

- [ ] **Projector algebra:** random orthonormal basis; kiểm tra `P^T=P`, `P^2=P`, `||gP||≤||g||`, compact và dense implementations bằng nhau.
- [ ] **Activation preservation:** với toy `Linear`, kiểm tra `(W-αgP)K ≈ WK` khi `PK≈0`; đo relative residual, không chỉ absolute tolerance.
- [ ] **Threshold behavior:** eigenvalues ngay dưới/trên `5e-4*λ_max`, zero-rank/full-rank, repeated eigenvalues và invalid manifest.
- [ ] **Trust-region dense oracle:** so `DtrplLayer`/analytic small-vocabulary solution với `SdtrplLayer`; accepted token giữ distribution, violated token thỏa `final_kl≤ε_pres+tolerance` và `η*>0`.
- [ ] **Sparse approximation:** union support khác nhau, selected token bắt buộc, discarded mass/default normalization, `K∈{16,64,256}`, chunked và unchunked outputs.
- [ ] **Preservation loss direction:** finite difference xác nhận gradient của `KL(π_θ̃ || stopgrad(π_proj))`; projected target không nhận gradient.
- [ ] **GRIT-FO update:** toy MLP so implementation với hand-computed `θ-α(g_task+λ_pres v)`; test `λ_pres=0`, `P=I`, `ε_pres=∞`, mọi token vi phạm và không token vi phạm.
- [ ] **Predictor transaction:** exception trong preservation forward vẫn restore parameters bitwise/within dtype tolerance và không advance optimizer/scheduler.
- [ ] **FSDP parity:** one GPU non-FSDP, one-rank FSDP và two-rank FSDP cho cùng seed cho projected gradient/update tương đương; kiểm tra gradient accumulation.
- [ ] **Checkpoint/resume:** preservation iterator, optimizer, scheduler, projector fingerprint và global step khôi phục để next-step metrics/update khớp uninterrupted run.

Definition of done cho Phase 2: tests deterministic, không `xfail` cho path GRIT-FO được hỗ trợ, và dense-vs-sparse tolerances được ghi rõ theo dtype.

## 10. Phase 3 — Smoke run

### 10.1 Smoke ladder

- [ ] **Level 0 — import/config:** compose `config/_runs/grit_smoke.yaml`, load artifacts và validate model/module fingerprints mà không khởi tạo rollout.
- [ ] **Level 1 — single-process:** tiny causal LM hoặc `Qwen3-0.6B`, một task batch và một preservation batch, một GRIT-FO update; xác nhận loss/gradient/restore finite.
- [ ] **Level 2 — end-to-end Ray/vLLM:** `Qwen3-0.6B` + GSM8K từ TROLL, `n_gpus=1` nếu GPU ≥24 GB hoặc `n_gpus=4` theo debug config, 2–5 outer steps, validation/checkpoint một lần.
- [ ] **Level 3 — distributed:** cùng seed/config trên 2 hoặc 4 GPUs, FSDP sharding, optimizer/parameter hashes đồng nhất giữa ranks sau mỗi step.

### 10.2 Acceptance criteria

- Unmodified TROLL baseline và GRIT dùng cùng task rollout/reward path.
- `π_base` hash không thay đổi; projector artifact chỉ được read.
- Không có OOM/deadlock trong ref-logit call, two backward passes, virtual step hoặc vLLM weight sync.
- `predictor_restore_residual` dưới dtype tolerance trước final update.
- `final_kl` của projected sparse targets không vượt `epsilon_pres` ngoài solver tolerance.
- `projected_fraction=0` cho batch nằm trong bound và preservation correction norm bằng 0.
- Checkpoint resume tạo cùng next update trong deterministic smoke.
- Ghi wall time, tokens/s, peak GPU/CPU memory và overhead so với matching projected-task-only baseline.

Definition of done cho Phase 3: một command được lưu trong README/experiment log có thể chạy từ clean environment đến checkpoint mà không sửa hard-coded path.

## 11. Phase 4 — GRIT-HVP

Chỉ bắt đầu sau khi GRIT-FO hoàn thành Phase 2 và Phase 3.

- [ ] **Files:** `verl/verl/experimental/grit/update.py`, `dp_actor.py`, `config/method/grit_hvp.yaml`, HVP tests.
- [ ] Giữ task graph với `create_graph=True` chỉ khi batch có `v≠0`; không trả cost HVP cho accepted predictor.
- [ ] Tính `u=P v` trên protected components rồi Pearlmutter product `H(θ)u` bằng `torch.autograd.grad`; unprotected components dùng identity theo định nghĩa global `P`.
- [ ] Kết hợp correction `v + αH(θ)Pv`; preservation term vẫn không bị project lần nữa.
- [ ] Dùng central finite difference của task gradient chỉ làm test oracle trên toy model, không làm production backend.
- [ ] So HVP với explicit Hessian trên model rất nhỏ và central finite difference; kiểm tra sign convention giữa maximizing `J_task` và minimizing policy loss.
- [ ] Benchmark activation memory/runtime dưới gradient checkpointing và FSDP; hard fail nếu graph bị free trước HVP.
- [ ] Chỉ mở full-model HVP experiments nếu GRIT-HVP update vượt numerical tests và không tăng memory quá budget đã ghi nhận.

Definition of done: `GRIT-HVP` và `GRIT-FO` trùng nhau khi `v=0`; HVP relative error đạt tolerance; end-to-end smoke hoàn tất với metrics/cost riêng.

## 12. Baselines và ablations

### 12.1 Experimental ladder

1. **Primary safety-forgetting study:** `Qwen2.5-7B-Instruct`, `PKU-SafeRLHF` task RL, `D_pres` 1,000 mixed Common sense/Math/Code samples theo NSPO; 3 seeds sau khi single-seed pipeline ổn định.
2. **Cross-domain sanity study:** `Qwen3-0.6B` hoặc `Qwen3-1.7B` trên GSM8K từ TROLL, dùng held-out instruction/STEM/code preservation mix; mục tiêu là kiểm tra phương pháp không phụ thuộc safety reward.
3. Chỉ scale sang `Llama3-8B-Instruct`, DAPO/Eurus hoặc model lớn hơn sau khi hai study trên có task-retention Pareto signal.

### 12.2 Baselines tối thiểu, cùng model/data/seed/optimizer budget

- [ ] Frozen `π_base` — mốc task và preservation trước RL.
- [ ] Vanilla GRPO/PPO — không null projection, không preservation correction.
- [ ] GRPO/PPO + fixed-base KL regularization — baseline regularization phổ biến.
- [ ] NSPO — task gradient projection, `λ_pres=0`.
- [ ] TROLL — moving `π_old` trust region từ official path.
- [ ] Fixed-base trust correction only — `P=I`, giữ predictor/correction.
- [ ] GRIT-FO — full minimal method.
- [ ] GRIT-HVP — chỉ thêm sau Phase 4.

Không trộn DPO/SafeRLHF/MoCAN/PeCAN/W-DOOR/BFPO vào first implementation matrix. Chỉ thêm các paper baselines này nếu claim cuối cùng là state-of-the-art safety alignment thay vì cơ chế forgetting-resistant RL.

### 12.3 Ablations theo thứ tự ưu tiên

- [ ] Component: `P=I`, `λ_pres=0`, `α=0`, moving `π_old` anchor thay fixed `π_base`.
- [ ] Trust region: `epsilon_pres`, `lambda_pres`, sparse `K`/mass threshold/default mass; dense oracle chỉ ở model/vocabulary nhỏ.
- [ ] Projector: threshold, `D_pres` size/domain coverage, `gate_proj|up_proj` so với full MLP, fixed projector so với refresh.
- [ ] Approximation: `GRIT-FO` so với `GRIT-HVP`; SGD exact mode so với AdamW engineering mode sau correctness gate.
- [ ] Data: in-domain, mixed-domain và deliberately mismatched `D_pres` để định lượng coverage limitation.

### 12.4 Metrics và reporting

- Task: reward, success/accuracy; safety study báo ASR trên AdvBench, PKU-SafeRLHF, HarmBench, JailbreakBench, SORRY-Bench, HarmfulQA và ALERT nếu evaluation dependencies sẵn có.
- Preservation: delta từ `π_base` trên MMLU, SuperGPQA, AlpacaEval, GSM8K, MATH, OlympiadBench và LiveCodeBench; dùng cùng evaluation harness và decoding settings.
- Mechanism: held-out fixed-base KL, violation/projected fraction, `η*`, `||gP||/||g||`, null residual, correction/task norm ratio và cosine.
- Systems: throughput, wall time, peak GPU memory, CPU projector size, sparse nnz/token, HVP overhead.
- Statistics: 3 independent seeds cho main table, mean±std; cùng sampled data order/rollout seeds khi có thể; plot task gain so với preservation drop thay vì chỉ một scalar aggregate.

## 13. Rủi ro / Câu hỏi mở

| Rủi ro | Ảnh hưởng | Quyết định/giảm thiểu |
|---|---|---|
| Upstream repositories thiếu visible root license trong checkout | Không thể mặc nhiên copy/phân phối | License clearance là gate trước import; giữ provenance và commit pins |
| Repository GRIT chưa có code | Không có local baseline/call graph để regression | Import TROLL nguyên trạng và smoke trước GRIT diff |
| TROLL embedded `verl` đã cũ so với current upstream | Bug fixes/API mới bị thiếu | Pin cho v1; rebase riêng sau test suite, không làm cùng GRIT PR |
| Virtual predictor dưới FSDP | In-place shard mutation/restore có thể sai lifecycle | v1 chỉ `FSDP1 + use_orig_params=True`; transaction tests và `try/finally`; hard fail unsupported modes |
| Optimizer không khớp equations | AdamW virtual/final semantics có thể sai | correctness mode dùng SGD exact; AdamW là ablation có nhãn, không âm thầm đổi |
| Full projector memory | `down_proj` covariance/projector rất lớn | compact basis, deduplicate shared inputs, v1 chỉ `gate_proj|up_proj`, benchmark trước mở rộng |
| Sparse KL khác dense KL | Bound có thể chỉ đúng trên approximation | dense oracle tests, retained-mass metrics, held-out dense checks trên small model |
| `D_pres` không đại diện | Fixed-base bound không bảo vệ ngoài support | mixed domains, held-out preservation evaluation, coverage ablation |
| Fixed projector stale | Linear preservation giảm theo training | log residual theo step; refresh chỉ là ablation, không thay v1 |
| Ref worker/LoRA path làm rơi logits | Anchor sai hoặc thiếu | v1 standalone ref only; startup assertion; LoRA deferred |
| Hai backward passes tăng cost/memory | OOM hoặc giảm throughput | micro-batch preservation riêng, sparse logits, activation checkpointing, cost metrics |
| Safety reward/evaluation dùng judge lớn | Compute và reproducibility cao | smoke bằng GSM8K; safety main run chỉ bắt đầu khi reward/judge artifacts được pin |

Giả định mặc định của kế hoạch: mục tiêu đầu tiên là chứng minh correctness và end-to-end feasibility của GRIT-FO, không tái lập toàn bộ bảng kết quả NSPO/TROLL trong cùng milestone; phần cứng full-scale chưa được chỉ định nên smoke có 1-GPU và multi-GPU gates, còn main study 7B được schedule sau khi resource budget được xác nhận.

## 14. Definition of Done

### Inspect repo

- [ ] Current skeleton, upstream commits, versions, licenses, call graph và known source defects được ghi lại; không có symbol/file giả định là đang tồn tại trong GRIT.
- [ ] Unmodified pinned TROLL hoàn thành một baseline update.

### Design architecture

- [ ] Config/data/artifact contracts ở Section 7 được implement đúng, unsupported paths hard fail.
- [ ] Fixed `π_base`, task-only projector và preservation-only correction có ownership rõ ràng.

### Minimal GRIT-FO

- [ ] Projector artifact deterministic, compact và fingerprinted.
- [ ] Task gradients được right-project tại mỗi backward; preservation gradient không bị project.
- [ ] Predictor được tạo/restore an toàn; chỉ một final optimizer/scheduler step được commit.
- [ ] Sparse fixed-base trust-region và stop-gradient preservation loss chạy end-to-end.

### Mathematical tests

- [ ] Projector, dense/sparse trust-region, GRIT-FO update, finite difference, FSDP parity và checkpoint tests đều pass.
- [ ] Tolerances theo dtype và solver được định nghĩa, không dựa vào visual inspection.

### Smoke run

- [ ] Single-process và Ray/vLLM/FSDP smoke hoàn tất, không OOM/deadlock/non-finite.
- [ ] Checkpoint resume deterministic; logs đủ mechanism và systems metrics.

### Baselines / ablations

- [ ] Vanilla, fixed-base KL, NSPO, TROLL, trust-only và GRIT-FO chạy trên cùng experimental contract.
- [ ] Main results báo task-retention Pareto, 3 seeds và systems overhead.
- [ ] GRIT-HVP chỉ được đưa vào bảng sau khi riêng Phase 4 pass tests và smoke.

Toàn bộ milestone được xem là hoàn tất khi một người khác có thể clone repository sạch, chuẩn bị artifacts, chạy baseline, chạy GRIT-FO, resume checkpoint và tái tạo metrics bằng documented commands mà không sửa source hoặc hard-coded path.
