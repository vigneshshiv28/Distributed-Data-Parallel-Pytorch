# PyTorch Distributed Data Parallel (DDP) — Notes

# 1. Basic DDP Setup

- **Hardware:** 4 GPUs
- **Model replication:** each GPU has a full copy of the model
- **Data parallelism:** each GPU gets a different mini-batch

## Example mapping

| GPU | Data |
| --- | --- |
| GPU0 | Batch A |
| GPU1 | Batch B |
| GPU2 | Batch C |
| GPU3 | Batch D |

---

# 2. Overall Architecture (Mental Picture)

![image.png](image.png)

- **Same model copied on each GPU**
- Each GPU independently runs:

```python
output = model(x)
loss = criterion(output, target)
loss.backward()
```

- No communication in the forward pass; synchronization happens for **gradients** during backward.

---

# 3. Process Initialization

- DDP uses one process per GPU (typical pattern):

```python
torch.multiprocessing.spawn(...)
```

Example:

```python
mp.spawn(example, nprocs=4)
```

This yields:

| Rank | GPU |
| --- | --- |
| 0 | GPU0 |
| 1 | GPU1 |
| 2 | GPU2 |
| 3 | GPU3 |

---

# 4. Process Group Initialization

- Initializes the communication “world”:

```python
dist.init_process_group()
```

- After this, GPUs/processes can communicate using collectives like:
    - `all_reduce`
    - `broadcast`
    - `all_gather`

---

# 5. DDP Constructor

Wrap the model:

```python
ddp_model = DDP(model)
```

Internally, DDP performs:

1. **Parameter broadcast**
2. **Reducer creation**
3. **Bucket formation**
4. **Hook registration**

---

# 6. Parameter Broadcast (Initial Sync)

- Initially, weights may differ across ranks.
- DDP **broadcasts parameters from rank 0** so everyone starts identical.

![image.png](image%201.png)

Result:

- All GPUs have identical initial weights.

---

# 7. Forward Pass (No Communication)

- Each GPU runs forward on its own batch:
    - GPU0: Forward(A)
    - GPU1: Forward(B)
    - GPU2: Forward(C)
    - GPU3: Forward(D)

No synchronization happens here.

---

# 8. Local Loss Computation

- Each GPU computes its own loss:
    - GPU0 → Loss0
    - GPU1 → Loss1
    - GPU2 → Loss2
    - GPU3 → Loss3

Loss values differ because input data differs.

---

# 9. Backward Pass Starts

```python
loss.backward()
```

- Autograd traverses the graph backward.
- Gradients are computed **layer-by-layer**.

---

# 10. Local Gradient Computation (Before Sync)

For a parameter **W**, each GPU computes a different local gradient:

- GPU0: `g0 = ∂L0 / ∂W`
- GPU1: `g1 = ∂L1 / ∂W`
- GPU2: `g2 = ∂L2 / ∂W`
- GPU3: `g3 = ∂L3 / ∂W`

At this point, gradients differ across GPUs.

---

# 11. DDP Hooks

- DDP registers **autograd hooks** on parameters.
- When a gradient becomes ready:
    1. gradient is computed by autograd
    2. **DDP hook fires**
    3. **Reducer is notified**

---

# 12. Gradient Buckets

- DDP groups parameter gradients into **buckets** (chunks), e.g.
    - Bucket0: `[Layer30.grad, Layer29.grad]`
    - Bucket1: `[Layer28.grad, Layer27.grad]`

Why buckets?

- reduce communication overhead
- improve bandwidth utilization

---

# 13. Reducer Logic (High-Level)

Reducer workflow:

1. gradient arrives
2. mark parameter “ready”
3. check whether the bucket is complete
4. when complete → launch communication (all-reduce)

---

# 14. Important Timing (Critical Detail)

**Communication happens during backward**, not after backward completes.

This enables overlap between compute and communication.

---

# 15. AllReduce Communication (Bucket-Level)

When a bucket is ready, DDP uses NCCL (typical GPU backend) to all-reduce:

- `AllReduce(g0, g1, g2, g3)`

---

# 16. Gradient Averaging

AllReduce conceptually does:

1. **Sum**
    - `g_sum = g0 + g1 + g2 + g3`
2. **Divide**
    - `g_avg = g_sum / 4`

After all-reduce, **every GPU has the same averaged gradient**.

---

# 17. Communication Diagram (Concept)

- All GPUs participate in an **ALLREDUCE**
- Outcome: identical gradients on all ranks after synchronization

---

# 18. Overlap of Compute + Communication (DDP Optimization)

While:

- earlier layers are still computing gradients

DDP can already:

- communicate gradients for later layers whose buckets are complete

This hides communication latency and improves scaling.

---

# 19. Timeline Visualization (Intuition)

- Backward compute and all-reduce overlap in time
- Communication latency is partially hidden behind computation

---

# 20. Internal Sequence (End-to-End)

1. Forward pass
2. `loss.backward()`
3. layer gradient computed
4. DDP hook fires
5. gradient added to a bucket
6. bucket fills
7. launch NCCL all-reduce
8. averaged gradients available on all GPUs
9. `optimizer.step()`

---

# 21. Final Optimizer Step

```python
optimizer.step()
```

After synchronization:

- all GPUs have the **same gradients**
- applying the optimizer update keeps **model weights identical** across GPUs

---

# 22. Why DDP Works (Effective Batch Size)

Example:

- local batch size per GPU: 8
- number of GPUs: 4

Effective global batch size:

- `8 × 4 = 32`

Gradient used for the update:

- `∇L = (∇L0 + ∇L1 + ∇L2 + ∇L3) / 4`

This resembles large-batch training.

---

# 23. Important Insight: What DDP Synchronizes

DDP synchronizes **only parameter gradients**.

It does **not** synchronize:

- activations
- losses
- forward outputs

Each GPU runs forward/backward independently; only gradients are shared/averaged.

---

# 24. Internal Components of DDP (Cheat Sheet)

| Component | Purpose |
| --- | --- |
| Autograd Engine | Computes gradients |
| Reducer | Manages synchronization |
| Gradient Buckets | Groups gradients for efficient comms |
| NCCL | GPU communication backend |
| CUDA Streams | Async execution / overlap |

---

# 25. Simplified Mental Model

1. GPU computes local gradients
2. DDP hook detects “ready” gradients
3. reducer launches all-reduce
4. gradients are averaged
5. all GPUs update identically

---

# 26. Most Important Insight (One-Liner)

DDP does **not** do: “complete backward, then communicate”.

Instead, it does:

- **compute gradients layer-by-layer AND communicate simultaneously**

This overlap is why DDP scales efficiently.