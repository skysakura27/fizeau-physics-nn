"""CPU 训练速度实测 + GPU 加速预估"""
import time, torch
from src.models.integrated_net import UnrolledPINN

model = UnrolledPINN(n_iters=5, share_weights=True)
model.eval()
x = torch.randn(16, 1, 128, 128)

# warmup
with torch.no_grad():
    _ = model(x)

# inference
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(5):
        _ = model(x)
inf_time = (time.perf_counter() - t0) / 5

# training step
model.train()
y = torch.randn(16, 1, 128, 128)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
for _ in range(2):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()

t0 = time.perf_counter()
for _ in range(5):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
train_step = (time.perf_counter() - t0) / 5

batches_per_epoch = 29
epoch_time = train_step * batches_per_epoch
total_100 = epoch_time * 100

print(f"CPU batch(16) 推理: {inf_time*1000:.0f} ms")
print(f"CPU batch(16) 训练步: {train_step*1000:.0f} ms")
print(f"CPU 每 epoch: {epoch_time:.1f} s")
print(f"CPU 100 epochs: {total_100/60:.1f} min")
print()
print("GPU 预估 (典型加速比):")
print(f"  NVIDIA RTX 3060 (~15x): {total_100/60/15:.1f} min")
print(f"  NVIDIA RTX 4090 (~40x): {total_100/60/40:.1f} min")
print(f"  Google Colab T4 (~10x): {total_100/60/10:.1f} min (免费)")
print(f"  Intel Arc (IPEX ~5x):   {total_100/60/5:.1f} min")
