"""测 ControlNet 微调的峰值显存：几种省显存配置各跑 5 步，打印 torch 峰值显存（不保存任何东西）"""
import sys

import torch

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
INIT_CN = "lllyasviel/control_v11p_sd15_canny"
DEV = "cuda"


def run(res, batch, eightbit, dtype):
    from diffusers import ControlNetModel, UNet2DConditionModel
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=dtype).to(DEV).eval().requires_grad_(False)
    cn = ControlNetModel.from_pretrained(INIT_CN).to(DEV).train()
    unet.enable_gradient_checkpointing(); cn.enable_gradient_checkpointing()
    if eightbit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(cn.parameters(), lr=1e-5)
    else:
        opt = torch.optim.AdamW(cn.parameters(), lr=1e-5)
    lat_res = res // 8
    for _ in range(5):
        lat = torch.randn(batch, 4, lat_res, lat_res, device=DEV)
        emb = torch.randn(batch, 77, 768, device=DEV)                 # 文本编码与 VAE 在 CPU/预计算，不占 GPU
        cond = torch.rand(batch, 3, res, res, device=DEV)
        t = torch.randint(0, 1000, (batch,), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            down, mid = cn(lat, t, encoder_hidden_states=emb, controlnet_cond=cond, return_dict=False)
            pred = unet(lat, t, encoder_hidden_states=emb, down_block_additional_residuals=down,
                        mid_block_additional_residual=mid).sample
            loss = pred.float().pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    del unet, cn, opt; torch.cuda.empty_cache()
    return peak, reserved


if __name__ == "__main__":
    res, batch, eightbit = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "8bit"
    dtype = torch.bfloat16 if sys.argv[4] == "bf16" else torch.float32
    peak, reserved = run(res, batch, eightbit, dtype)
    print(f"res {res} batch {batch} {'8-bit' if eightbit else 'fp32'} AdamW, frozen UNet {sys.argv[4]}: "
          f"peak allocated {peak:.0f} MiB, peak reserved {reserved:.0f} MiB (+ ~500 MiB CUDA context)", flush=True)
