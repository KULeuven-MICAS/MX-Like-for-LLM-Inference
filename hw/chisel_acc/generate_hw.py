"""
generate_hw.py — 从 Python 调用 Chisel 生成 FusedDotProductUnit RTL。

用法示例：
    python generate_hw.py                                              # 默认参数
    python generate_hw.py --type-a E4M3 --type-b E2M1 --scale UE5M3 --vec 16
    python generate_hw.py --batch                                      # 穷举所有组合

合法值：
    type-a / type-b : INT8 E5M2 E4M3 E3M2 E2M3 E2M1
    scale           : UE8M0 UE7M1 UE6M2 UE5M3 UE4M4 UE3M5 UE2M6
    vec             : 任意正整数（典型值 1 2 4 8 16 32）
"""

import subprocess
import argparse
from itertools import product
from pathlib import Path

CHISEL_DIR    = Path(__file__).parent  # hw/chisel_acc/
ELEMENT_TYPES = ["INT8", "E5M2", "E4M3", "E3M2", "E2M3", "E2M1"]
SCALE_TYPES   = ["UE8M0", "UE7M1", "UE6M2", "UE5M3", "UE4M4", "UE3M5", "UE2M6"]


def generate(type_a: str, type_b: str, scale: str, vec: int, out_dir: str = None) -> str:
    """
    调用 sbt 生成一个 FusedDotProductUnit 的 SystemVerilog。
    返回输出目录路径。
    """
    if out_dir is None:
        out_dir = f"generated/fused_dot/{type_a}_{type_b}_{scale}_vec{vec}"

    sbt_cmd = (
        f"runMain mx.GenerateFusedDotProduct"
        f" --type-a {type_a}"
        f" --type-b {type_b}"
        f" --scale  {scale}"
        f" --vec    {vec}"
        f" --out-dir {out_dir}"
    )

    print(f">>> 生成: {type_a} x {type_b}, scale={scale}, vec={vec}")
    result = subprocess.run(["sbt", sbt_cmd], cwd=CHISEL_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"sbt 失败（返回码 {result.returncode}）")
    return out_dir


def generate_batch(configs: list[dict]) -> list[str]:
    """
    在同一个 sbt session 里批量生成，避免每次都重启 JVM。

    configs 格式：
        [{"type_a": "E5M2", "type_b": "E4M3", "scale": "UE8M0", "vec": 16}, ...]
    """
    cmds = []
    out_dirs = []
    for c in configs:
        out_dir = c.get("out_dir") or (
            f"generated/fused_dot/{c['type_a']}_{c['type_b']}_{c['scale']}_vec{c['vec']}"
        )
        out_dirs.append(out_dir)
        cmds.append(
            f"runMain mx.GenerateFusedDotProduct"
            f" --type-a {c['type_a']}"
            f" --type-b {c['type_b']}"
            f" --scale  {c['scale']}"
            f" --vec    {c['vec']}"
            f" --out-dir {out_dir}"
        )

    print(f">>> 批量生成 {len(cmds)} 个配置（共用一个 sbt session）…")
    result = subprocess.run(["sbt", "; ".join(cmds)], cwd=CHISEL_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"sbt 批量生成失败（返回码 {result.returncode}）")
    return out_dirs


def generate_all(vec_sizes: list[int] = None) -> list[str]:
    """穷举所有 (type_a, type_b, scale, vec) 组合。"""
    if vec_sizes is None:
        vec_sizes = [1, 2, 4, 8, 16, 32]
    configs = [
        {"type_a": a, "type_b": b, "scale": s, "vec": v}
        for a, b, s, v in product(ELEMENT_TYPES, ELEMENT_TYPES, SCALE_TYPES, vec_sizes)
    ]
    return generate_batch(configs)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Python 驱动 Chisel 硬件生成")
    parser.add_argument("--type-a",  default="E5M2",  choices=ELEMENT_TYPES)
    parser.add_argument("--type-b",  default="E5M2",  choices=ELEMENT_TYPES)
    parser.add_argument("--scale",   default="UE8M0", choices=SCALE_TYPES)
    parser.add_argument("--vec",     default=8,        type=int)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch",   action="store_true",
                        help="穷举所有组合（vec=1/2/4/8/16/32）")
    a = parser.parse_args()

    if a.batch:
        generate_all()
    else:
        generate(
            type_a  = a.type_a,
            type_b  = a.type_b,
            scale   = a.scale,
            vec     = a.vec,
            out_dir = a.out_dir,
        )
