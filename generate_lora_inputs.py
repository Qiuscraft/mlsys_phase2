import argparse
import logging
from pathlib import Path

import torch

from mlsys_phase2.logging_utils import setup_logging


D_VALUES = (3584, 4608)
RANK = 16
logger = logging.getLogger(__name__)


def generate_inputs(output_root: Path, seed: int, scale: float, overwrite: bool) -> None:
  output_root.mkdir(parents=True, exist_ok=True)

  torch.manual_seed(seed)

  for d in D_VALUES:
      out_dir = output_root / f"d{d}"
      out_dir.mkdir(parents=True, exist_ok=True)

      paths = {
          "W": out_dir / "W.pt",
          "X": out_dir / "X.pt",
          "A": out_dir / "A.pt",
          "B": out_dir / "B.pt",
      }

      if not overwrite and all(p.exists() for p in paths.values()):
          logger.info("[skip] d=%s: all files already exist in %s", d, out_dir)
          continue

      logger.info("[generate] d=%s -> %s", d, out_dir)

      # 生成 CPU float32 contiguous tensor。
      # 使用较小 scale，避免 W @ X 数值过大，同时保持 benchmark 形状真实。
      W = (torch.randn((d, d), dtype=torch.float32) * scale).contiguous()
      X = (torch.randn((d, d), dtype=torch.float32) * scale).contiguous()
      A = (torch.randn((d, RANK), dtype=torch.float32) * scale).contiguous()
      B = (torch.randn((d, RANK), dtype=torch.float32) * scale).contiguous()

      torch.save(W, paths["W"])
      torch.save(X, paths["X"])
      torch.save(A, paths["A"])
      torch.save(B, paths["B"])

      del W, X, A, B

      logger.info("[done] d=%s", d)

  logger.info("All inputs generated under: %s", output_root.resolve())


def main() -> None:
  setup_logging(log_to_file=False)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-root",
      default="inputs",
      help="输出根目录，默认 ./inputs",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=20260508,
      help="随机种子",
  )
  parser.add_argument(
      "--scale",
      type=float,
      default=0.01,
      help="随机张量缩放系数",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="覆盖已存在文件",
  )
  args = parser.parse_args()

  generate_inputs(
      output_root=Path(args.output_root),
      seed=args.seed,
      scale=args.scale,
      overwrite=args.overwrite,
  )


if __name__ == "__main__":
  main()