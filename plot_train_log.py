# -*- coding: utf-8 -*-
"""把训练日志画成 loss 曲线，并汇总每轮验证指标。

训练脚本每轮往日志里写一行，格式（逗号或空白分隔均可）：
    epoch,train_loss,val_loss
    1,3.8421,3.6102
    2,2.9310,2.8874
    ...
可选再加一列 mAP：
    epoch,train_loss,val_loss,mAP
    1,3.8421,3.6102,0.612

用法：
    python plot_train_log.py --log train_log.csv
    python plot_train_log.py --log train_log.csv --out curves.png
    python plot_train_log.py --demo            # 没有日志时看效果
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # 服务器/无 GUI 环境也能出图
import matplotlib.pyplot as plt  # noqa: E402


def read_log(path):
    """读取训练日志，返回列名和逐轮数据。逗号或空白分隔都能读。"""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        delimiter = "," if "," in first else None
        reader = csv.reader(f, delimiter=delimiter)
        header = [h.strip() for h in next(reader)]
        for line in reader:
            values = [x for x in line if x.strip() != ""]
            if not values:
                continue
            rows.append([float(x) for x in values])
    if not rows:
        raise SystemExit("日志为空，请检查文件格式")
    return header, rows


def demo_data():
    """构造一份形状真实的假日志，用来验证绘图流程。"""
    rows = []
    for epoch in range(1, 21):
        train_loss = 4.0 * (0.83 ** epoch) + 0.35
        val_loss = train_loss * 1.04 + 0.02
        rows.append([epoch, round(train_loss, 4), round(val_loss, 4)])
    return ["epoch", "train_loss", "val_loss"], rows


def plot(header, rows, out):
    epochs = [r[0] for r in rows]
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(epochs, [r[1] for r in rows], marker="o", ms=3, label=header[1])
    if len(header) > 2:
        plt.plot(epochs, [r[2] for r in rows], marker="s", ms=3, label=header[2])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("RetinaFace training curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"曲线已保存：{out}")

    best = min(rows, key=lambda r: r[2] if len(r) > 2 else r[1])
    print(f"最优轮次：epoch={int(best[0])}，train_loss={best[1]:.4f}，"
          f"val_loss={best[2] if len(best) > 2 else float('nan'):.4f}")
    if len(header) > 3:
        last_map = rows[-1][3]
        print(f"最后一轮 mAP：{last_map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="绘制 RetinaFace 训练曲线")
    parser.add_argument("--log", help="CSV 日志路径")
    parser.add_argument("--out", default="train_curves.png", help="输出图片路径")
    parser.add_argument("--demo", action="store_true", help="没有日志时演示")
    args = parser.parse_args()

    if args.demo:
        header, rows = demo_data()
    elif args.log:
        if not Path(args.log).exists():
            sys.exit(f"找不到日志文件：{args.log}")
        header, rows = read_log(args.log)
    else:
        parser.error("请提供 --log 或使用 --demo")

    plot(header, rows, args.out)


if __name__ == "__main__":
    main()
