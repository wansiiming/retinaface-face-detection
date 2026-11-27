# -*- coding: utf-8 -*-
"""Anchor 生成器单元测试。

anchors.py 在模块顶层 import torch，但我们只测 generate_all_anchors（纯 numpy），
所以这里用一个空的 torch 替身占位，避免为了跑测试去装 2GB 的 PyTorch。

运行：pytest -q tests
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("torch", types.ModuleType("torch"))

from anchors import AnchorGenerator  # noqa: E402


@pytest.fixture(scope="module")
def anchors():
    return AnchorGenerator(input_size=640).generate_all_anchors()


def test_total_anchor_count(anchors):
    """P2~P7 六层、每层 3 种尺度：102375 个候选框。"""
    assert anchors.shape == (102375, 4)
    assert anchors.dtype == np.float32


def test_anchor_count_per_level():
    """每层数量 = 特征图边长² × 3。"""
    gen = AnchorGenerator()
    expected = sum(lvl["size"] ** 2 for lvl in gen.levels) * gen.num_anchors_per_loc
    assert expected == 102375


def test_first_anchor_is_centered_on_stride():
    """P2 第一个 anchor：步长 4，中心 (2, 2)，边长 16。"""
    gen = AnchorGenerator()
    first = gen.generate_all_anchors()[0]
    assert np.allclose(first, [-6.0, -6.0, 10.0, 10.0])


def test_three_scales_per_location():
    """同一位置生成 3 个框，边长依次为 base 的 1 / 1.5 / 2 倍。"""
    gen = AnchorGenerator()
    base = gen.generate_base_anchors(16)
    widths = base[:, 2] - base[:, 0]
    assert np.allclose(widths, [16.0, 24.0, 32.0])


def test_anchors_are_sorted_by_level():
    """P2 的框在前、P7 在后；同一行内 anchor 的 x 坐标按步长递增。"""
    all_anchors = AnchorGenerator().generate_all_anchors()
    p2_count = 160 * 160 * 3
    # 换层时框的尺寸变化，x1 不能直接比大小；比较中心点横坐标
    center_x = (all_anchors[:, 0] + all_anchors[:, 2]) / 2
    assert center_x[p2_count] > center_x[0]          # P3 起点在 P2 之后
    first_row = center_x[:160 * 3:3]                 # P2 第一行，每个位置取第一个尺度
    assert np.allclose(np.diff(first_row), 4.0)      # 步长 4
