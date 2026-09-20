# -*- coding: utf-8 -*-
"""智枢星品牌 PNG 资产渲染:从矢量设计参数直接绘制 PWA / favicon 图标。

用法:  python scripts/render_brand_assets.py
输出:  web/mobile/icon-*.png、apple-touch-icon.png,以及 src/.../static/assets/favicon-*.png
说明:  4x 超采样绘制后 LANCZOS 缩小,保证边缘平滑;与 logo-mark.svg 同一几何定义。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / "web" / "mobile"
ASSETS = ROOT / "src" / "zhishuxing" / "webapp" / "static" / "assets"

# ---- 品牌色(与 logo-mark.svg 一致) ----
C_INDIGO = (99, 102, 241)   # #6366f1
C_VIOLET_MID = (124, 77, 240)  # #7c4df0
C_VIOLET = (139, 92, 246)   # #8b5cf6
C_CYAN = (34, 211, 238)     # #22d3ee
C_STAR_TOP = (255, 255, 255)
C_STAR_BOT = (165, 243, 252)  # #a5f3fc

SS = 4  # 超采样倍数


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def tile_gradient(size: int) -> Image.Image:
    """135° 三停渐变底板(靛蓝→紫→亮紫),叠加左上青色光泽。"""
    n = size * SS
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    t = np.clip((xs + ys) / (2 * n), 0, 1)
    img = np.zeros((n, n, 3), dtype=np.float32)
    m1, m2 = 0.55, 0.72
    lo = t < m1
    mid = (t >= m1) & (t < m2)
    hi = t >= m2
    for c in range(3):
        img[..., c][lo] = C_INDIGO[c] + (C_VIOLET_MID[c] - C_INDIGO[c]) * (t[lo] / m1)
        img[..., c][mid] = C_VIOLET_MID[c] + (C_VIOLET[c] - C_VIOLET_MID[c]) * ((t[mid] - m1) / (m2 - m1))
        img[..., c][hi] = C_VIOLET[c]
    # 左上青色光泽:径向衰减
    cx, cy, r = 0.2 * n, 0.1 * n, 1.0 * n
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / r
    alpha = np.clip(1 - d, 0, 1) ** 1.6 * 0.5
    for c in range(3):
        img[..., c] = img[..., c] * (1 - alpha) + C_CYAN[c] * alpha
    return Image.fromarray(img.astype(np.uint8), "RGB")


def star_polygon(cx: float, cy: float, r: float, samples_per_curve: int = 48):
    """中心四角星(凹边),与 logo-mark.svg 的三次贝塞尔外形一致,按半径 r 缩放。"""
    # 原始形状定义在 64 视箱中,中心 (32,32),北尖点 (32,19.5) → 相对半径 12.5
    k = r / 12.5
    P = lambda x, y: (cx + (x - 32) * k, cy + (y - 32) * k)
    segs = [
        ((32, 19.5), (33.6, 27.6), (36.4, 30.4), (44.5, 32)),
        ((44.5, 32), (36.4, 33.6), (33.6, 36.4), (32, 44.5)),
        ((32, 44.5), (30.4, 36.4), (27.6, 33.6), (19.5, 32)),
        ((19.5, 32), (27.6, 30.4), (30.4, 27.6), (32, 19.5)),
    ]
    pts = []
    for p0, p1, p2, p3 in segs:
        p0, p1, p2, p3 = P(*p0), P(*p1), P(*p2), P(*p3)
        for i in range(samples_per_curve):
            t = i / samples_per_curve
            mt = 1 - t
            x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
            y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
            pts.append((x, y))
    return pts


def star_gradient_mask(n: int, box) -> Image.Image:
    """星体填充:垂直渐变 白→#a5f3fc,返回已裁剪到 box 的 RGB 图。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h = max(y1 - y0, 1)
    grad = np.zeros((h, max(x1 - x0, 1), 3), dtype=np.float32)
    for yy in range(h):
        t = yy / h
        grad[yy, :] = _lerp(C_STAR_TOP, C_STAR_BOT, t)
    return Image.fromarray(grad.astype(np.uint8), "RGB")


def draw_mark(base: Image.Image, scale: float = 1.0, center=None):
    """在 base(尺寸 size*SS 的 RGBA)上绘制轨道 + 星体;scale 相对默认布局。"""
    n = base.size[0]
    cx, cy = center or (n / 2, n / 2)
    k = (n / 64.0) * scale  # 64 视箱 → 像素

    def orbit(angle_deg, alpha, dot_specs):
        rx, ry = 21 * k, 12 * k
        layer = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        d.ellipse(bbox, outline=(255, 255, 255, int(255 * alpha)), width=max(int(2.4 * k), SS))
        for t_deg, dot_r, color, dot_alpha in dot_specs:
            t = math.radians(t_deg)
            px = cx + rx * math.cos(t)
            py = cy + ry * math.sin(t)
            dr = max(dot_r * k, SS)
            d.ellipse([px - dr, py - dr, px + dr, py + dr], fill=color + (int(255 * dot_alpha),))
        return layer.rotate(angle_deg, resample=Image.BICUBIC, center=(cx, cy))

    base.alpha_composite(orbit(-26, 0.8, [(0, 3.0, (255, 255, 255), 1.0), (180, 2.1, (103, 232, 249), 1.0)]))
    base.alpha_composite(orbit(26, 0.38, [(22.5, 2.1, (255, 255, 255), 0.85)]))

    # 星体:发光 + 渐变填充
    pts = star_polygon(cx, cy, 12.5 * k)
    glow = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon(pts, fill=(165, 243, 252, 110))
    glow = glow.filter(ImageFilter.GaussianBlur(2.2 * k))
    base.alpha_composite(glow)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    box = (min(xs), min(ys), max(xs), max(ys))
    grad = star_gradient_mask(n, box)
    star_mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(star_mask).polygon(pts, fill=255)
    star_layer = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    star_layer.paste(grad, (int(box[0]), int(box[1])))
    base.paste(star_layer, (0, 0), star_mask)

    # 右上小青点
    d = ImageDraw.Draw(base)
    px, py = cx + 12.8 * k, cy - 12.8 * k
    dr = max(1.4 * k, SS)
    d.ellipse([px - dr, py - dr, px + dr, py + dr], fill=(103, 232, 249, 255))


def render_icon(size: int, maskable: bool = False, out: Path | None = None) -> Image.Image:
    n = size * SS
    grad = tile_gradient(size).convert("RGBA")
    if maskable:
        full = grad  # 满铺方形,无圆角(启动器自行裁切)
        mark = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        draw_mark(mark, scale=0.62)
        full.alpha_composite(mark)
    else:
        radius = int(0.234 * n)
        mask = Image.new("L", (n, n), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=radius, fill=255)
        full = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        full.paste(grad, (0, 0), mask)
        draw_mark(full, scale=1.0)
        # 内描边高光(与 SVG 一致)
        ImageDraw.Draw(full).rounded_rectangle(
            [int(1.2 * SS), int(1.2 * SS), n - int(1.2 * SS) - 1, n - int(1.2 * SS) - 1],
            radius=radius - int(0.8 * SS), outline=(255, 255, 255, 56), width=SS)
    img = full.resize((size, size), Image.LANCZOS)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"  ✓ {out.relative_to(ROOT)}")
    return img


def main():
    print("渲染智枢星品牌 PNG 图标…")
    render_icon(512, False, MOBILE / "icon-512.png")
    render_icon(192, False, MOBILE / "icon-192.png")
    render_icon(180, False, MOBILE / "apple-touch-icon.png")
    render_icon(512, True, MOBILE / "icon-maskable-512.png")
    render_icon(64, False, MOBILE / "splash-logo.png")
    render_icon(48, False, ASSETS / "favicon-48.png")
    render_icon(32, False, ASSETS / "favicon-32.png")
    print("完成。")


if __name__ == "__main__":
    main()
