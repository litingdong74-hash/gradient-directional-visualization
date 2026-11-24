#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梯度与方向导数 · 可旋转的 3D 演示
==================================

数学约定
--------
设二元函数 f(x, y) 在点 P0 = (x0, y0) 处可微。

梯度（gradient）:
    ∇f(P0) = ( ∂f/∂x , ∂f/∂y ) |_{P0}

它指向函数值**上升最快**的方向，模长 |∇f| 就是最大方向导数。

方向导数（directional derivative）沿单位向量 u = (cosφ, sinφ):
    D_u f(P0) = lim_{t→0} [f(P0 + t u) - f(P0)] / t = ∇f(P0) · u
              = |∇f| cosθ          （θ 为 u 与 ∇f 的夹角）

因此：
  * θ = 0°   → 沿梯度方向，方向导数取最大值 |∇f|；
  * θ = 90°  → 沿等值线方向，方向导数为 0；
  * θ > 90°  → 函数值下降。

图上元素
--------
  * 曲面             z = f(x, y)
  * 黑点 + 竖虚线    选定点 P0 及其在 xy 平面上的投影
  * 红色曲线         沿 ∇f 方向的最陡上升路径
  * 橙色粗线         曲面沿 u 的截线 f(P0 + t u)
  * 橙色虚线         该截线在 P0 的切线，斜率就是 D_u f
                    （凸曲面上它在曲面下方，可能被挡住）
  * 右上俯视图       等值线、∇f、u 与夹角 θ（这里没有透视遮挡，最清楚）
  * 右下文字面板     全部数值与公式

交互
----
  鼠标拖动        3D 旋转（matplotlib 自带）
  滚轮            缩放
  ← / →           绕竖直轴旋转
  ↑ / ↓           改变俯仰角
  [ / ]           改变方向角 φ（每次 10°），所有量实时更新
  R               自动旋转开/关
  P               保存当前画面为 PNG
  Q               退出

命令行
------
  python gradient_directional.py                          # 默认：抛物面、P0=(0.6,0.45)、φ=100°
  python gradient_directional.py --preset saddle          # 马鞍面
  python gradient_directional.py --point 0.5 0.5 --angle 45
  python gradient_directional.py --save demo.png --no-show
  python gradient_directional.py --gif demo.gif --no-show

依赖: numpy, matplotlib（Windows 中文需要 Microsoft YaHei / SimHei 之一）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# 配色
# ----------------------------------------------------------------------------
C_GRAD = "#d62728"      # 红：梯度 / 最陡上升
C_U = "#ff8c00"         # 橙：方向 u
C_U_DARK = "#e07000"    # 深橙：切线
C_PT = "#111111"        # 黑：选定点
C_TEXT = "#2b2b2b"
C_HINT = "#777777"


# ----------------------------------------------------------------------------
# 曲面预设：换函数只需要改这里（f 与解析梯度 grad 成对给出）
# ----------------------------------------------------------------------------
@dataclass
class Surface:
    name: str
    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad: Callable[[float, float], tuple[float, float]]
    latex: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    default_point: tuple[float, float]
    default_angle: float          # 单位：度
    elev: float = 34.0            # 默认俯仰角
    azim: float = -63.0           # 默认方位角
    resolution: int = 61


PRESETS: dict[str, Surface] = {
    "bowl": Surface(
        name="椭圆抛物面",
        f=lambda x, y: x * x + y * y,
        grad=lambda x, y: (2.0 * x, 2.0 * y),
        latex=r"$z=f(x,y)=x^{2}+y^{2}$",
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        default_point=(0.6, 0.45),
        default_angle=100.0,
    ),
    "saddle": Surface(
        name="双曲抛物面（马鞍面）",
        f=lambda x, y: x * x - y * y,
        grad=lambda x, y: (2.0 * x, -2.0 * y),
        latex=r"$z=f(x,y)=x^{2}-y^{2}$",
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        default_point=(0.5, 0.4),
        default_angle=-25.0,
        elev=26.0,
        azim=55.0,
    ),
    "hill": Surface(
        name="高斯小山",
        f=lambda x, y: 3.0 * np.exp(-(x * x + y * y) / 2.0) + 0.2 * x,
        grad=lambda x, y: (
            -3.0 * x * np.exp(-(x * x + y * y) / 2.0) + 0.2,
            -3.0 * y * np.exp(-(x * x + y * y) / 2.0),
        ),
        latex=r"$z=f(x,y)=3e^{-(x^{2}+y^{2})/2}+0.2x$",
        xlim=(-2.5, 2.5),
        ylim=(-2.5, 2.5),
        default_point=(0.6, 0.6),
        default_angle=200.0,
        elev=30.0,
        azim=48.0,
    ),
}


def numeric_gradient(f, x: float, y: float, h: float = 1e-5) -> tuple[float, float]:
    """中心差分求梯度：没有解析式时使用。"""
    fx = (f(x + h, y) - f(x - h, y)) / (2.0 * h)
    fy = (f(x, y + h) - f(x, y - h)) / (2.0 * h)
    return float(fx), float(fy)


def setup_fonts() -> None:
    """挑一个系统里存在的中文字体，避免中文变成方框。"""
    from matplotlib import font_manager

    prefer = ["Microsoft YaHei", "SimHei", "PingFang SC",
              "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in prefer if name in installed]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"


class GradientDirectionDemo:
    """一个曲面 + 一个点 + 一个方向，所有图形都挂在这张图上。"""

    def __init__(self, surf: Surface, point: tuple[float, float],
                 angle_deg: float, figsize: tuple[float, float] = (15.4, 8.8),
                 elev: float | None = None, azim: float | None = None):
        self.surf = surf
        self.x0, self.y0 = (float(v) for v in point)
        self.phi = np.deg2rad(float(angle_deg))
        self.phi_deg = float(angle_deg) % 360.0

        # 视角
        self.elev = float(surf.elev if elev is None else elev)
        self.azim = float(surf.azim if azim is None else azim)
        self.auto_rotate = False
        self.timer = None

        # 网格与高度范围
        self.X, self.Y = np.meshgrid(
            np.linspace(*surf.xlim, surf.resolution),
            np.linspace(*surf.ylim, surf.resolution),
        )
        self.Z = np.asarray(surf.f(self.X, self.Y), dtype=float)
        self.zmin = float(np.nanmin(self.Z))
        self.zmax = float(np.nanmax(self.Z))
        zrange = max(self.zmax - self.zmin, 1e-9)
        self.zlo = self.zmin - 0.06 * zrange
        self.zhi = self.zmax + 0.02 * zrange
        self.z0 = float(surf.f(self.x0, self.y0))
        self.gx, self.gy = (float(v) for v in surf.grad(self.x0, self.y0))
        self.gnorm = float(np.hypot(self.gx, self.gy))

        self.width = max(surf.xlim[1] - surf.xlim[0],
                         surf.ylim[1] - surf.ylim[0])
        self.dyn3d: list = []

        self.fig = plt.figure(figsize=figsize, facecolor="white")
        self.ax = self.fig.add_axes([0.000, 0.020, 0.585, 0.945], projection="3d")
        self.axc = self.fig.add_axes([0.615, 0.545, 0.365, 0.425])
        self.axt = self.fig.add_axes([0.615, 0.015, 0.375, 0.505])

        self._build_static()
        self._draw()
        self._connect()
        self._print_summary()

    # ------------------------------------------------------------------
    # 静态图元：曲面、点、坐标轴
    # ------------------------------------------------------------------
    def _build_static(self) -> None:
        ax = self.ax
        ax.set_proj_type("persp", focal_length=0.30)
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_box_aspect((1.0, 1.0, 0.72))

        # 曲面
        ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap="viridis", linewidth=0, antialiased=True,
            rstride=1, cstride=1, alpha=0.96, shade=True,
        )

        # 选定点、名字、到 xy 平面的投影虚线
        ax.scatter([self.x0], [self.y0], [self.z0], s=95, c=C_PT,
                   edgecolors="white", linewidths=1.5, depthshade=False, zorder=12)
        ax.text(self.x0 - 0.02, self.y0 + 0.05, self.z0 + 0.26, r"$P_0$",
                fontsize=12.5, color=C_PT, zorder=12, ha="right", va="bottom")
        ax.plot([self.x0, self.x0], [self.y0, self.y0], [self.zlo, self.z0],
                ls=(0, (2, 3)), color="0.45", lw=1.1, zorder=4)
        ax.scatter([self.x0], [self.y0], [self.zlo], s=18, facecolors="none",
                   edgecolors="0.45", linewidths=1.0, depthshade=False)

        # 坐标轴
        ax.set_xlim(*self.surf.xlim)
        ax.set_ylim(*self.surf.ylim)
        ax.set_zlim(self.zlo, self.zhi)
        ax.set_xlabel("x", fontsize=11, labelpad=2)
        ax.set_ylabel("y", fontsize=11, labelpad=2)
        ax.set_zlabel("z = f(x, y)", fontsize=10.5, labelpad=4)
        ax.tick_params(labelsize=8, pad=1)
        ax.set_title(f"{self.surf.name}：{self.surf.latex}", fontsize=13, pad=6)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.975, 0.98, 0.988, 1.0))
            axis.pane.set_edgecolor((0.84, 0.84, 0.84, 1.0))
        ax.grid(True, color="0.9", linewidth=0.6)

        self.fig.text(
            0.004, 0.004,
            "鼠标拖动旋转 · 滚轮缩放 · ←/→ 旋转 · ↑/↓ 俯仰 · [ ] 改变方向 φ "
            "· R 自动旋转 · P 存图 · Q 退出",
            fontsize=8.2, color=C_HINT, ha="left", va="bottom",
        )

    # ------------------------------------------------------------------
    # 动态图元：随 φ 变化
    # ------------------------------------------------------------------
    def _clear_dyn(self) -> None:
        for artist in self.dyn3d:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self.dyn3d.clear()

    def _reach(self, ux: float, uy: float) -> float:
        """从 P0 沿 u 走到边界时的正向参数 t。"""
        cands: list[float] = []
        for p, d, (lo, hi) in ((self.x0, ux, self.surf.xlim),
                               (self.y0, uy, self.surf.ylim)):
            if abs(d) > 1e-12:
                for t in ((lo - p) / d, (hi - p) / d):
                    if t > 1e-9:
                        cands.append(t)
        return min(cands) if cands else 1.0

    def _draw(self) -> None:
        self._clear_dyn()

        x0, y0, z0 = self.x0, self.y0, self.z0
        gx, gy, gnorm = self.gx, self.gy, self.gnorm
        ux, uy = float(np.cos(self.phi)), float(np.sin(self.phi))
        du = gx * ux + gy * uy

        # --- 红色：沿 ∇f 的最陡上升曲线 ---------------------------------
        if gnorm > 1e-9:
            ghx, ghy = gx / gnorm, gy / gnorm
            tg = np.linspace(0.0, min(0.95, 0.9 * self._reach(ghx, ghy)), 80)
            gx_path = x0 + tg * ghx
            gy_path = y0 + tg * ghy
            gz_path = np.asarray(self.surf.f(gx_path, gy_path), dtype=float)
            grad_path = self.ax.plot(gx_path, gy_path, gz_path, color=C_GRAD,
                                     lw=2.4, alpha=0.95, zorder=11)[0]
            self.dyn3d.append(grad_path)
            self.dyn3d.append(self.ax.text(
                gx_path[-1] + 0.06, gy_path[-1] + 0.04, gz_path[-1] + 0.18,
                r"$\nabla f$", color=C_GRAD, fontsize=12.5,
                ha="left", va="bottom", zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
            ))

        # --- 橙色：沿 u 的截线与切线 --------------------------------------
        tmax = self._reach(ux, uy)
        tt = min(0.90, 0.72 * tmax)
        t = np.linspace(-tt, tt, 141)
        px, py = x0 + t * ux, y0 + t * uy
        pz = np.asarray(self.surf.f(px, py), dtype=float)
        trace = self.ax.plot(px, py, pz, color=C_U, lw=3.4,
                             solid_capstyle="round", zorder=11)[0]
        self.dyn3d.append(trace)
        self.dyn3d.append(self.ax.text(
            px[-1] + 0.05, py[-1] + 0.05, pz[-1] + 0.24, r"$u$",
            color=C_U, fontsize=13, ha="left", va="bottom", zorder=12,
        ))

        # --- 切线：它在 P0 的斜率就是 D_u f -------------------------------
        s = 0.85 * min(1.0, tmax)
        t_end = min(0.95 * s, 0.9 * tmax)
        tan_line = self.ax.plot(
            [x0 - 0.45 * s * ux, x0 + t_end * ux],
            [y0 - 0.45 * s * uy, y0 + t_end * uy],
            [z0 - 0.45 * s * du, z0 + t_end * du],
            ls=(0, (5, 3)), color=C_U_DARK, lw=1.9, zorder=11,
        )[0]
        self.dyn3d.append(tan_line)

        # --- 2D 俯视图与文字面板 ------------------------------------------
        self._draw_contour(ux, uy, du)
        self._draw_info(ux, uy, du)

    def _draw_contour(self, ux: float, uy: float, du: float) -> None:
        ax = self.axc
        ax.clear()
        ax.contourf(self.X, self.Y, self.Z, levels=24, cmap="viridis", alpha=0.92)
        ax.contour(self.X, self.Y, self.Z, levels=10, colors="white",
                   linewidths=0.55, alpha=0.55)

        x0, y0 = self.x0, self.y0
        ax.plot([x0], [y0], "o", ms=8, mfc=C_PT, mec="white", mew=1.4, zorder=12)

        # 梯度箭头（太长时等比缩短显示，数值不缩）
        g_disp = 1.0
        if self.gnorm > 1e-9:
            g_disp = min(1.0, 0.42 * self.width / self.gnorm)
            gsx, gsy = self.gx * g_disp, self.gy * g_disp
            ax.annotate("", xy=(x0 + gsx, y0 + gsy), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=C_GRAD, lw=2.5,
                                        mutation_scale=18, shrinkA=0, shrinkB=0),
                        zorder=11)
            ax.text(x0 + gsx * 1.12, y0 + gsy * 1.12, r"$\nabla f$",
                    color=C_GRAD, fontsize=12.5, ha="center", va="center")

            a_g = np.degrees(np.arctan2(self.gy, self.gx))
            a_u = np.degrees(np.arctan2(uy, ux))
            d = (a_u - a_g + 180.0) % 360.0 - 180.0
            r = 0.24 * self.width
            aa = np.deg2rad(np.linspace(a_g, a_g + d, 60))
            ax.plot(x0 + r * np.cos(aa), y0 + r * np.sin(aa),
                    color="0.35", lw=1.3, zorder=10)
            am = np.deg2rad(a_g + d / 2.0)
            ax.text(x0 + r * 1.30 * np.cos(am), y0 + r * 1.30 * np.sin(am),
                    rf"$\theta={abs(d):.0f}^\circ$", color="0.20", fontsize=10.5,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55))

        # 方向 u
        lu = 0.30 * self.width
        ax.annotate("", xy=(x0 + ux * lu, y0 + uy * lu), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C_U, lw=2.5,
                                    mutation_scale=18, shrinkA=0, shrinkB=0),
                    zorder=11)
        ax.text(x0 + ux * lu * 1.10, y0 + uy * lu * 1.10, r"$u$",
                color=C_U, fontsize=12.5, ha="center", va="center")

        # 沿 u 的参考虚线
        tmax = self._reach(ux, uy)
        ax.plot([x0 - 0.85 * tmax * ux, x0 + 0.85 * tmax * ux],
                [y0 - 0.85 * tmax * uy, y0 + 0.85 * tmax * uy],
                ls=(0, (4, 3)), color=C_U_DARK, lw=1.0, alpha=0.8, zorder=9)

        # 沿 ∇f 的最陡上升方向（与 3D 红线对应）
        if self.gnorm > 1e-9:
            tg = np.linspace(0.0, 0.85, 2)
            ax.plot(x0 + tg * self.gx / self.gnorm, y0 + tg * self.gy / self.gnorm,
                    color=C_GRAD, lw=2.0, alpha=0.85, zorder=10)

        ax.set_xlim(*self.surf.xlim)
        ax.set_ylim(*self.surf.ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7.5)
        ax.set_title("俯视图：等值线、梯度 $\\nabla f$ 与方向 $u$",
                     fontsize=11, pad=6)
        for spine in ax.spines.values():
            spine.set_color("0.75")

    def _draw_info(self, ux: float, uy: float, du: float) -> None:
        a = self.axt
        a.clear()
        a.axis("off")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)

        gnorm = self.gnorm
        g_disp = min(1.0, 0.42 * self.width / gnorm) if gnorm > 1e-9 else 1.0
        items: list[tuple[str, str, float, str, float]] = []
        items.append(("梯度与方向导数", "#1a1a1a", 15.5, "bold", 0.098))
        items.append((self.surf.latex, C_TEXT, 12.5, "normal", 0.070))
        items.append((f"$P_0=({self.x0:.2f},\\ {self.y0:.2f})$，"
                      f"$f(P_0)={self.z0:.2f}$", "#333333", 10.6, "normal", 0.056))
        items.append((f"$\\nabla f=({self.gx:.2f},\\ {self.gy:.2f})$，"
                      f"$|\\nabla f|={gnorm:.2f}$", C_GRAD, 10.6, "normal", 0.056))
        items.append((f"$u=(\\cos\\varphi,\\ \\sin\\varphi)$，"
                      f"$\\varphi={self.phi_deg:.0f}^\\circ$", C_U, 10.6, "normal", 0.056))
        items.append(("", C_TEXT, 9.0, "normal", 0.016))

        if gnorm > 1e-9:
            cos_t = float(np.clip(du / gnorm, -1.0, 1.0))
            theta = float(np.degrees(np.arccos(cos_t)))
            phi_star = float(np.degrees(np.arctan2(self.gy, self.gx))) % 360.0
            items.append((f"$D_u f=\\nabla f\\cdot u={du:+.2f}$",
                          C_U_DARK, 13.0, "bold", 0.066))
            items.append((f"$\\quad=|\\nabla f|\\cos\\theta="
                          f"{gnorm:.2f}\\times{cos_t:+.2f}={du:+.2f}$",
                          "#333333", 10.4, "normal", 0.052))
            items.append((f"$\\theta={theta:.1f}^\\circ$ 是 $u$ 与 $\\nabla f$ 的夹角",
                          "#333333", 10.4, "normal", 0.052))
            items.append((f"最陡上升方向 $\\varphi^*={phi_star:.1f}^\\circ$，"
                          f"最大方向导数 ${gnorm:.2f}$", "#333333", 10.4, "normal", 0.060))
            items.append((r"$\theta<90^\circ$ 上升 · $\theta=90^\circ$ 不变 · "
                          r"$\theta>90^\circ$ 下降", "#777777", 8.8, "normal", 0.046))
        else:
            items.append(("$\\nabla f=0$：临界点，任何方向的方向导数都是 0",
                          C_U_DARK, 11.0, "bold", 0.070))

        items.append(("图例", "#555555", 9.8, "bold", 0.046))
        items.append(("● 选定点 $P_0$；灰线：俯视图中的等值线",
                      "#444444", 9.0, "normal", 0.044))
        items.append(("红：$\\nabla f$ 方向，曲面上最陡的上升路径",
                      C_GRAD, 9.0, "normal", 0.044))
        items.append(("橙：$u$ 方向的曲面截线（斜率 = $D_u f$）",
                      C_U, 9.0, "normal", 0.044))
        items.append(("虚线：截线在 $P_0$ 处的切线",
                      C_U_DARK, 9.0, "normal", 0.044))

        if g_disp < 0.999:
            items.append(("注：俯视图中 $\\nabla f$ 箭头按显示需要等比缩短",
                          "#777777", 8.6, "normal", 0.040))

        y = 0.985
        for text, color, size, weight, gap in items:
            if text:
                a.text(0.0, y, text, transform=a.transAxes, va="top", ha="left",
                       fontsize=size, color=color, fontweight=weight)
            y -= gap

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _redraw_view(self) -> None:
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("right", "d"):
            self.azim += 8.0
        elif key in ("left", "a"):
            self.azim -= 8.0
        elif key in ("up", "w"):
            self.elev = min(85.0, self.elev + 5.0)
        elif key in ("down", "s"):
            self.elev = max(5.0, self.elev - 5.0)
        elif key == "bracketright":
            self.phi = (self.phi + np.deg2rad(10.0)) % (2.0 * np.pi)
            self.phi_deg = (self.phi_deg + 10.0) % 360.0
            self._draw()
        elif key == "bracketleft":
            self.phi = (self.phi - np.deg2rad(10.0)) % (2.0 * np.pi)
            self.phi_deg = (self.phi_deg - 10.0) % 360.0
            self._draw()
        elif key == "r":
            self._toggle_autorotate()
            return
        elif key == "p":
            self.save("demo_snapshot.png")
            return
        elif key in ("q", "escape"):
            plt.close(self.fig)
            return
        else:
            return
        self._redraw_view()

    def _toggle_autorotate(self) -> None:
        if not hasattr(self.fig.canvas, "new_timer"):
            print("[info] 当前后端不支持自动旋转，请用鼠标拖动或 ←/→ 键")
            return
        self.auto_rotate = not self.auto_rotate
        if self.auto_rotate:
            if self.timer is None:
                self.timer = self.fig.canvas.new_timer(interval=40)
                self.timer.add_callback(self._tick)
            self.timer.start()
        elif self.timer is not None:
            self.timer.stop()

    def _tick(self) -> None:
        self.azim = (self.azim + 0.9) % 360.0
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.fig.canvas.draw_idle()

    def _on_close(self, _event) -> None:
        if self.timer is not None:
            self.timer.stop()

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def save(self, path: str, dpi: int = 165) -> None:
        self.fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        print(f"[saved] {path}")

    def _print_summary(self) -> None:
        du = self.gx * np.cos(self.phi) + self.gy * np.sin(self.phi)
        print("=" * 56)
        print(f"曲面: {self.surf.name}   {self.surf.latex}")
        print(f"P0 = ({self.x0:.3f}, {self.y0:.3f}), f(P0) = {self.z0:.4f}")
        print(f"grad f = ({self.gx:.3f}, {self.gy:.3f}), |grad f| = {self.gnorm:.4f}")
        print(f"phi = {self.phi_deg:.1f} deg, u = ({np.cos(self.phi):+.3f}, {np.sin(self.phi):+.3f})")
        print(f"D_u f = {du:+.4f}")
        if self.gnorm > 1e-12:
            theta = np.degrees(np.arccos(np.clip(du / self.gnorm, -1.0, 1.0)))
            phi_star = np.degrees(np.arctan2(self.gy, self.gx)) % 360.0
            print(f"theta(u, grad f) = {theta:.2f} deg")
            print(f"steepest ascent: phi* = {phi_star:.2f} deg, max D_u f = {self.gnorm:.4f}")
        print("=" * 56)


# ----------------------------------------------------------------------------
# 存 GIF
# ----------------------------------------------------------------------------
def export_gif(demo: GradientDirectionDemo, path: str,
               frames: int = 40, fps: int = 12, width: int = 900) -> None:
    """左右摆动 ±60° 存成 GIF（不需要 ffmpeg），全程能看到选定点。"""
    from PIL import Image

    if demo.auto_rotate and demo.timer is not None:
        demo.timer.stop()
    azim0, elev0 = demo.azim, demo.elev
    shots = []
    try:
        for i in range(frames):
            s = np.sin(2.0 * np.pi * i / frames)
            demo.azim = azim0 + 60.0 * s
            demo.elev = min(80.0, elev0 + 8.0 + 3.0 * np.sin(4.0 * np.pi * i / frames))
            demo.ax.view_init(elev=demo.elev, azim=demo.azim)
            demo.fig.canvas.draw()
            buf = np.asarray(demo.fig.canvas.buffer_rgba())
            img = Image.fromarray(buf).convert("RGB")
            ratio = width / img.width
            img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
            shots.append(img.convert("P", palette=Image.ADAPTIVE, colors=200))
        shots[0].save(path, save_all=True, append_images=shots[1:],
                      duration=int(1000 / fps), loop=0, optimize=True)
        print(f"[saved] {path}  ({frames} frames, {fps} fps)")
    finally:
        demo.azim, demo.elev = azim0, elev0
        demo.ax.view_init(elev=elev0, azim=azim0)


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="二元函数的梯度与方向导数 3D 演示（可旋转）")
    p.add_argument("--preset", choices=sorted(PRESETS), default="bowl",
                   help="曲面预设：bowl（抛物面）/ saddle（马鞍面）/ hill（高斯小山）")
    p.add_argument("--point", nargs=2, type=float, metavar=("X0", "Y0"),
                   help="选定点坐标，默认用预设里的点")
    p.add_argument("--angle", type=float, help="方向角 φ（度），默认用预设值")
    p.add_argument("--save", metavar="PNG", help="保存一张静态图后继续/退出")
    p.add_argument("--gif", metavar="GIF", help="保存摆动旋转 GIF")
    p.add_argument("--elev", type=float, default=None, help="初始俯仰角（度）")
    p.add_argument("--azim", type=float, default=None, help="初始方位角（度）")
    p.add_argument("--no-show", action="store_true", help="不弹出交互窗口（用于批量出图）")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_fonts()

    surf = PRESETS[args.preset]
    point = tuple(args.point) if args.point else surf.default_point
    angle = args.angle if args.angle is not None else surf.default_angle

    if not (surf.xlim[0] < point[0] < surf.xlim[1] and
            surf.ylim[0] < point[1] < surf.ylim[1]):
        raise SystemExit(f"点 {point} 不在区域 {surf.xlim}×{surf.ylim} 内")

    if args.no_show:
        plt.switch_backend("Agg")

    demo = GradientDirectionDemo(surf, point, angle,
                                 elev=args.elev, azim=args.azim)

    if args.save:
        demo.save(args.save)
    if args.gif:
        export_gif(demo, args.gif)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
