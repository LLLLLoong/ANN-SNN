"""
SNN 可视化绘图工具 -- SNN Visualization for Publication
======================================================
基于 spikingjelly.visualizing 模块的增强版论文绘图工具,
专门为脉冲神经网络 (SNN) / ANN-SNN 转换研究设计。

功能列表:
  1. 膜电位热力图      (Membrane Potential Heatmap)
  2. 脉冲发放光栅图    (Spike Raster Plot + Firing Rate)
  3. 单神经元动力学    (Single Neuron V-S Dynamics)
  4. 3D 脉冲柱状图     (3D Spike Bar Chart)
  5. 脉冲特征图        (Spiking Feature Map)
  6. 替代梯度函数对比  (Surrogate Gradient Comparison)
  7. 逐层发放率分析    (Layer-wise Firing Rate)
  8. QCFS 量化阶梯可视化 (QCFS Quantization Step Function)
  9. IF / LIF 神经元模型对比 (IF vs LIF Comparison)
  10. ANN-SNN 精度差距分析 (Accuracy Gap Analysis)

使用方法:
    python plot_snn_figures.py                    # 生成所有示例图
    python plot_snn_figures.py --only spike       # 仅生成脉冲光栅图
    python plot_snn_figures.py --only surrogate   # 仅生成替代梯度图

所有图片保存至 ./figures_snn/ 目录, PDF + PNG (300 DPI)。
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict

# ============================================================================
#  全局样式 (Global Style - 学术论文级)
# ============================================================================

PALETTE = {
    'blue':       '#2E86AB',
    'red':        '#E8505B',
    'green':      '#1B998B',
    'orange':     '#F39C12',
    'purple':     '#8E44AD',
    'teal':       '#16A085',
    'pink':       '#E91E63',
    'dark_blue':  '#1A5276',
    'grey':       '#7F8C8D',
    'gold':       '#D4AC0D',
    'text':       '#2C3E50',
    'grid':       '#E0E0E0',
}

COLOR_LIST = [
    PALETTE['blue'], PALETTE['red'], PALETTE['green'],
    PALETTE['orange'], PALETTE['purple'], PALETTE['teal'],
    PALETTE['pink'], PALETTE['dark_blue'],
]

MARKER_LIST = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']


def setup_style():
    """论文级 matplotlib 全局样式。"""
    plt.rcParams.update({
        'font.family':          'serif',
        'font.serif':           ['Times New Roman', 'DejaVu Serif', 'SimSun'],
        'font.size':            11,
        'axes.titlesize':       13,
        'axes.labelsize':       12,
        'xtick.labelsize':      10,
        'ytick.labelsize':      10,
        'legend.fontsize':      10,
        'lines.linewidth':      1.8,
        'lines.markersize':     6,
        'axes.linewidth':       1.0,
        'axes.edgecolor':       PALETTE['text'],
        'axes.labelcolor':      PALETTE['text'],
        'axes.grid':            True,
        'grid.alpha':           0.3,
        'grid.linewidth':       0.6,
        'grid.color':           PALETTE['grid'],
        'xtick.direction':      'in',
        'ytick.direction':      'in',
        'xtick.major.width':    0.8,
        'ytick.major.width':    0.8,
        'legend.framealpha':    0.9,
        'legend.edgecolor':     PALETTE['grid'],
        'legend.fancybox':      True,
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
        'savefig.pad_inches':   0.05,
        'mathtext.fontset':     'stix',
    })


def save_fig(fig, name, output_dir='./figures_snn'):
    """同时保存 PDF 和 PNG。"""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(output_dir, f'{name}.png'), format='png')
    print(f'  [OK] saved: {output_dir}/{name}.pdf / .png')


# ============================================================================
#  自定义颜色映射 (Custom Colormaps)
# ============================================================================

# 膜电位专用: 蓝(低) -> 白(阈值附近) -> 红(高/脉冲)
VOLTAGE_CMAP = LinearSegmentedColormap.from_list(
    'voltage', ['#1A5276', '#2E86AB', '#85C1E9', '#FDFEFE', '#F5B7B1', '#E8505B', '#922B21']
)

# 脉冲专用: 白(0) -> 深蓝(1)
SPIKE_CMAP = LinearSegmentedColormap.from_list(
    'spike', ['#FAFAFA', '#2E86AB']
)


# ============================================================================
#  图 1: 膜电位热力图 (Membrane Potential Heatmap)
# ============================================================================

def plot_membrane_heatmap(voltages, title='Membrane Potential of Neurons',
                           xlabel='Time Step', ylabel='Neuron Index',
                           colorbar_label='$V_{mem}$', v_threshold=1.0):
    """
    增强版膜电位热力图, 带阈值参考线。

    参数:
        voltages:  np.ndarray, shape=[T, N], T个时间步, N个神经元
        v_threshold: 阈值电压, 在 colorbar 上标注
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

    im = ax.imshow(voltages.T, aspect='auto', cmap=VOLTAGE_CMAP,
                   extent=[-0.5, voltages.shape[0] - 0.5,
                           voltages.shape[1] - 0.5, -0.5])

    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.set_ylabel(colorbar_label, rotation=90, va='top', fontsize=11)
    # 标注阈值线
    if v_threshold is not None:
        cbar.ax.axhline(y=v_threshold, color=PALETTE['red'], linewidth=1.5, linestyle='--')
        cbar.ax.text(1.5, v_threshold, '$V_{th}$', va='center', fontsize=9,
                     color=PALETTE['red'], fontweight='bold')

    fig.tight_layout()
    return fig


# ============================================================================
#  图 2: 脉冲发放光栅图 (Spike Raster Plot)
# ============================================================================

def plot_spike_raster(spikes, title='Spike Raster Plot',
                      xlabel='Time Step', ylabel='Neuron Index',
                      plot_firing_rate=True, firing_rate_title='Rate'):
    """
    增强版脉冲发放光栅图, 右侧附加发放率热力图。

    参数:
        spikes: np.ndarray, shape=[T, N], 0/1 脉冲矩阵
    """
    spikes_T = spikes.T  # [N, T]
    N, T = spikes_T.shape

    if plot_firing_rate:
        fig = plt.figure(figsize=(10, 5), dpi=200)
        gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.05)
        ax_spikes = fig.add_subplot(gs[0, 0:5])
        ax_rate = fig.add_subplot(gs[0, 5])
    else:
        fig, ax_spikes = plt.subplots(figsize=(10, 5), dpi=200)

    ax_spikes.set_title(title, fontweight='bold', pad=10)
    ax_spikes.set_xlabel(xlabel)
    ax_spikes.set_ylabel(ylabel)

    # 使用 eventplot — 效果更美观
    t = np.arange(T)
    for i in range(N):
        spike_times = t[spikes_T[i] == 1]
        color = COLOR_LIST[i % len(COLOR_LIST)]
        ax_spikes.eventplot(spike_times, lineoffsets=i, linewidths=1.2,
                            linelengths=0.8, colors=color)

    ax_spikes.set_xlim(-0.5, T - 0.5)
    ax_spikes.set_ylim(-0.5, N - 0.5)
    ax_spikes.invert_yaxis()
    ax_spikes.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_spikes.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if plot_firing_rate:
        firing_rate = np.mean(spikes_T, axis=1, keepdims=True)  # [N, 1]
        ax_rate.imshow(firing_rate, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(0.01, firing_rate.max()))
        for i in range(N):
            color = 'white' if firing_rate[i, 0] > 0.5 * firing_rate.max() else PALETTE['text']
            ax_rate.text(0, i, f'{firing_rate[i, 0]:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')
        ax_rate.get_xaxis().set_visible(False)
        ax_rate.set_title(firing_rate_title, fontsize=10, fontweight='bold')
        ax_rate.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_rate.set_yticklabels([])

    fig.tight_layout()
    return fig


# ============================================================================
#  图 3: 单神经元电压-脉冲动力学 (Single Neuron V-S Dynamics)
# ============================================================================

def plot_neuron_dynamics(v, s, v_threshold=1.0, v_reset=0.0,
                         title='Membrane Potential and Spikes of a Neuron',
                         input_current=None):
    """
    增强版单神经元电压与脉冲图, 可选显示输入电流。

    参数:
        v: np.ndarray, shape=[T], 膜电位
        s: np.ndarray, shape=[T], 脉冲 (0 or 1)
        input_current: np.ndarray or None, shape=[T], 输入电流
    """
    T = len(v)
    t = np.arange(T)

    n_rows = 3 if input_current is not None else 2
    fig = plt.figure(figsize=(10, 2.2 * n_rows), dpi=200)
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=[3, 1] if n_rows == 2 else [2, 2, 1],
                           hspace=0.15)

    row = 0
    # (可选) 输入电流
    if input_current is not None:
        ax_input = fig.add_subplot(gs[row])
        ax_input.fill_between(t, input_current, alpha=0.3, color=PALETTE['green'])
        ax_input.plot(t, input_current, color=PALETTE['green'], linewidth=1.2)
        ax_input.set_ylabel('Input $I(t)$')
        ax_input.set_xlim(-0.5, T - 0.5)
        ax_input.set_title(title, fontweight='bold', pad=8)
        ax_input.set_xticklabels([])
        row += 1

    # 膜电位
    ax_v = fig.add_subplot(gs[row])
    ax_v.plot(t, v, color=PALETTE['blue'], linewidth=1.8, label='$V_{mem}(t)$')
    ax_v.axhline(v_threshold, linestyle='--', color=PALETTE['red'], linewidth=1.2,
                 label=f'$V_{{th}}={v_threshold}$')
    if v_reset is not None:
        ax_v.axhline(v_reset, linestyle='-.', color=PALETTE['green'], linewidth=1.0,
                     alpha=0.7, label=f'$V_{{reset}}={v_reset}$')

    # 在脉冲发放时刻加上标记
    spike_times = t[s == 1]
    spike_v = v[s == 1]
    ax_v.scatter(spike_times, spike_v, color=PALETTE['red'], s=40, zorder=5,
                 marker='v', label='Spike')

    ax_v.set_ylabel('Membrane Potential')
    ax_v.set_xlim(-0.5, T - 0.5)
    ax_v.legend(loc='upper right', fontsize=8, ncol=2)
    if input_current is None:
        ax_v.set_title(title, fontweight='bold', pad=8)
    ax_v.set_xticklabels([])
    row += 1

    # 脉冲
    ax_s = fig.add_subplot(gs[row])
    spike_mask = (s == 1)
    ax_s.eventplot(t[spike_mask], lineoffsets=0, linewidths=2.0,
                   linelengths=0.8, colors=PALETTE['red'])
    ax_s.set_xlim(-0.5, T - 0.5)
    ax_s.set_xlabel('Time Step $t$')
    ax_s.set_ylabel('Spike')
    ax_s.set_yticks([])
    ax_s.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


# ============================================================================
#  图 4: 3D 脉冲柱状图 (3D Spike Bar)
# ============================================================================

def plot_3d_spike_bar(array, title='Firing Rate over Time',
                      xlabel='Neuron Index', ylabel='Time Step', zlabel='Rate'):
    """
    增强版 3D 柱状图, 展示多个神经元在不同时间步的发放率。
    参数:
        array: np.ndarray, shape=[T, N]
    """
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    array_T = array.T  # [N, T]
    N, T_len = array_T.shape
    xs = np.arange(T_len)

    cmap = plt.get_cmap('coolwarm')
    for i in range(N):
        color = cmap(i / max(1, N - 1))
        ax.bar(xs, array_T[i], zs=i, zdir='x', color=color, alpha=0.85, width=0.8)

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_zlabel(zlabel, labelpad=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.view_init(elev=25, azim=-55)

    fig.tight_layout()
    return fig


# ============================================================================
#  图 5: 2D 脉冲特征图 (Spiking Feature Map)
# ============================================================================

def plot_spiking_feature_map(spikes, nrows, ncols, space=2,
                              title='Spiking Feature Maps', cmap='gray_r'):
    """
    增强版 2D 脉冲特征图拼接展示。

    参数:
        spikes: np.ndarray, shape=[C, H, W], C 个通道的脉冲特征图
        nrows, ncols: 排列行列数 (nrows * ncols == C)
    """
    C, H, W = spikes.shape
    assert nrows * ncols == C, f'nrows*ncols={nrows*ncols} != C={C}'

    canvas = np.ones(((H + space) * nrows, (W + space) * ncols)) * 0.5
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            y0 = r * (H + space) + space // 2
            x0 = c * (W + space) + space // 2
            canvas[y0:y0+H, x0:x0+W] = spikes[idx]
            idx += 1

    fig, ax = plt.subplots(figsize=(max(5, ncols * 1.2), max(4, nrows * 1.2)), dpi=200)
    ax.imshow(canvas, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.axis('off')

    fig.tight_layout()
    return fig


# ============================================================================
#  图 6: 替代梯度函数对比 (Surrogate Gradient Comparison)
# ============================================================================

def plot_surrogate_gradients(title='Surrogate Gradient Functions'):
    """
    绘制多种替代梯度函数的原函数和梯度, 与 Heaviside 阶梯函数对比。
    包含: PiecewiseQuadratic, Sigmoid, ATan, SoftSign, PiecewiseExp
    """
    x = np.linspace(-3, 3, 1000)

    # --- 定义各函数 ---
    def heaviside(x):
        return (x >= 0).astype(float)

    # Sigmoid 替代函数
    def sigmoid_prim(x, alpha=4.0):
        return 1 / (1 + np.exp(-alpha * x))
    def sigmoid_grad(x, alpha=4.0):
        s = sigmoid_prim(x, alpha)
        return alpha * s * (1 - s)

    # ATan 替代函数
    def atan_prim(x, alpha=2.0):
        return np.arctan(np.pi / 2 * alpha * x) / np.pi + 0.5
    def atan_grad(x, alpha=2.0):
        return alpha / 2 / (1 + (np.pi / 2 * alpha * x) ** 2)

    # PiecewiseQuadratic 替代函数
    def pwq_prim(x, alpha=1.5):
        y = np.zeros_like(x)
        mask = np.abs(x) <= 1 / alpha
        y[mask] = -0.5 * alpha**2 * np.abs(x[mask]) * x[mask] + alpha * x[mask] + 0.5
        y[x > 1/alpha] = 1.0
        return y
    def pwq_grad(x, alpha=1.5):
        y = np.zeros_like(x)
        mask = np.abs(x) <= 1 / alpha
        y[mask] = -alpha**2 * np.abs(x[mask]) + alpha
        return y

    # PiecewiseExp 替代函数
    def pwe_prim(x, alpha=2.0):
        y = np.where(x >= 0, 1 - 0.5 * np.exp(-alpha * x), 0.5 * np.exp(alpha * x))
        return y
    def pwe_grad(x, alpha=2.0):
        return alpha / 2 * np.exp(-alpha * np.abs(x))

    # SoftSign 替代函数
    def ss_prim(x, alpha=3.0):
        return 0.5 * (alpha * x / (1 + np.abs(alpha * x)) + 1)
    def ss_grad(x, alpha=3.0):
        return alpha / 2 / (1 + np.abs(alpha * x)) ** 2

    surrogates = OrderedDict([
        ('Sigmoid',            {'prim': sigmoid_prim, 'grad': sigmoid_grad, 'params': {'alpha': 4.0}}),
        ('ATan',               {'prim': atan_prim,    'grad': atan_grad,    'params': {'alpha': 2.0}}),
        ('PiecewiseQuadratic', {'prim': pwq_prim,     'grad': pwq_grad,     'params': {'alpha': 1.5}}),
        ('PiecewiseExp',       {'prim': pwe_prim,     'grad': pwe_grad,     'params': {'alpha': 2.0}}),
        ('SoftSign',           {'prim': ss_prim,      'grad': ss_grad,      'params': {'alpha': 3.0}}),
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图: 原函数 ---
    ax1.plot(x, heaviside(x), color='black', linewidth=2, linestyle='-.', label='Heaviside', alpha=0.6)
    for i, (name, funcs) in enumerate(surrogates.items()):
        alpha_val = funcs['params']['alpha']
        y = funcs['prim'](x, alpha_val)
        ax1.plot(x, y, color=COLOR_LIST[i], linewidth=1.8,
                 label=f'{name} ($\\alpha$={alpha_val})')

    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$g(x)$')
    ax1.set_title('(a) Primitive Functions', fontweight='bold', pad=10)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.1, 1.3)

    # --- 右图: 梯度 ---
    for i, (name, funcs) in enumerate(surrogates.items()):
        alpha_val = funcs['params']['alpha']
        y = funcs['grad'](x, alpha_val)
        ax2.plot(x, y, color=COLOR_LIST[i], linewidth=1.8,
                 label=f'{name} ($\\alpha$={alpha_val})')

    ax2.set_xlabel('$x$')
    ax2.set_ylabel("$g'(x)$")
    ax2.set_title('(b) Surrogate Gradients', fontweight='bold', pad=10)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlim(-3, 3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ============================================================================
#  图 7: 逐层发放率分析 (Layer-wise Firing Rate)
# ============================================================================

def plot_layerwise_firing_rate(layer_names, firing_rates, silent_ratios=None,
                                title='Layer-wise Firing Rate Analysis'):
    """
    逐层平均发放率和静默神经元比例的双面板图。

    参数:
        layer_names:   list of str
        firing_rates:  list of float
        silent_ratios: list of float or None
    """
    n = len(layer_names)
    has_silent = silent_ratios is not None

    fig, axes = plt.subplots(1, 2 if has_silent else 1,
                             figsize=(12 if has_silent else 7, 5))
    if not has_silent:
        axes = [axes]

    x = np.arange(n)

    # --- 发放率 ---
    ax = axes[0]
    bars = ax.barh(x, firing_rates, color=PALETTE['blue'], edgecolor='white',
                   height=0.6, zorder=3)
    for bar, rate in zip(bars, firing_rates):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{rate:.3f}', va='center', fontsize=9, color=PALETTE['text'])
    ax.set_yticks(x)
    ax.set_yticklabels(layer_names, fontsize=9)
    ax.set_xlabel('Average Firing Rate')
    ax.set_title('(a) Firing Rate per Layer', fontweight='bold', pad=10)
    ax.invert_yaxis()

    # --- 静默比例 ---
    if has_silent:
        ax2 = axes[1]
        bars2 = ax2.barh(x, silent_ratios, color=PALETTE['red'], edgecolor='white',
                         height=0.6, zorder=3, alpha=0.85)
        for bar, ratio in zip(bars2, silent_ratios):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{ratio:.1%}', va='center', fontsize=9, color=PALETTE['text'])
        ax2.set_yticks(x)
        ax2.set_yticklabels(layer_names, fontsize=9)
        ax2.set_xlabel('Silent Neuron Ratio')
        ax2.set_title('(b) Silent Neurons per Layer', fontweight='bold', pad=10)
        ax2.invert_yaxis()

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ============================================================================
#  图 8: QCFS 量化阶梯函数可视化
# ============================================================================

def plot_qcfs_quantization(L_values=None, title='QCFS Quantization Step Function'):
    """
    绘制 QCFS 中不同量化级别 L 下的阶梯激活函数。
    clip(floor(x*L + 0.5) / L, 0, 1)

    参数:
        L_values: list of int, 量化级别, 例如 [2, 4, 8, 16]
    """
    if L_values is None:
        L_values = [2, 4, 8, 16]

    x = np.linspace(-0.5, 1.8, 2000)
    fig, ax = plt.subplots(figsize=(7, 5))

    # ReLU 参考
    relu = np.maximum(x, 0)
    ax.plot(x, relu, color='black', linewidth=1.5, linestyle=':', label='ReLU', alpha=0.5)

    for i, L in enumerate(L_values):
        y = np.clip(np.floor(x * L + 0.5) / L, 0, 1)
        ax.plot(x, y, color=COLOR_LIST[i % len(COLOR_LIST)], linewidth=2.0,
                label=f'QCFS ($L={L}$)', alpha=0.9)

    ax.set_xlabel('Pre-activation $x$')
    ax.set_ylabel('Post-activation $a(x)$')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(-0.1, 1.3)

    # 标注 clip 区域
    ax.axvspan(-0.5, 0, alpha=0.05, color='grey')
    ax.axvspan(1, 1.8, alpha=0.05, color='grey')
    ax.text(-0.25, 0.5, 'Clipped\n(=0)', ha='center', va='center', fontsize=8,
            color=PALETTE['grey'], style='italic')
    ax.text(1.4, 0.5, 'Clipped\n(=1)', ha='center', va='center', fontsize=8,
            color=PALETTE['grey'], style='italic')

    fig.tight_layout()
    return fig


# ============================================================================
#  图 9: IF vs LIF 神经元对比 (Neuron Model Comparison)
# ============================================================================

def plot_if_vs_lif(title='IF vs LIF Neuron Dynamics'):
    """
    对比 IF 和 LIF 神经元在相同输入下的膜电位变化。
    使用纯 numpy 模拟, 无需 torch 依赖。
    """
    T = 100
    np.random.seed(42)
    # 恒定输入 + 微小噪声
    I = np.ones(T) * 0.3 + np.random.randn(T) * 0.02

    v_threshold = 1.0
    v_reset = 0.0
    tau = 10.0

    # --- IF 模拟 ---
    v_if = np.zeros(T)
    s_if = np.zeros(T)
    v = v_reset
    for t in range(T):
        v = v + I[t]
        if v >= v_threshold:
            s_if[t] = 1
            v = v_reset
        v_if[t] = v

    # --- LIF 模拟 ---
    v_lif = np.zeros(T)
    s_lif = np.zeros(T)
    v = v_reset
    for t in range(T):
        v = v + (I[t] - (v - v_reset)) / tau
        if v >= v_threshold:
            s_lif[t] = 1
            v = v_reset
        v_lif[t] = v

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # IF
    axes[0].plot(np.arange(T), v_if, color=PALETTE['blue'], linewidth=1.5, label='$V_{IF}(t)$')
    axes[0].axhline(v_threshold, color=PALETTE['red'], linestyle='--', linewidth=1, alpha=0.7)
    spike_t_if = np.where(s_if == 1)[0]
    axes[0].scatter(spike_t_if, np.ones_like(spike_t_if) * v_threshold, color=PALETTE['red'],
                    marker='v', s=50, zorder=5, label='Spike')
    axes[0].set_ylabel('Membrane Potential')
    axes[0].set_title('(a) Integrate-and-Fire (IF)', fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper right')

    # LIF
    axes[1].plot(np.arange(T), v_lif, color=PALETTE['green'], linewidth=1.5, label='$V_{LIF}(t)$')
    axes[1].axhline(v_threshold, color=PALETTE['red'], linestyle='--', linewidth=1, alpha=0.7)
    spike_t_lif = np.where(s_lif == 1)[0]
    axes[1].scatter(spike_t_lif, np.ones_like(spike_t_lif) * v_threshold, color=PALETTE['red'],
                    marker='v', s=50, zorder=5, label='Spike')
    axes[1].set_ylabel('Membrane Potential')
    axes[1].set_xlabel('Time Step $t$')
    axes[1].set_title(f'(b) Leaky Integrate-and-Fire (LIF, $\\tau={tau}$)', fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ============================================================================
#  图 10: ANN-SNN 精度差距分析 (Accuracy Gap)
# ============================================================================

def plot_accuracy_gap(timesteps, ann_acc, snn_methods,
                      title='ANN-SNN Accuracy Gap Analysis',
                      ylabel='Top-1 Accuracy (%)'):
    """
    ANN-SNN 转换精度差距分析图: ANN基准线 + 多种SNN方法曲线 + 差距阴影。

    参数:
        timesteps:    list of int
        ann_acc:      float, ANN 基准精度
        snn_methods:  OrderedDict, key=方法名, value=list of acc
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ANN baseline
    ax.axhline(ann_acc, color='black', linewidth=2, linestyle='--', alpha=0.6,
               label=f'ANN Baseline ({ann_acc:.1f}%)')

    for i, (method, accs) in enumerate(snn_methods.items()):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        marker = MARKER_LIST[i % len(MARKER_LIST)]
        ax.plot(timesteps, accs, color=color, marker=marker, markersize=7,
                linewidth=2, label=method, alpha=0.9)
        # 差距阴影 (仅对最后一个方法显示)
        if i == len(snn_methods) - 1:
            ax.fill_between(timesteps, accs, ann_acc, alpha=0.08, color=color)

    ax.set_xlabel('Time Steps $T$')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=12)

    # log2 x轴
    if all(t > 0 and (t & (t - 1) == 0) for t in timesteps):
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(timesteps)

    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(None, ann_acc + 3)

    fig.tight_layout()
    return fig


# ============================================================================
#  主程序 (Main)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SNN Visualization Tool')
    parser.add_argument('--only', type=str, default=None,
                        help='Generate specific figure(s): heatmap, spike, neuron, 3d, '
                             'feature, surrogate, firing_rate, qcfs, if_lif, gap, all')
    parser.add_argument('--output', type=str, default='./figures_snn',
                        help='Output directory (default: ./figures_snn)')
    args = parser.parse_args()

    setup_style()
    targets = args.only.split(',') if args.only else ['all']

    print('='*60)
    print('  SNN Visualization -- Publication-Quality Figures')
    print('='*60)

    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1) 膜电位热力图
    # ------------------------------------------------------------------
    if 'all' in targets or 'heatmap' in targets:
        print('\n> Generating membrane potential heatmap...')
        T, N = 50, 16
        # 模拟 LIF 神经元膜电位
        voltages = np.zeros((T, N))
        v = np.zeros(N)
        for t in range(T):
            inp = np.random.rand(N) * 0.5
            v = v * 0.9 + inp  # 简单的 LIF 衰减
            fire = v >= 1.0
            v[fire] = 0.0
            voltages[t] = v

        fig = plot_membrane_heatmap(voltages)
        save_fig(fig, 'membrane_heatmap', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2) 脉冲发放光栅图
    # ------------------------------------------------------------------
    if 'all' in targets or 'spike' in targets:
        print('\n> Generating spike raster plot...')
        T, N = 100, 20
        spikes = np.zeros((T, N))
        v = np.zeros(N)
        for t in range(T):
            inp = np.random.rand(N) * 0.4
            v = v * 0.85 + inp
            fire = v >= 1.0
            spikes[t, fire] = 1
            v[fire] = 0.0

        fig = plot_spike_raster(spikes)
        save_fig(fig, 'spike_raster', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3) 单神经元动力学
    # ------------------------------------------------------------------
    if 'all' in targets or 'neuron' in targets:
        print('\n> Generating single neuron dynamics...')
        T = 150
        v_th, v_rst, tau = 1.0, 0.0, 8.0
        I = np.ones(T) * 0.25
        I[30:60] = 0.5  # 阶梯输入
        I[90:120] = 0.6

        v_arr = np.zeros(T)
        s_arr = np.zeros(T)
        v = v_rst
        for t in range(T):
            v = v + (I[t] - (v - v_rst)) / tau
            if v >= v_th:
                s_arr[t] = 1
                v = v_rst
            v_arr[t] = v

        fig = plot_neuron_dynamics(v_arr, s_arr, v_threshold=v_th, v_reset=v_rst,
                                   input_current=I)
        save_fig(fig, 'neuron_dynamics', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4) 3D 脉冲柱状图
    # ------------------------------------------------------------------
    if 'all' in targets or '3d' in targets:
        print('\n> Generating 3D spike bar chart...')
        T, N = 8, 6
        rates = np.zeros((T, N))
        init_rate = np.random.rand(N)
        for i in range(T):
            logits = init_rate * (i + 1) ** 1.5
            rates[i] = np.exp(logits) / np.exp(logits).sum()  # softmax

        fig = plot_3d_spike_bar(rates)
        save_fig(fig, '3d_spike_bar', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5) 脉冲特征图
    # ------------------------------------------------------------------
    if 'all' in targets or 'feature' in targets:
        print('\n> Generating spiking feature maps...')
        C, H, W = 32, 8, 8
        spikes = (np.random.rand(C, H, W) > 0.75).astype(float)
        fig = plot_spiking_feature_map(spikes, nrows=4, ncols=8, space=2,
                                        title='Spiking Feature Maps (Conv Layer Output)')
        save_fig(fig, 'spiking_feature_map', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 6) 替代梯度函数
    # ------------------------------------------------------------------
    if 'all' in targets or 'surrogate' in targets:
        print('\n> Generating surrogate gradient comparison...')
        fig = plot_surrogate_gradients()
        save_fig(fig, 'surrogate_gradients', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 7) 逐层发放率
    # ------------------------------------------------------------------
    if 'all' in targets or 'firing_rate' in targets:
        print('\n> Generating layer-wise firing rate...')
        layer_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5',
                        'FC1', 'FC2', 'Output']
        firing_rates = [0.312, 0.245, 0.198, 0.156, 0.134, 0.089, 0.067, 0.042]
        silent_ratios = [0.05, 0.08, 0.12, 0.18, 0.22, 0.31, 0.38, 0.45]

        fig = plot_layerwise_firing_rate(layer_names, firing_rates, silent_ratios)
        save_fig(fig, 'layerwise_firing_rate', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 8) QCFS 量化阶梯
    # ------------------------------------------------------------------
    if 'all' in targets or 'qcfs' in targets:
        print('\n> Generating QCFS quantization step function...')
        fig = plot_qcfs_quantization(L_values=[2, 4, 8, 16, 32])
        save_fig(fig, 'qcfs_quantization', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9) IF vs LIF 对比
    # ------------------------------------------------------------------
    if 'all' in targets or 'if_lif' in targets:
        print('\n> Generating IF vs LIF comparison...')
        fig = plot_if_vs_lif()
        save_fig(fig, 'if_vs_lif', args.output)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 10) ANN-SNN 精度差距
    # ------------------------------------------------------------------
    if 'all' in targets or 'gap' in targets:
        print('\n> Generating accuracy gap analysis...')
        timesteps = [1, 2, 4, 8, 16, 32, 64]
        ann_acc = 95.21
        snn_methods = OrderedDict([
            ('QCFS',           [45.2, 72.8, 89.6, 92.4, 93.8, 94.5, 95.0]),
            ('CS-QCFS (Ours)', [62.3, 82.1, 92.3, 94.1, 95.0, 95.2, 95.2]),
        ])
        fig = plot_accuracy_gap(timesteps, ann_acc, snn_methods)
        save_fig(fig, 'accuracy_gap', args.output)
        plt.close(fig)

    print('\n' + '='*60)
    print('  [DONE] All SNN figures generated!')
    print(f'  Output dir: {os.path.abspath(args.output)}')
    print('='*60)


if __name__ == '__main__':
    main()
