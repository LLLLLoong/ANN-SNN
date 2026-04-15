"""
论文绘图工具 — Publication-Quality Figure Generator
====================================================
适用于 ANN-SNN 转换项目的论文图表绘制。

使用方法:
    python plot_paper_figures.py                  # 生成所有示例图
    python plot_paper_figures.py --only training  # 仅生成训练曲线
    
所有图片保存至 ./figures/ 目录，格式为 PDF + PNG (300 DPI)。

依赖: matplotlib, numpy, pandas (如需读取log), seaborn (可选)
"""

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
from collections import OrderedDict

# ============================================================================
#  全局样式配置 (Global Style Configuration)
# ============================================================================

# 论文级别配色方案
PALETTE = {
    # 主色系 — 灵感来自 Nature / Science 期刊
    'blue':       '#2E86AB',
    'red':        '#E8505B',
    'green':      '#1B998B',
    'orange':     '#F39C12',
    'purple':     '#8E44AD',
    'teal':       '#16A085',
    'pink':       '#E91E63',
    'brown':      '#795548',
    'grey':       '#7F8C8D',
    'dark_blue':  '#1A5276',
    'light_blue': '#85C1E9',
    'gold':       '#D4AC0D',
    # 渐变色
    'bg_light':   '#FAFAFA',
    'grid':       '#E0E0E0',
    'text':       '#2C3E50',
    'text_light': '#7F8C8D',
}

# 多模型/方法对比时的颜色列表
COLOR_LIST = [
    PALETTE['blue'], PALETTE['red'], PALETTE['green'],
    PALETTE['orange'], PALETTE['purple'], PALETTE['teal'],
    PALETTE['pink'], PALETTE['brown'],
]

# 标记符号列表
MARKER_LIST = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

# Hatching patterns for bar charts
HATCH_LIST = ['', '///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '---']


def setup_style():
    """配置全局 matplotlib 样式，使输出图像符合学术论文标准。"""
    plt.rcParams.update({
        # --- 字体 ---
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif', 'SimSun'],
        'font.size':          11,
        'axes.titlesize':     13,
        'axes.labelsize':     12,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'legend.fontsize':    10,
        # --- 线条 ---
        'lines.linewidth':    1.8,
        'lines.markersize':   6,
        # --- 坐标轴 ---
        'axes.linewidth':     1.0,
        'axes.edgecolor':     PALETTE['text'],
        'axes.labelcolor':    PALETTE['text'],
        'axes.grid':          True,
        'grid.alpha':         0.3,
        'grid.linewidth':     0.6,
        'grid.color':         PALETTE['grid'],
        # --- 刻度 ---
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        # --- 图例 ---
        'legend.framealpha':  0.9,
        'legend.edgecolor':   PALETTE['grid'],
        'legend.fancybox':    True,
        # --- 图像 ---
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
        # --- 数学字体 ---
        'mathtext.fontset':   'stix',
    })


def save_fig(fig, name, output_dir='./figures'):
    """同时保存 PDF 和 PNG 格式。"""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(output_dir, f'{name}.png'), format='png')
    print(f'  [OK] saved: {output_dir}/{name}.pdf / .png')


# ============================================================================
#  数据加载工具 (Data Loading Utilities)
# ============================================================================

def load_training_log(log_path):
    """
    加载训练日志文件。
    日志格式: Epoch  Train_Loss  Val_Loss  Val_Acc  Time(s)
    返回 dict: {'epoch': [...], 'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}
    """
    epochs, train_losses, val_losses, val_accs = [], [], [], []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Epoch'):
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                epochs.append(int(parts[0]))
                train_losses.append(float(parts[1]))
                val_losses.append(float(parts[2]))
                val_accs.append(float(parts[3]))
    return {
        'epoch': np.array(epochs),
        'train_loss': np.array(train_losses),
        'val_loss': np.array(val_losses),
        'val_acc': np.array(val_accs) * 100,  # 转换为百分比
    }


# ============================================================================
#  图表 1: 训练曲线 (Training Curves)
# ============================================================================

def plot_training_curves(logs, labels=None, title='Training Curves on CIFAR-10'):
    """
    绘制训练过程中的 loss 和 accuracy 曲线（双 Y 轴）。
    
    参数:
        logs:   list of dict, 每个 dict 由 load_training_log() 返回
        labels: list of str, 每条曲线的标签
    """
    if not isinstance(logs, list):
        logs = [logs]
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(logs))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- 左图: Loss 曲线 ---
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        ax1.plot(log['epoch'], log['train_loss'], color=color, linestyle='-',
                 label=f'{label} (Train)', alpha=0.85, linewidth=1.5)
        # val_loss 的量级较小，乘以系数使得可视化更清晰
        # 这里直接用原始值
        ax1.plot(log['epoch'], log['val_loss'] * 1000, color=color, linestyle='--',
                 label=f'{label} (Val ×1e3)', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Training & Validation Loss', fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=8, ncol=1)
    ax1.set_xlim(0, max(log['epoch'][-1] for log in logs))

    # --- 右图: Accuracy 曲线 ---
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        ax2.plot(log['epoch'], log['val_acc'], color=color,
                 marker=MARKER_LIST[i % len(MARKER_LIST)],
                 markevery=max(1, len(log['epoch']) // 15),
                 markersize=5, label=label, alpha=0.85, linewidth=1.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('(b) Validation Accuracy', fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, max(log['epoch'][-1] for log in logs))
    # 让 y 轴上限留一点空间
    max_acc = max(log['val_acc'].max() for log in logs)
    ax2.set_ylim(None, min(100, max_acc + 3))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 2: 模型准确率对比柱状图 (Bar Chart Comparison)
# ============================================================================

def plot_accuracy_comparison(methods, accuracies, title='Accuracy Comparison on CIFAR-10',
                              ylabel='Top-1 Accuracy (%)', highlight_idx=None):
    """
    绘制不同方法/模型之间的准确率对比柱状图。
    
    参数:
        methods:       list of str, 方法名称
        accuracies:    list of float, 对应准确率 (%)
        highlight_idx: int or None, 要高亮的柱子索引（用于突出自己的方法）
    """
    n = len(methods)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    
    x = np.arange(n)
    bar_width = 0.6
    
    colors = []
    for i in range(n):
        if highlight_idx is not None and i == highlight_idx:
            colors.append(PALETTE['red'])
        else:
            colors.append(PALETTE['blue'])
    
    bars = ax.bar(x, accuracies, width=bar_width, color=colors, 
                  edgecolor='white', linewidth=0.8, zorder=3)
    
    # 在柱子上方添加数值标签
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=PALETTE['text'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=12)
    
    # y 轴范围
    min_acc = min(accuracies)
    ax.set_ylim(max(0, min_acc - 5), 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 3: 分组柱状图 (Grouped Bar Chart)
# ============================================================================

def plot_grouped_bar(categories, group_data, group_labels,
                     title='Performance Comparison', ylabel='Accuracy (%)'):
    """
    分组柱状图：对比多个方法在不同类别（数据集/网络）上的表现。
    
    参数:
        categories:   list of str, 分组类别名 (如 ['CIFAR-10', 'CIFAR-100'])
        group_data:   list of list, 每个子 list 为一个方法的数据
        group_labels: list of str, 每个方法的名称
    """
    n_categories = len(categories)
    n_groups = len(group_labels)
    
    fig, ax = plt.subplots(figsize=(max(7, n_categories * 2.5), 5))
    
    bar_width = 0.8 / n_groups
    x = np.arange(n_categories)
    
    for i, (data, label) in enumerate(zip(group_data, group_labels)):
        offset = (i - (n_groups - 1) / 2) * bar_width
        bars = ax.bar(x + offset, data, width=bar_width * 0.9,
                      color=COLOR_LIST[i % len(COLOR_LIST)],
                      edgecolor='white', linewidth=0.6,
                      hatch=HATCH_LIST[i % len(HATCH_LIST)],
                      label=label, zorder=3)
        # 数值标签
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7.5,
                    color=PALETTE['text'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=9, ncol=min(n_groups, 3))
    
    min_val = min(min(d) for d in group_data)
    ax.set_ylim(max(0, min_val - 8), None)
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 4: 准确率 vs 时间步 (Accuracy vs Timestep)
# ============================================================================

def plot_accuracy_vs_timestep(timesteps, acc_dict, title='SNN Accuracy vs. Time Steps',
                               xlabel='Time Steps $T$', ylabel='Accuracy (%)'):
    """
    绘制不同方法在不同时间步下的 SNN 准确率折线图。
    这是 ANN-SNN 转换论文中非常常见的图。
    
    参数:
        timesteps: list of int, 时间步数
        acc_dict:  OrderedDict, key=方法名, value=list of accuracy
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for i, (method, accs) in enumerate(acc_dict.items()):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        marker = MARKER_LIST[i % len(MARKER_LIST)]
        ax.plot(timesteps, accs, color=color, marker=marker,
                markersize=7, label=method, linewidth=2, alpha=0.9)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=12)
    
    # x 轴使用 log2 刻度（时间步通常为2的幂）
    if all(t > 0 and (t & (t - 1) == 0) for t in timesteps):  # 检查是否全为2的幂
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(timesteps)
    
    ax.legend(loc='lower right', fontsize=9)
    
    max_acc = max(max(v) for v in acc_dict.values())
    ax.set_ylim(None, min(100, max_acc + 3))
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 5: 混淆矩阵热力图 (Confusion Matrix Heatmap)
# ============================================================================

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix',
                           normalize=True, cmap='Blues'):
    """
    绘制混淆矩阵热力图。
    
    参数:
        cm:          np.ndarray, 混淆矩阵 (n_classes x n_classes)
        class_names: list of str, 类别名称
        normalize:   bool, 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', fontsize=10)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    
    # 在每个格子中写入数值
    thresh = cm.max() / 2.0
    fmt = '.2f' if normalize else 'd'
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > thresh else PALETTE['text']
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center', color=color, fontsize=8)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title, fontweight='bold', pad=12)
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 6: 雷达图 (Radar / Spider Chart)
# ============================================================================

def plot_radar_chart(categories, data_dict, title='Multi-dimensional Comparison'):
    """
    绘制雷达图，用于多维度对比不同方法。
    
    参数:
        categories: list of str, 各维度名称
        data_dict:  dict, key=方法名, value=list of float (每个维度的值, 0~100)
    """
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for i, (method, values) in enumerate(data_dict.items()):
        vals = values + values[:1]
        color = COLOR_LIST[i % len(COLOR_LIST)]
        ax.plot(angles, vals, 'o-', color=color, linewidth=2,
                markersize=5, label=method)
        ax.fill(angles, vals, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontweight='bold', pad=20, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 7: Loss 曲面 / 学习率调度可视化
# ============================================================================

def plot_lr_schedule(epochs, lr_values=None, schedule_type='cosine',
                     title='Learning Rate Schedule'):
    """
    绘制学习率调度曲线。
    
    参数:
        epochs:        int, 总训练轮数
        lr_values:     list or None, 自定义学习率列表; None时自动生成
        schedule_type: 'cosine', 'step', 'warmup_cosine'
    """
    epoch_range = np.arange(epochs)
    
    if lr_values is None:
        base_lr = 0.1
        if schedule_type == 'cosine':
            lr_values = 0.5 * base_lr * (1 + np.cos(np.pi * epoch_range / epochs))
        elif schedule_type == 'step':
            lr_values = np.where(epoch_range < 100, base_lr,
                        np.where(epoch_range < 200, base_lr * 0.1, base_lr * 0.01))
        elif schedule_type == 'warmup_cosine':
            warmup = 5
            lr_values = np.where(
                epoch_range < warmup,
                base_lr * epoch_range / warmup,
                0.5 * base_lr * (1 + np.cos(np.pi * (epoch_range - warmup) / (epochs - warmup)))
            )
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(epoch_range, lr_values, color=PALETTE['blue'], linewidth=2)
    ax.fill_between(epoch_range, lr_values, alpha=0.15, color=PALETTE['blue'])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlim(0, epochs - 1)
    ax.set_ylim(0, None)
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 8: 箱线图 (Box Plot) — 多次实验结果对比
# ============================================================================

def plot_box_comparison(data_dict, title='Experimental Results Distribution',
                         ylabel='Accuracy (%)'):
    """
    箱线图，展示多次实验结果的分布。
    
    参数:
        data_dict: dict, key=方法名, value=list of float (多次实验结果)
    """
    fig, ax = plt.subplots(figsize=(max(6, len(data_dict) * 1.5), 5))
    
    labels = list(data_dict.keys())
    data = list(data_dict.values())
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                                                     markeredgecolor=PALETTE['red'], markersize=6))
    
    for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        box.set_facecolor(color + '40')  # 半透明
        box.set_edgecolor(color)
        box.set_linewidth(1.5)
        median.set_color(PALETTE['red'])
        median.set_linewidth(2)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=12)
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 9: 综合子图面板 (Multi-panel Figure)
# ============================================================================

def plot_multi_panel(logs, labels=None, title='Comprehensive Training Analysis'):
    """
    四合一面板：Train Loss, Val Loss, Val Acc, Learning Rate。
    
    参数:
        logs:   list of dict
        labels: list of str
    """
    if not isinstance(logs, list):
        logs = [logs]
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(logs))]
    
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        marker = MARKER_LIST[i % len(MARKER_LIST)]
        me = max(1, len(log['epoch']) // 12)
        
        # (a) Train Loss
        axes[0].plot(log['epoch'], log['train_loss'], color=color,
                     label=label, linewidth=1.5, alpha=0.85)
        
        # (b) Val Loss (×1e3)
        axes[1].plot(log['epoch'], log['val_loss'] * 1000, color=color,
                     label=label, linewidth=1.5, alpha=0.85)
        
        # (c) Val Accuracy
        axes[2].plot(log['epoch'], log['val_acc'], color=color,
                     marker=marker, markevery=me, markersize=4,
                     label=label, linewidth=1.5, alpha=0.85)
        
        # (d) Smoothed Accuracy (滑动平均)
        window = min(10, len(log['val_acc']) // 5)
        if window > 1:
            smoothed = np.convolve(log['val_acc'], np.ones(window)/window, mode='valid')
            axes[3].plot(log['epoch'][:len(smoothed)], smoothed, color=color,
                         label=f'{label} (smooth)', linewidth=2, alpha=0.9)
    
    titles = ['Train Loss', 'Val Loss (×1e$^3$)', 'Val Accuracy (%)', 'Smoothed Val Accuracy (%)']
    ylabels = ['Loss', 'Loss (×1e$^3$)', 'Accuracy (%)', 'Accuracy (%)']
    
    for idx, ax in enumerate(axes):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabels[idx])
        ax.set_title(f'{panel_labels[idx]} {titles[idx]}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8, loc='best')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    return fig


# ============================================================================
#  图表 10: 散点图 (Scatter — Accuracy vs Latency Trade-off)
# ============================================================================

def plot_scatter_tradeoff(data_dict, title='Accuracy-Latency Trade-off',
                           xlabel='Latency (Time Steps)', ylabel='Accuracy (%)',
                           annotate=True):
    """
    散点图，展示准确率 vs 延迟（时间步）的 trade-off。
    
    参数:
        data_dict: dict, key=方法名, value=(latency, accuracy) tuple
        annotate:  bool, 是否标注方法名
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    for i, (method, (lat, acc)) in enumerate(data_dict.items()):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        marker = MARKER_LIST[i % len(MARKER_LIST)]
        ax.scatter(lat, acc, c=color, marker=marker, s=120, label=method,
                   edgecolors='white', linewidth=1, zorder=3)
        if annotate:
            ax.annotate(method, (lat, acc), textcoords='offset points',
                       xytext=(8, 5), fontsize=8, color=color, fontweight='bold')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    
    fig.tight_layout()
    return fig


# ============================================================================
#  图表 11: 表格式图表 (Table Figure)
# ============================================================================

def plot_table(col_headers, row_headers, cell_data, title='Results Summary',
               highlight_cols=None):
    """
    绘制论文中常见的结果表格图。
    
    参数:
        col_headers:    list of str
        row_headers:    list of str
        cell_data:      2D list
        highlight_cols: list of int, 要高亮的列索引
    """
    fig, ax = plt.subplots(figsize=(max(6, len(col_headers) * 1.8), max(3, len(row_headers) * 0.6 + 1)))
    ax.axis('off')
    
    table = ax.table(cellText=cell_data,
                     colLabels=col_headers,
                     rowLabels=row_headers,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    
    # 表头样式
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor(PALETTE['dark_blue'])
        cell.set_text_props(color='white', fontweight='bold')
    
    # 行标签样式
    for i in range(len(row_headers)):
        cell = table[i + 1, -1]
        cell.set_facecolor(PALETTE['light_blue'] + '60')
        cell.set_text_props(fontweight='bold')
    
    # 高亮列
    if highlight_cols:
        for i in range(1, len(row_headers) + 1):
            for j in highlight_cols:
                table[i, j].set_facecolor('#FFF9C4')
                table[i, j].set_text_props(fontweight='bold', color=PALETTE['red'])
    
    ax.set_title(title, fontweight='bold', fontsize=13, pad=20)
    fig.tight_layout()
    return fig


# ============================================================================
#  主程序 (Main)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='论文绘图工具')
    parser.add_argument('--only', type=str, default=None,
                        help='仅生成指定类型的图: training, comparison, timestep, '
                             'confusion, radar, lr, box, panel, scatter, table, all')
    parser.add_argument('--output', type=str, default='./figures',
                        help='输出目录 (默认: ./figures)')
    args = parser.parse_args()
    
    setup_style()
    targets = args.only.split(',') if args.only else ['all']
    
    print('='*60)
    print('  Publication-Quality Figure Generator')
    print('='*60)
    
    # ------------------------------------------------------------------
    # 1) 训练曲线 — 使用真实项目数据
    # ------------------------------------------------------------------
    if 'all' in targets or 'training' in targets:
        print('\n> Generating training curves...')
        log_files = {
            'ResNet-18': './saved_models/cifar10/resnet18/origin_T[4]_log.txt',
            'VGG-16':    './saved_models/cifar10/vgg16/origin_T[4]_log.txt',
        }
        logs, labels = [], []
        for label, path in log_files.items():
            if os.path.exists(path):
                logs.append(load_training_log(path))
                labels.append(label)
                print(f'    [OK] loaded: {path}')
            else:
                print(f'    [MISS] not found: {path}')
        
        if logs:
            fig = plot_training_curves(logs, labels, title='Training Curves on CIFAR-10')
            save_fig(fig, 'training_curves', args.output)
            plt.close(fig)
    
    # ------------------------------------------------------------------
    # 2) 准确率对比柱状图 — 示例数据
    # ------------------------------------------------------------------
    if 'all' in targets or 'comparison' in targets:
        print('\n> Generating accuracy comparison bar chart...')
        methods = ['ANN Baseline', 'QCFS (T=4)', 'QCFS (T=8)', 'QCFS (T=16)',
                   'CS-QCFS (T=4)', 'CS-QCFS (T=8)']
        accuracies = [95.21, 91.58, 93.82, 94.76, 93.45, 95.02]
        fig = plot_accuracy_comparison(methods, accuracies, highlight_idx=4)
        save_fig(fig, 'accuracy_comparison', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 3) 分组柱状图 — 示例数据
    # ------------------------------------------------------------------
    if 'all' in targets or 'grouped' in targets or 'comparison' in targets:
        print('\n> Generating grouped bar chart...')
        categories = ['CIFAR-10\nResNet-18', 'CIFAR-10\nVGG-16', 'CIFAR-100\nResNet-18']
        group_data = [
            [93.92, 94.99, 74.85],  # QCFS
            [94.45, 95.12, 76.23],  # CS-QCFS (Ours)
            [92.18, 93.47, 72.56],  # Baseline
        ]
        group_labels = ['QCFS', 'CS-QCFS (Ours)', 'Baseline']
        fig = plot_grouped_bar(categories, group_data, group_labels)
        save_fig(fig, 'grouped_comparison', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 4) 准确率 vs 时间步
    # ------------------------------------------------------------------
    if 'all' in targets or 'timestep' in targets:
        print('\n> Generating accuracy vs timestep...')
        timesteps = [1, 2, 4, 8, 16, 32, 64]
        acc_dict = OrderedDict([
            ('ANN Baseline',  [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            ('QCFS',          [45.2, 72.8, 89.6, 92.4, 93.8, 94.5, 95.0]),
            ('CS-QCFS (Ours)',[62.3, 82.1, 92.3, 94.1, 95.0, 95.2, 95.2]),
        ])
        # ANN baseline 是固定值
        acc_dict['ANN Baseline'] = [95.21] * len(timesteps)
        fig = plot_accuracy_vs_timestep(timesteps, acc_dict)
        save_fig(fig, 'accuracy_vs_timestep', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 5) 混淆矩阵
    # ------------------------------------------------------------------
    if 'all' in targets or 'confusion' in targets:
        print('\n> Generating confusion matrix...')
        np.random.seed(42)
        n_classes = 10
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        # 生成一个看起来合理的混淆矩阵
        cm = np.random.randint(0, 10, size=(n_classes, n_classes))
        np.fill_diagonal(cm, np.random.randint(85, 98, size=n_classes))
        fig = plot_confusion_matrix(cm, cifar10_classes,
                                     title='Confusion Matrix — CS-QCFS on CIFAR-10')
        save_fig(fig, 'confusion_matrix', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 6) 雷达图
    # ------------------------------------------------------------------
    if 'all' in targets or 'radar' in targets:
        print('\n> Generating radar chart...')
        categories = ['Accuracy', 'Latency', 'Energy\nEfficiency',
                       'Memory', 'Conversion\nLoss', 'Scalability']
        data_dict = {
            'QCFS':           [88, 60, 75, 80, 70, 85],
            'CS-QCFS (Ours)': [94, 85, 82, 78, 90, 88],
            'Direct SNN':     [82, 90, 88, 70, 50, 72],
        }
        fig = plot_radar_chart(categories, data_dict)
        save_fig(fig, 'radar_chart', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 7) 学习率调度
    # ------------------------------------------------------------------
    if 'all' in targets or 'lr' in targets:
        print('\n> Generating LR schedule...')
        fig = plot_lr_schedule(300, schedule_type='cosine',
                                title='Cosine Annealing Learning Rate Schedule')
        save_fig(fig, 'lr_schedule_cosine', args.output)
        plt.close(fig)
        
        fig = plot_lr_schedule(300, schedule_type='warmup_cosine',
                                title='Warmup + Cosine Annealing LR Schedule')
        save_fig(fig, 'lr_schedule_warmup', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 8) 箱线图 — 多次实验分布
    # ------------------------------------------------------------------
    if 'all' in targets or 'box' in targets:
        print('\n> Generating box plot...')
        np.random.seed(42)
        box_data = {
            'QCFS\n(T=4)':      np.random.normal(93.5, 0.3, 10).tolist(),
            'QCFS\n(T=8)':      np.random.normal(94.2, 0.25, 10).tolist(),
            'CS-QCFS\n(T=4)':   np.random.normal(94.8, 0.2, 10).tolist(),
            'CS-QCFS\n(T=8)':   np.random.normal(95.1, 0.15, 10).tolist(),
        }
        fig = plot_box_comparison(box_data)
        save_fig(fig, 'box_comparison', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 9) 综合面板
    # ------------------------------------------------------------------
    if 'all' in targets or 'panel' in targets:
        print('\n> Generating multi-panel figure...')
        log_files = {
            'ResNet-18': './saved_models/cifar10/resnet18/origin_T[4]_log.txt',
            'VGG-16':    './saved_models/cifar10/vgg16/origin_T[4]_log.txt',
        }
        logs, labels = [], []
        for label, path in log_files.items():
            if os.path.exists(path):
                logs.append(load_training_log(path))
                labels.append(label)
        if logs:
            fig = plot_multi_panel(logs, labels)
            save_fig(fig, 'multi_panel', args.output)
            plt.close(fig)
    
    # ------------------------------------------------------------------
    # 10) 散点图 — Accuracy vs Latency
    # ------------------------------------------------------------------
    if 'all' in targets or 'scatter' in targets:
        print('\n> Generating scatter plot...')
        scatter_data = OrderedDict([
            ('ANN',              (1,  95.21)),
            ('QCFS (T=4)',       (4,  91.58)),
            ('QCFS (T=8)',       (8,  93.82)),
            ('QCFS (T=16)',      (16, 94.76)),
            ('CS-QCFS (T=4)',    (4,  93.45)),
            ('CS-QCFS (T=8)',    (8,  95.02)),
            ('Direct SNN (T=4)', (4,  88.30)),
        ])
        fig = plot_scatter_tradeoff(scatter_data)
        save_fig(fig, 'scatter_tradeoff', args.output)
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # 11) 结果表格
    # ------------------------------------------------------------------
    if 'all' in targets or 'table' in targets:
        print('\n> Generating table figure...')
        col_headers = ['Method', 'T', 'CIFAR-10', 'CIFAR-100', 'ImageNet']
        row_headers = ['1', '2', '3', '4', '5']
        cell_data = [
            ['QCFS',           '4',  '91.58', '72.34', '65.12'],
            ['QCFS',           '8',  '93.82', '74.85', '68.45'],
            ['CS-QCFS (Ours)', '4',  '93.45', '75.23', '67.89'],
            ['CS-QCFS (Ours)', '8',  '95.02', '76.56', '70.12'],
            ['ANN Baseline',   '-',  '95.21', '77.80', '71.50'],
        ]
        fig = plot_table(col_headers, row_headers, cell_data,
                          title='Summary of Experimental Results',
                          highlight_cols=[2, 3, 4])
        save_fig(fig, 'results_table', args.output)
        plt.close(fig)
    
    print('\n' + '='*60)
    print('  [DONE] All figures generated!')
    print(f'  Output dir: {os.path.abspath(args.output)}')
    print('='*60)


if __name__ == '__main__':
    main()
