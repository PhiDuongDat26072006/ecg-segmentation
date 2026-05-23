# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse


# Tên 12 chuyển đạo chuẩn
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
N_LEADS = len(LEAD_NAMES)

# Màu cho từng lớp phân vùng
SEG_COLORS = {
    0: '#4A90D9',   # Sóng P - xanh dương
    1: '#D94A4A',   # Phức bộ QRS - đỏ
    2: '#4AD98E',   # Sóng T - xanh lá
    3: 'none',      # Baseline - trong suốt
}

SEG_LABELS = {0: 'Sóng P', 1: 'Phức bộ QRS', 2: 'Sóng T', 3: 'Baseline'}
CLS_LABELS = {0: 'Bình thường', 1: 'Rung nhĩ'}


def _color_segments(ax, labels, n_samples):
    """Tô màu nền cho các vùng phân đoạn (P, QRS, T) trên trục ax."""
    for c in range(3):  # Chỉ tô P=0, QRS=1, T=2 (bỏ qua Baseline=3)
        mask = (labels == c)
        if not np.any(mask):
            continue

        # Tìm các vùng liên tiếp
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [n_samples]])

        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.3, color=SEG_COLORS[c])


def plot_ecg_comparison(predictions_path, record_idx=0, save_path=None):
    """
    Vẽ đồ thị ECG so sánh Ground Truth và Prediction cho 1 bản ghi (12 chuyển đạo).

    Args:
        predictions_path: Đường dẫn file predictions.npz
        record_idx: Chỉ số bản ghi cần vẽ (0-indexed)
        save_path: Đường dẫn lưu file ảnh PNG (nếu None, tự tạo tên)
    """
    # Load dữ liệu dự đoán
    data = np.load(predictions_path)
    seg_pred = data['seg_pred']   # (N, L) — nhãn phân vùng dự đoán
    seg_true = data['seg_true']   # (N, L) — nhãn phân vùng thật
    cls_pred = data['cls_pred']   # (N,)   — nhãn phân loại dự đoán
    cls_true = data['cls_true']   # (N,)   — nhãn phân loại thật
    signals  = data['signals']    # (N, L) — tín hiệu ECG gốc

    n_total = len(signals)
    n_records = n_total // N_LEADS

    if record_idx < 0 or record_idx >= n_records:
        print(f'Error: record_idx phải nằm trong khoảng [0, {n_records - 1}]')
        return

    # Lấy 12 chuyển đạo của bản ghi này
    start_idx = record_idx * N_LEADS
    n_samples = signals.shape[1]

    # Nhãn phân loại của bản ghi
    gt_cls = CLS_LABELS.get(cls_true[start_idx], str(cls_true[start_idx]))
    pred_cls = CLS_LABELS.get(cls_pred[start_idx], str(cls_pred[start_idx]))

    # ==================== Vẽ đồ thị ====================
    fig, axes = plt.subplots(N_LEADS, 2, figsize=(24, 30), sharex=True)

    fig.suptitle(
        f'Bản ghi #{record_idx + 1}  —  '
        f'Phân loại: GT = {gt_cls} | Pred = {pred_cls}',
        fontsize=18, fontweight='bold', y=0.998
    )

    axes[0, 0].set_title('Ground Truth (Nhãn thật)', fontsize=15, fontweight='bold', pad=12)
    axes[0, 1].set_title('Prediction (Dự đoán)', fontsize=15, fontweight='bold', pad=12)

    for i, lead in enumerate(LEAD_NAMES):
        idx = start_idx + i
        signal = signals[idx]
        gt = seg_true[idx]
        pred = seg_pred[idx]

        for col, labels in enumerate([gt, pred]):
            ax = axes[i, col]

            # Vẽ tín hiệu ECG
            ax.plot(signal, color='black', linewidth=0.6)

            # Tô màu nền cho các vùng phân đoạn
            _color_segments(ax, labels, n_samples)

            # Nhãn chuyển đạo ở bên trái
            ax.set_ylabel(lead, fontsize=12, fontweight='bold', rotation=0, labelpad=30)
            ax.tick_params(axis='both', labelsize=7)
            ax.set_xlim(0, n_samples)

    # Nhãn trục X ở hàng cuối
    axes[-1, 0].set_xlabel('Sample', fontsize=12)
    axes[-1, 1].set_xlabel('Sample', fontsize=12)

    # Legend
    legend_patches = [
        mpatches.Patch(color=SEG_COLORS[0], alpha=0.3, label='Sóng P'),
        mpatches.Patch(color=SEG_COLORS[1], alpha=0.3, label='Phức bộ QRS'),
        mpatches.Patch(color=SEG_COLORS[2], alpha=0.3, label='Sóng T'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=13,
               bbox_to_anchor=(0.5, 0.001))

    plt.tight_layout(rect=[0, 0.025, 1, 0.99])

    # Lưu file ảnh
    if save_path is None:
        save_path = f'ecg_record_{record_idx}.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to: {save_path}')
    plt.close()


# ==================== Command-line interface ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vẽ đồ thị ECG so sánh Ground Truth và Prediction.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--predictions', type=str, default='predictions.npz',
                        help='Đường dẫn file predictions.npz')
    parser.add_argument('--record', type=int, default=0,
                        help='Chỉ số bản ghi cần vẽ (0-indexed)')
    parser.add_argument('--save', type=str, default=None,
                        help='Đường dẫn lưu file ảnh PNG')

    args = parser.parse_args()
    plot_ecg_comparison(args.predictions, args.record, args.save)