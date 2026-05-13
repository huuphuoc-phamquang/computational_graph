"""
Dự đoán điểm cuối kỳ (final) từ điểm giữa kỳ (midterm)
sử dụng Hồi quy Tuyến tính triển khai bằng Đồ thị Tính toán (Computational Graph).

Mô hình:  ŷ = w * x + b
Hàm mất mát:  L = (1/N) * Σ(ŷ - y)²
Tối ưu:  Gradient Descent + Backpropagation

Môn: CS523 - Data Structures & Algorithms
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================
# 1. CÁC NODE TRONG ĐỒ THỊ TÍNH TOÁN
# ============================================================

class Node:
    """Node cơ sở trong đồ thị tính toán."""
    def __init__(self, name=""):
        self.name = name
        self.output = None
        self.grad = None  # gradient ∂L/∂output

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class InputNode(Node):
    """Node đầu vào (x, y) - không cần backward."""
    def __init__(self, name=""):
        super().__init__(name)
        self.value = None

    def set_value(self, value):
        self.value = value
        self.output = value

    def forward(self):
        self.output = self.value
        return self.output

    def backward(self):
        pass  # Không cần tính gradient cho input


class ParameterNode(Node):
    """Node tham số (w, b) - cần cập nhật bằng gradient descent."""
    def __init__(self, init_value=0.0, name=""):
        super().__init__(name)
        self.output = init_value
        self.grad = 0.0

    def forward(self):
        return self.output

    def backward(self):
        pass  # Gradient được gán từ node phía sau


class MultiplyNode(Node):
    """Node nhân: z = a * b  →  ∂z/∂a = b, ∂z/∂b = a"""
    def __init__(self, name="Multiply"):
        super().__init__(name)
        self.a = None
        self.b = None

    def forward(self, a, b):
        self.a = a
        self.b = b
        self.output = a * b
        return self.output

    def backward(self):
        return self.b, self.a  # ∂z/∂a, ∂z/∂b


class AddNode(Node):
    """Node cộng: z = a + b  →  ∂z/∂a = 1, ∂z/∂b = 1"""
    def __init__(self, name="Add"):
        super().__init__(name)

    def forward(self, a, b):
        self.output = a + b
        return self.output

    def backward(self):
        return 1.0, 1.0  # ∂z/∂a, ∂z/∂b


class SubtractNode(Node):
    """Node trừ: z = a - b  →  ∂z/∂a = 1, ∂z/∂b = -1"""
    def __init__(self, name="Subtract"):
        super().__init__(name)

    def forward(self, a, b):
        self.output = a - b
        return self.output

    def backward(self):
        return 1.0, -1.0


class SquareNode(Node):
    """Node bình phương: z = a²  →  ∂z/∂a = 2a"""
    def __init__(self, name="Square"):
        super().__init__(name)
        self.a = None

    def forward(self, a):
        self.a = a
        self.output = a ** 2
        return self.output

    def backward(self):
        return 2 * self.a


class MeanNode(Node):
    """Node trung bình: z = mean(a)  →  ∂z/∂aᵢ = 1/N"""
    def __init__(self, name="Mean"):
        super().__init__(name)
        self.n = None

    def forward(self, a):
        self.n = len(a) if hasattr(a, '__len__') else 1
        self.output = np.mean(a)
        return self.output

    def backward(self):
        return 1.0 / self.n


# ============================================================
# 2. ĐỒ THỊ TÍNH TOÁN CHO LINEAR REGRESSION
# ============================================================

class LinearRegressionGraph:
    """
    Đồ thị tính toán cho mô hình hồi quy tuyến tính.

    Forward pass:
        x → [× w] → z₁ → [+ b] → ŷ → [- y] → e → [²] → e² → [mean] → L

    Backward pass (chain rule):
        ∂L/∂w = (2/N) · Σ(ŷᵢ - yᵢ) · xᵢ
        ∂L/∂b = (2/N) · Σ(ŷᵢ - yᵢ)
    """

    def __init__(self):
        # Khởi tạo tham số ngẫu nhiên
        np.random.seed(42)
        self.w = ParameterNode(init_value=np.random.randn() * 0.1, name="w")
        self.b = ParameterNode(init_value=np.random.randn() * 0.1, name="b")

        # Khởi tạo các node tính toán
        self.mul_node = MultiplyNode(name="Multiply (w·x)")
        self.add_node = AddNode(name="Add (+b)")
        self.sub_node = SubtractNode(name="Subtract (ŷ-y)")
        self.sq_node = SquareNode(name="Square (e²)")
        self.mean_node = MeanNode(name="Mean (MSE)")

        # Lưu lịch sử huấn luyện
        self.loss_history = []
        self.w_history = []
        self.b_history = []

    def forward(self, x, y):
        """
        Forward pass qua đồ thị tính toán.

        x → [× w] → z₁ → [+ b] → ŷ → [- y] → e → [²] → e² → [mean] → L
        """
        z1 = self.mul_node.forward(x, self.w.output)      # z₁ = w * x
        y_hat = self.add_node.forward(z1, self.b.output)   # ŷ  = z₁ + b
        error = self.sub_node.forward(y_hat, y)            # e  = ŷ - y
        sq_error = self.sq_node.forward(error)             # e² = e²
        loss = self.mean_node.forward(sq_error)            # L  = mean(e²)
        return y_hat, loss

    def backward(self, x, y):
        """
        Backward pass - tính gradient bằng chain rule.

        ∂L/∂e² = 1/N           (từ MeanNode)
        ∂e²/∂e = 2e            (từ SquareNode)
        ∂e/∂ŷ  = 1             (từ SubtractNode)
        ∂ŷ/∂z₁ = 1             (từ AddNode)
        ∂ŷ/∂b  = 1             (từ AddNode)
        ∂z₁/∂w = x             (từ MultiplyNode)

        ⟹ ∂L/∂w = (1/N) · 2e · 1 · 1 · x = (2/N) · Σ(ŷ-y)·x
        ⟹ ∂L/∂b = (1/N) · 2e · 1 · 1     = (2/N) · Σ(ŷ-y)
        """
        N = len(x)
        y_hat = self.mul_node.output + self.b.output
        error = y_hat - y

        # Chain rule từ Loss → w, b
        dL_de2 = self.mean_node.backward()       # 1/N
        de2_de = self.sq_node.backward()          # 2 * error
        de_dyhat = 1.0                            # từ SubtractNode
        dyhat_dz1 = 1.0                           # từ AddNode
        dyhat_db = 1.0                            # từ AddNode
        dz1_dw = x                                # từ MultiplyNode

        # Tổng hợp gradient
        dL_de = dL_de2 * de2_de * de_dyhat        # (1/N) * 2 * error
        self.w.grad = np.sum(dL_de * dz1_dw)      # ∂L/∂w
        self.b.grad = np.sum(dL_de * dyhat_db)     # ∂L/∂b

        return self.w.grad, self.b.grad

    def update_params(self, lr):
        """Cập nhật tham số bằng Gradient Descent."""
        self.w.output -= lr * self.w.grad
        self.b.output -= lr * self.b.grad

    def train(self, X, y, lr=0.01, epochs=1000, verbose=True):
        """Huấn luyện mô hình."""
        for epoch in range(epochs):
            # Forward
            y_hat, loss = self.forward(X, y)

            # Backward
            self.backward(X, y)

            # Update
            self.update_params(lr)

            # Lưu lịch sử
            self.loss_history.append(loss)
            self.w_history.append(self.w.output)
            self.b_history.append(self.b.output)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:>4d}/{epochs} | Loss = {loss:.6f} | w = {self.w.output:.4f} | b = {self.b.output:.4f}")

        return self.w.output, self.b.output

    def predict(self, X):
        """Dự đoán điểm cuối kỳ."""
        return self.w.output * X + self.b.output

    def r_squared(self, X, y):
        """Tính hệ số R²."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def export_training_data(self, X, y):
        """Xuất dữ liệu huấn luyện sang JSON cho web demo."""
        data = {
            "dataset": {
                "midterm": X.tolist(),
                "final": y.tolist()
            },
            "result": {
                "w": float(self.w.output),
                "b": float(self.b.output),
                "mse": float(self.loss_history[-1]),
                "r_squared": float(self.r_squared(X, y))
            },
            "history": {
                "loss": self.loss_history,
                "w": self.w_history,
                "b": self.b_history
            }
        }
        output_path = Path(__file__).parent / "training_data.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ Đã xuất dữ liệu huấn luyện ra: {output_path}")
        return data


# ============================================================
# 3. TRỰC QUAN HÓA
# ============================================================

def plot_computational_graph(save_path="computational_graph.png"):
    """Vẽ sơ đồ đồ thị tính toán (Forward + Backward)."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-3, 5)
    ax.axis("off")
    ax.set_title("Đồ thị Tính toán - Hồi quy Tuyến tính\n(Computational Graph - Linear Regression)",
                 fontsize=14, fontweight="bold", pad=20)

    # Node definitions: (x, y, label, color)
    nodes = [
        (0, 3, "x\n(midterm)", "#4FC3F7"),
        (0, 1, "w", "#FFB74D"),
        (3, 2, "×", "#81C784"),
        (0, -1, "b", "#FFB74D"),
        (6, 1, "+", "#81C784"),
        (9, 1, "ŷ\n(pred)", "#CE93D8"),
        (9, 3, "y\n(final)", "#4FC3F7"),
        (11, 2, "−", "#81C784"),
        (13, 2, "( )²", "#81C784"),
        (15, 2, "mean", "#81C784"),
        (15, 0, "L\n(MSE)", "#EF5350"),
    ]

    # Draw nodes
    for (nx, ny, label, color) in nodes:
        circle = plt.Circle((nx, ny), 0.7, color=color, ec="black", lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(nx, ny, label, ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

    # Forward edges (blue)
    forward_edges = [
        (0, 3, 3, 2), (0, 1, 3, 2),   # x,w → ×
        (3, 2, 6, 1),                    # × → +
        (0, -1, 6, 1),                   # b → +
        (6, 1, 9, 1),                    # + → ŷ
        (9, 1, 11, 2),                   # ŷ → −
        (9, 3, 11, 2),                   # y → −
        (11, 2, 13, 2),                  # − → ()²
        (13, 2, 15, 2),                  # ()² → mean
        (15, 2, 15, 0),                  # mean → L
    ]

    for (x1, y1, x2, y2) in forward_edges:
        ax.annotate("", xy=(x2 - 0.7, y2), xytext=(x1 + 0.7, y1),
                     arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2))

    # Backward annotations (red, dashed)
    backward_labels = [
        (15, -0.8, "∂L/∂L = 1"),
        (13, 0.5, "∂L/∂e² = 1/N"),
        (11, 0.5, "∂L/∂e = 2e/N"),
        (9, -0.5, "∂L/∂ŷ = 2e/N"),
        (6, -0.5, "∂L/∂z₁ = 2e/N"),
        (0, -2.2, "∂L/∂b = Σ2e/N"),
        (0, -0.1, ""),
        (3, 0.3, "∂L/∂w = Σ2ex/N"),
    ]
    for (bx, by, label) in backward_labels:
        if label:
            ax.text(bx, by, label, ha="center", va="center", fontsize=7,
                    color="#C62828", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", edgecolor="#C62828", alpha=0.8))

    # Legend
    fwd_patch = mpatches.Patch(color="#1565C0", label="Forward pass")
    bwd_patch = mpatches.Patch(color="#C62828", label="Backward pass (gradients)")
    inp_patch = mpatches.Patch(color="#4FC3F7", label="Input nodes")
    param_patch = mpatches.Patch(color="#FFB74D", label="Parameter nodes (w, b)")
    op_patch = mpatches.Patch(color="#81C784", label="Operation nodes")
    loss_patch = mpatches.Patch(color="#EF5350", label="Loss node")
    ax.legend(handles=[fwd_patch, bwd_patch, inp_patch, param_patch, op_patch, loss_patch],
              loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Saved: {save_path}")
    plt.close()


def plot_regression(X, y, model, save_path="regression_result.png"):
    """Vẽ scatter plot và đường hồi quy."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(X, y, alpha=0.5, s=30, color="#42A5F5", edgecolors="#1565C0", label="Dữ liệu thực tế")

    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.w.output * x_line + model.b.output
    ax.plot(x_line, y_line, color="#EF5350", linewidth=2.5,
            label=f"ŷ = {model.w.output:.4f}·x + {model.b.output:.4f}")

    r2 = model.r_squared(X, y)
    ax.set_xlabel("Điểm Giữa kỳ (Midterm)", fontsize=12)
    ax.set_ylabel("Điểm Cuối kỳ (Final)", fontsize=12)
    ax.set_title(f"Dự đoán Điểm Cuối kỳ từ Điểm Giữa kỳ\nR² = {r2:.4f}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Saved: {save_path}")
    plt.close()


def plot_loss_curve(loss_history, save_path="loss_curve.png"):
    """Vẽ đường cong loss qua các epoch."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(loss_history, color="#EF5350", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("Đường cong Loss trong quá trình huấn luyện", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Saved: {save_path}")
    plt.close()


# ============================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    print("=" * 60)
    print("  DỰ ĐOÁN ĐIỂM CUỐI KỲ BẰNG ĐỒ THỊ TÍNH TOÁN")
    print("  (Computational Graph - Linear Regression)")
    print("=" * 60)

    # --- Đọc dữ liệu ---
    data_path = Path(__file__).parent / "TRAIN2.xlsx"
    df = pd.read_excel(data_path)
    X = df["midterm"].values.astype(np.float64)
    y = df["final"].values.astype(np.float64)
    print(f"\n📂 Đọc dữ liệu: {len(X)} mẫu")
    print(f"   Midterm: mean={X.mean():.2f}, std={X.std():.2f}")
    print(f"   Final  : mean={y.mean():.2f}, std={y.std():.2f}")

    # --- Vẽ đồ thị tính toán ---
    print("\n🔧 Vẽ đồ thị tính toán...")
    base = Path(__file__).parent
    plot_computational_graph(str(base / "computational_graph.png"))

    # --- Khởi tạo và huấn luyện ---
    print("\n🚀 Bắt đầu huấn luyện...")
    model = LinearRegressionGraph()
    w, b = model.train(X, y, lr=0.001, epochs=2000, verbose=True)

    # --- Kết quả ---
    r2 = model.r_squared(X, y)
    print(f"\n{'=' * 60}")
    print(f"  KẾT QUẢ MÔ HÌNH")
    print(f"{'=' * 60}")
    print(f"  Công thức: ŷ = {w:.4f} · x + {b:.4f}")
    print(f"  MSE      : {model.loss_history[-1]:.6f}")
    print(f"  R²       : {r2:.4f}")
    print(f"{'=' * 60}")

    # --- Vẽ kết quả ---
    print("\n📊 Vẽ biểu đồ...")
    plot_regression(X, y, model, str(base / "regression_result.png"))
    plot_loss_curve(model.loss_history, str(base / "loss_curve.png"))

    # --- Xuất dữ liệu cho web demo ---
    print("\n📤 Xuất dữ liệu cho web demo...")
    model.export_training_data(X, y)

    # --- Ví dụ dự đoán ---
    print("\n🔮 Ví dụ dự đoán:")
    test_scores = [3.0, 5.0, 7.0, 8.5]
    for s in test_scores:
        pred = model.predict(s)
        print(f"   Midterm = {s:.1f} → Final (dự đoán) = {pred:.2f}")

    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()