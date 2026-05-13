# 🎓 Dự đoán Điểm Cuối Kỳ bằng Đồ thị Tính toán
> Phạm Quang Hữu Phước - 24521411

> **CS523 – Cấu trúc Dữ liệu và Giải thuật**  
> Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM

## 📋 Mô tả

Chương trình dự đoán **điểm cuối kỳ (final)** từ **điểm giữa kỳ (midterm)** sử dụng mô hình **Hồi quy Tuyến tính** triển khai bằng **Đồ thị Tính toán (Computational Graph)**.

### Mô hình

$$\hat{y} = w \cdot x + b$$

- **Forward pass**: tính giá trị dự đoán qua các node trong đồ thị
- **Backward pass**: tính gradient bằng chain rule (backpropagation)
- **Tối ưu**: Gradient Descent

### Kết quả

| Tham số | Giá trị |
|---------|---------|
| w (weight) | 0.9038 |
| b (bias) | 1.3081 |
| MSE | 0.1215 |
| R² | 0.9778 |

## 📂 Cấu trúc file

```
├── TRAIN2.xlsx          # Dataset: 515 mẫu (midterm, final)
├── src.py               # Python: huấn luyện mô hình bằng computational graph
├── index.html           # Web demo: trực quan hóa training pipeline
├── report.tex           # Báo cáo LaTeX thuyết minh phương pháp
└── README.md
```

## 🚀 Hướng dẫn chạy

### 1. Cài đặt thư viện

```bash
pip install numpy pandas matplotlib openpyxl
```

### 2. Chạy training (Python)

```bash
python src.py
```

Kết quả:
- `training_data.json` – dữ liệu huấn luyện cho web demo
- `computational_graph.png` – sơ đồ đồ thị tính toán
- `regression_result.png` – scatter plot + đường hồi quy
- `loss_curve.png` – đường cong loss

### 3. Chạy web demo

Mở file `index.html` trực tiếp trong trình duyệt, sau đó nhấn **▶ Train** để xem quá trình huấn luyện trực quan.

> **Lưu ý**: Chạy `src.py` trước để sinh `training_data.json`, web demo sẽ tự load dữ liệu này. Nếu không có file JSON, web demo vẫn chạy được với dữ liệu mẫu.

## 📐 Đồ thị Tính toán

```
x ──→ [× w] ──→ z₁ ──→ [+ b] ──→ ŷ ──→ [− y] ──→ e ──→ [²] ──→ e² ──→ [mean] ──→ L
```

**Gradient (Backward):**

```
∂L/∂w = (2/N) · Σ(ŷ − y) · x
∂L/∂b = (2/N) · Σ(ŷ − y)
```

## 🛠 Công nghệ

- **Python** + NumPy 
- **Matplotlib** cho trực quan hóa
- **HTML/CSS/JS** cho web demo
