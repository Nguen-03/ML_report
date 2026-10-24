## Cấu trúc thư mục dự án (Project Structure)

Mã nguồn được tổ chức theo mô hình mô-đun hóa, tách biệt giữa dữ liệu, mã nguồn xử lý và kết quả đầu ra. File kịch bản trung tâm `run_experiment.py` phối hợp toàn bộ quy trình từ dữ liệu thô đến việc xuất bảng biểu và hình vẽ.

```text
my_project/
├── data/                      # Thư mục chứa dữ liệu (Đã thêm vào .gitignore)
│   ├── raw/                   # Dữ liệu thô ban đầu (bất biến)
│   └── processed/             # Dữ liệu sau khi làm sạch và tiền xử lý
│
├──notebook/                   # chứa notebook
│
├── results/                   # Thư mục lưu trữ kết quả đầu ra
│   ├── tables/                # Lưu các file bảng biểu (.csv, .md, .tex)
│   └── figures/               # Lưu các file đồ thị, hình vẽ (.png, .pdf)
├── src/                       # Thư mục chứa toàn bộ mã nguồn lõi (Modules)
│   ├── __init__.py
│   ├── utils.py               # Các hàm bổ trợ (cố định hạt giống ngẫu nhiên, tạo thư mục)
│   ├── data_loader.py         # Đọc dữ liệu thô và phân chia tập dữ liệu
│   ├── models.py              # Định nghĩa cấu trúc mô hình hoặc thuật toán
│   ├── training.py            # Huấn luyện mô hình và thực hiện dự đoán
│   └── evaluation.py          # Tính toán số liệu và khởi tạo định dạng đầu ra
├── split_indices.json         # Lưu trữ chỉ số (index) cố định của 3 tập Train/Val/Test
├── .gitignore                 # Chặn đẩy thư mục data/ và các tệp tin tạm lên kho mã nguồn
├── requirements.txt           # Danh sách các thư viện và phiên bản cụ thể cần cài đặt
└── run_experiment.py          # 🚀 Kịch bản duy nhất điều khiển toàn bộ thí nghiệm (Orchestrator)
```
